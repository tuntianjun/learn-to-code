# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 10:34
# @Author  : wujunwei
# @FileName: pre_training.py
# @Software: PyCharm

import os

import time

# 指定运行的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm.optim.optim_factory as optim_factory

import model_mae
from utils.lr_sched import adjust_lr
import numpy as np

from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.cuda.amp import GradScaler

torch.cuda.empty_cache()

def get_argparser():
    '''预训练超参数设置'''
    # 模型参数
    parser = argparse.ArgumentParser('mae pre_training')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='每个GPU的batch_size')
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--model', default='mae_vit_base_patch16',
                        help='预训练模型')
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='对像素归一化来计算loss，有利于提点')
    parser.set_defaults(norm_pix_loss=True)

    # 优化器参数
    parser.add_argument('--lr', default=None, type=float, metavar='LR',
                        help='官方学习率随训练改变')
    parser.add_argument('--blr', default=1e-3, type=float, metavar='LR',
                        help='base lr,absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='模型学习率预热，模型从较小学习率开始随着训练迭代次数不断增加，有利于模型稳定和加快收敛速度')

    # 数据集参数
    parser.add_argument('--dataset_path', default='/home/wjw/dataset/cifar100',
                        help='数据集所在位置')
    parser.add_argument('--output_dir', default='/home/wjw/output_dir',
                        help='预训练权重存储位置')
    parser.add_argument('--log_dir', default='/home/wjw/log_dir',
                        help='tensorboard日志存储位置')
    parser.add_argument('--seed', default=0, type=int,
                        help='设置随机种子，保证实验结果可复现')
    parser.add_argument('--resume', default='', help='从检查点恢复')

    # 设备参数
    parser.add_argument('--device', default='cuda', help='默认使用GPU训练')
    parser.add_argument('--pin_memory', action='store_true',
                        help='创建DataLoader时设置为True可以使内存中tensor转化到GPU显存更迅速')
    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--no_pin_memory', action='store_false')
    parser.set_defaults(pin_memory=True)

    # 多张GPU上分布式训练参数 TODO

    return parser


def main(args):
    '''预训练主函数'''
    device = torch.device(args.device)

    # 固定随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 自动寻找高效算法(如最适合的卷积实现算法)，提高计算效率
    cudnn.benchmark = True

    # 数据增强
    transformer_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 自定义数据集
    dataset_train = datasets.ImageFolder(os.path.join(args.dataset_path, 'train'), transform=transformer_train)
    # print(dataset_train.class_to_idx)

    # 随机采样
    train_data_sampler = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=train_data_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        # 丢弃最后不足batch size大小的数据
        drop_last=True
    )

    # 载入模型
    model = model_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model_optim = model
    model.to(device)

    # 模型大小
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model_optim))
    print('Number of MAE pretrain params (M): %.2f' % (n_parameters / 1.e6))

    # 分布式训练设置,以及分布式使用多张GPU TODO

    # 学习率设置
    if args.lr is None:
        args.lr = args.blr * args.batch_size / 256
    # 优化器
    para = optim_factory.add_weight_decay(model_optim, args.weight_decay)
    optimizer = torch.optim.AdamW(para, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)

    # loss_scaler = GradScaler()
    loss_scaler = NativeScaler()

    # 训练
    model.train()
    print('Pretrain for %d epochs, warmup for %d epochs.' % (args.epochs, args.warmup_epochs))
    print('Learning rate %5f' % args.lr)
    start_time = time.time()

    for epoch in range(args.epochs):
        train_loss = []
        temp1 = time.time()

        optimizer.zero_grad()

        for i, (datas, labels) in enumerate(data_loader_train):
            '''
            adamw 优化器 + warm_up + sincos lr
            结果：loss 先降再升-循环 --> 最终原因：autocast()要和GradScaler()一起使用计算混合精度
                 似乎是混合精度的问题：
                 (1)加上loss_scaler = GradScaler()后可以正常下降，但下降幅度和速度远不如官方源码,60轮到0.07收敛
                 (2)去掉源码的train one epoch 2-->0.3-->0.22-->0.18
                 (3)使用源码的NativeScaler(),unscale gradient loss的变化一致
                 (4)会不会是loss表示方式有区别？看训练出来的模型微调后的结果？
                 (5)注释掉源码中关于分布式的内容对于loss的计算没有任何影响
            '''

            lr = adjust_lr(optimizer, i / len(data_loader_train) + epoch, args)
            datas = datas.to(device)

            # 混合精度计算
            # 具体使用在template/pytorch混合精度计算.py
            with torch.cuda.amp.autocast():
                loss, _, _ = model(datas, mask_ratio=args.mask_ratio)

            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(i + 1) % 1 == 0)
            optimizer.zero_grad()

            # 梯度放大
            # loss_scaler.scale(loss).backward()
            # scaler.step() 首先把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
            # loss_scaler.step(optimizer)
            # 看是否要增大scaler
            # loss_scaler.update()

            train_loss.append(loss.item())

            if (i+1) % 20 == 0:
                print('epoch:%d , lr:%5f,  batch:%d, loss:%5f' % (
                    epoch + 1, lr, i + 1, sum(train_loss) / len(train_loss)))

        temp2 = time.time()
        temp_time = temp2 - temp1
        temp = str(datetime.timedelta(seconds=int(temp_time)))
        print('Training 1 epoch time {}'.format(temp))

    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('------Pretrain Done!------')
    print('Training time {}'.format(total_time_str))

    # 保存模型
    torch.save(model.state_dict(), './mae_base_pretrain.pth')


if __name__ == '__main__':
    args = get_argparser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
