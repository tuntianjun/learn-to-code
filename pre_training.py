# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 10:34
# @Author  : wujunwei
# @FileName: pre_training.py
# @Software: PyCharm

import os
# 指定运行的GPU
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import argparse
import datetime
from pathlib import Path

import torch
import torch.backends.cudnn as cudann
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm.optim.optim_factory as optim_factory

import model_mae
from utils.lr_sched import adjust_lr
import numpy as np


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

    # 多张GPU上分布式训练参数
    # TODO

    return parser


def main(args):
    '''预训练主函数'''
    device = torch.device(args.device)

    # 固定随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 自动寻找高效算法(如最适合的卷积实现算法)，提高计算效率
    cudann.benchmark = True

    # 数据增强
    transformer_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 自定义数据集
    dataset_train = datasets.ImageFolder(os.path.join(args.dataset_path, 'train'), transform=transformer_train)
    print(dataset_train.class_to_idx)

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

    # 分布式训练设置,以及分布式使用多张GPU
    # TODO

    # 学习率设置
    if args.lr is None:
        args.lr = args.blr * args.batch_size / 256
    # 优化器
    para = optim_factory.add_weight_decay(model_optim, args.weight_decay)
    #optimizer = torch.optim.AdamW(para, lr=args.lr, betas=(0.9, 0.95))
    optimizer = torch.optim.SGD(para, lr=args.lr)

    # 训练
    model.train()
    print('pretrain for %d epochs' % args.epochs)
    print('learning rate %5f' % args.lr)
    start_time = time.time()

    optimizer.zero_grad()
    for epoch in range(args.epochs):
        train_loss = []
        temp1 = time.time()

        # lr = adjust_lr(optimizer, epoch, args)

        # MAE源码中每个epoch开始前都要先对优化器梯度置0
        '''对loss是否有影响？'''
        # optimizer.zero_grad()

        for i, (datas, labels) in enumerate(data_loader_train):
            '''
            1.adamw 优化器 + warm_up + sincos lr
            结果：loss 先降再升-循环
                 可能是陷入局部最优或是学习率太大
            '''
            '''
            2. adamw 优化器 + 固定学习率
            结果：loss收敛到一定程度就不再下降
            '''
            '''
            3. SGD 优化器 + warm_up + sincos lr 
            '''
            '''
            4. SGD 优化器 + 固定学习率
            '''
            lr = adjust_lr(optimizer, i / len(data_loader_train) + epoch, args)
            # optimizer = torch.optim.AdamW(para, lr=lr, betas=(0.9, 0.95))

            optimizer = torch.optim.SGD(para, lr=lr)

            datas = datas.to(device)
            # optimizer.zero_grad()

            # 混合精度计算
            with torch.cuda.amp.autocast():
                loss, _, _ = model(datas, mask_ratio=args.mask_ratio)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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
    print('Training time {}'.format(total_time_str))

    # 保存模型
    torch.save(model.state_dict(), './mae_base_pretrain_1.pth')


if __name__ == '__main__':
    args = get_argparser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
