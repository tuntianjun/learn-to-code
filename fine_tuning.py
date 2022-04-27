# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 10:35
# @Author  : wujunwei
# @FileName: fine_tuning.py
# @Software: PyCharm
import math
import os
# 指定运行的GPU
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudann
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
import timm.optim.optim_factory as optim_factory

import model_vit
from utils.lr_sched import adjust_lr
from utils.position_embedding import interpolate_pos_embed


def get_argparser():
    '''微调超参数设置'''
    # 模型参数
    parser = argparse.ArgumentParser('mae fine_tuning')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='每个GPU的batch_size')
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--model', default='vit_base_patch16', type=str,
                        help='微调模型')
    parser.add_argument('--num_classes', default=100, type=int,
                        help='分类类别数')
    parser.add_argument('--global_pool', default=False)
    parser.add_argument('--drop_path', default=0.1)

    # 优化器参数
    parser.add_argument('--lr', default=None, type=float, metavar='LR',
                        help='lr_decay')
    parser.add_argument('--blr', default=1e-3, type=float, metavar='LR',
                        help='base lr,absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    # parser.add_argument('--lr_decay', type=float, default=1e-2,
    #                   help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='模型学习率预热，模型从较小学习率开始随着训练迭代次数不断增加，有利于模型稳定和加快收敛速度')

    # 数据集参数
    parser.add_argument('--dataset_path', default='/home/wjw/dataset/cifar100',
                        help='数据集所在位置')
    #parser.add_argument('--checkpoint_path', default='/data/wjw/mae_base_pretrain.pth',
    #                    help='fine_tuning from checkpoint')
    parser.add_argument('--checkpoint_path', default='/home/wjw/mae_base_pretrain.pth',
                        help='fine_tuning from checkpoint')
    parser.add_argument('--output_dir', default='/home/wjw/finetuning_output_dir',
                        help='微调权重存储位置')
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

    # mix_up数据增强
    # TODO
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

    # 微调阶段使用强数据增强
    transformer_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformer_val = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 自定义数据集
    dataset_train = datasets.ImageFolder(os.path.join(args.dataset_path, 'train'), transform=transformer_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.dataset_path, 'val'), transform=transformer_val)
    # print(dataset_train.class_to_idx)

    train_data_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_data_sampler = torch.utils.data.RandomSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=train_data_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=test_data_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )

    # 载入模型
    model = model_vit.__dict__[args.model](
        num_classes=args.num_classes,
        global_pool=args.global_pool,
        # drop_path=args.drop_path
    )
    model_optim = model
    model.to(device)

    # 模型大小
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_optim))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # 分布式训练设置,以及分布式使用多张GPU
    # TODO

    # 学习率设置
    if args.lr is None:
        args.lr = args.blr * args.batch_size / 256

    # finetune 阶段使用 layer-wise learning rate decay

    # finetune 位置编码插值
    # TODO
    interpolate_pos_embed()

    # 优化器
    para = optim_factory.add_weight_decay(model_optim, args.weight_decay)
    optimizer = torch.optim.AdamW(para, lr=args.lr, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss()

    def train(model_train, epoch):
        model_train.train()
        train_loss = []
        for i, (datas, labels) in enumerate(data_loader_train):
            lr = adjust_lr(optimizer, i / len(data_loader_train) + epoch, args)
            datas = datas.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model_train(datas)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # prediction = torch.max(outputs, dim=1)[1]
            if (i + 1) % 20 == 0:
                print('epoch:%d , lr:%5f, batch:%d, loss:%5f' % (
                    epoch + 1, lr, i + 1, sum(train_loss) / len(train_loss)))

    def val(model_val):
        model_val.eval()
        correct = 0
        total = 0
        val_loss = []
        with torch.no_grad():
            for j, (data, label) in enumerate(data_loader_val):
                data = data.to(device)
                label = label.to(device)
                with torch.cuda.amp.autocast():
                    output = model_val(data)
                    loss = criterion(output, label)
                predict = torch.max(output, dim=1)[1]
                val_loss.append(loss.item())
                total += label.size(0)
                correct += (predict == label).sum()
            print('val loss :%5f,' % (sum(val_loss)/len(val_loss)))
            print("val accuracy：%5f " % (correct / total * 100), "%")

    model.load_state_dict(torch.load(args.checkpoint_path), strict=False)
    print('load pre_training model from: %s' % args.checkpoint_path)
    for epoch in range(args.epochs):
        start_time = time.time()
        train(model, epoch)

        if epoch == args.epochs:
            torch.save(model.state_dict(), '/data/wjw/mae_bae_finetune.pth')

        mid_time = time.time()
        temp1 = str(datetime.timedelta(seconds=int(mid_time-start_time)))
        print('Training time for one epoch{}'.format(temp1))

        val(model)
        end_time = time.time()
        temp2 = str(datetime.timedelta(seconds=int(end_time - mid_time)))
        print('Val time for one epoch{}'.format(temp2))
'''
    # 训练
    start_time = time.time()
    model.load_state_dict(torch.load(args.checkpoint_path))
    print('load pre_training model from: %s' % args.checkpoint_path)

    model.train()
    print('start finetune for: %d epochs' % args.epochs)
    for epoch in range(args.epochs):
        train_loss = []
        lr = adjust_lr(optimizer, epoch, args)
        for i, (datas, labels) in enumerate(data_loader_train):
            datas = datas.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(datas)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # prediction = torch.max(outputs, dim=1)[1]
            if (i + 1) % 20 == 0:
                print('epoch:%d , lr:%5f, batch:%d, loss:%5f' % (
                    epoch + 1, lr, i + 1, sum(train_loss) / len(train_loss)))

    # 保存模型
    torch.save(model.state_dict(), './mae_base_finetune.pth')
    # 载入模型
    model.load_state_dict(torch.load('./mae_base_finetune.pth'))

    # 验证
    model.eval()
    correct = 0
    total = 0
    # val_loss = []
    with torch.no_grad():
        for j, (data, label) in enumerate(data_loader_val):
            data = data.to(device)
            label = label.to(device)
            with torch.cuda.amp.autocast():
                output = model(data)
            predict = torch.max(output, dim=1)[1]
            total += label.size(0)
            correct += (predict == label).sum()
        print("test accuracy：%5f" % (correct / total * 100), "%")

    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
'''
if __name__ == '__main__':
    args = get_argparser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
