# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 10:35
# @Author  : wujunwei
# @FileName: fine_tuning.py
# @Software: PyCharm
import math
import os
# 指定运行的GPU
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import datetime
import json
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

    # 优化器参数
    parser.add_argument('--lr', default=None, type=float, metavar='LR',
                        help='官方学习率随训练改变')
    parser.add_argument('--blr', default=1e-3, type=float, metavar='LR',
                        help='base lr,absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--warm_up_epochs', default=10, type=int,
                        help='模型学习率预热，模型从较小学习率开始随着训练迭代次数不断增加，有利于模型稳定和加快收敛速度')

    # 数据集参数
    parser.add_argument('--dataset_path', default='/home/wjw/dataset/cifar100',
                        help='数据集所在位置')
    parser.add_argument('--checkpoint_path', default='/home/wjw/mae_pretrain_vit_base.pth',
                        help='fine_tuning from checkpoint')
    # parser.add_argument('--checkpoint_path', default='',
    #                    help='fine_tuning from checkpoint')
    parser.add_argument('--output_dir', default='/home/wjw/finetuning_output_dir',
                        help='微调权重存储位置')
    parser.add_argument('--log_dir', default='/home/wjw/log_dir',
                        help='tensorboard日志存储位置')
    parser.add_argument('--random_seed', default=0, type=int,
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
    # 自动寻找高效算法(如最适合的卷积实现算法)，提高计算效率
    cudann.benchmark = True

    # 数据增强
    transformer_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(),
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

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )

    # 载入模型
    model = model_vit.__dict__[args.model](
        num_classes=args.num_classes,
        global_pool=args.global_pool
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

    # 优化器
    para = optim_factory.add_weight_decay(model_optim, args.weight_decay)
    optimizer = torch.optim.AdamW(para, lr=args.lr, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss()

    warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * (
            math.cos((epoch - args.warm_up_epochs) / (args.epochs - args.warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    # 训练
    start_time = time.time()
    #model.load_state_dict(torch.load('/home/wjw/output_dir/checkpoint-59.pth'))
    print('load pre_training model from: %s' % args.checkpoint_path)

    model.train()
    print('start finetuning for: %d epochs' % args.epochs)
    for epoch in range(args.epochs):
        train_loss = []
        for i, (datas, labels) in enumerate(data_loader_train):
            datas = datas.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(datas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
            # prediction = torch.max(outputs, dim=1)[1]
            if (i + 1) % 20 == 0:
                print('epoch:%d , lr:%5f, batch:%d, loss:%5f' % (
                    epoch + 1, args.lr, i + 1, sum(train_loss) / len(train_loss)))

    # 每微调一轮测试一次
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for j, (data, label) in enumerate(data_loader_val):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            predict = torch.max(output, dim=1)[1]
            total += labels.size(0)
            correct += (predict == label).sum()
        if j % 20 == 0:
            print("test accuracy：%5f" % (correct / total * 100), "%")

    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # 保存模型
    torch.save(model.state_dict(), './mae_base_finetune.pth')


if __name__ == '__main__':
    args = get_argparser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
