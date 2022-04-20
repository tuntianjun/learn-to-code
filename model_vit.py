# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 10:34
# @Author  : wujunwei
# @FileName: model_vit.py
# @Software: PyCharm


import torch
import torch.nn as nn
import timm.models.vision_transformer
from functools import partial


class Vit(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, global_pool=False, **kwargs):
        super(Vit, self).__init__(**kwargs)

        # 最终的分类可选用 global_pool
        # self.num_classes = num_classes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            # 移除原有的norm
            del self.norm

    def forward_feature(self, x):
        # b:batch size
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            # 没有class token，对所有输出的token求均值
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            # 只取class token
            x = self.norm(x)
            outcome = x[:, 0]

        # 对于class token
        # outcome = nn.Linear(outcome.shape[-1], self.num_classes)
        return outcome


# 关于 functools.partial(func, *args, **keywords):
# func: 需要被扩展的函数，返回的函数其实是一个类 func 的函数
# *args: 需要被固定的位置参数
# **kwargs: 需要被固定的关键字参数
# 如果在原来的函数 func 中关键字不存在，将会扩展，如果存在，则会覆盖

def vit_base_patch16(**kwargs):
    model = Vit(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = Vit(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = Vit(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
