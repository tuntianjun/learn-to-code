# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 10:34
# @Author  : wujunwei
# @FileName: model_mae.py
# @Software: PyCharm

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from functools import partial
from utils.position_embedding import get_2d_sincos_pos_embed


class Mae(nn.Module):
    # 关于norm_pix_loss：计算MSE loss使用
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # mae encoder:默认vit_base
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 初始化class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        # 初始化位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim), requires_grad=False)

        # vit的transformer blocks
        self.blocks = nn.ModuleList([
            # Block(embed_dim,num_heads,mlp_ratio,qkv_bias=True,qk_scale=None,norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # mae decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # decoder的位置编码
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            # Block(decoder_embed_dim,decoder_num_heads,mlp_ratio,qkv_bias=True,qk_scale=None,norm_layer=norm_layer)
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # decoder输出维度等于每个patch的像素个数
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_channels, bias=True)

        self.norm_pix_loss = norm_pix_loss

        '''不进行2d初始化结果怎么样？'''
        '''没有位置编码和网络参数的初始化，预训练模型微调结果很差'''
        self.initialize_weights()

    # 位置编码的权重初始化
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    # 网络参数的初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # patch变为特征，用于原图patch和生成的patch进行MSE loss的计算
    def patchify(self, imgs):
        '''从（b,3,w,h）--> （b,l,patch_size**2*3）'''
        patch_size = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

        h = w = imgs.shape[2] // patch_size
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
        x = torch.einsum('bchpwq->bhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, patch_size ** 2 * 3))
        return x

    def unpatchify(self, x):
        '''从x还原回patch'''
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, 3))
        x = torch.einsum('bhwpqc->bchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * patch_size, h * patch_size))
        return imgs

    # 随机shuffle并还原
    def random_mask(self, x, mask_ratio):
        '''核心部分'''
        batch_size, length, dim = x.shape
        # 未被mask的长度
        length_keep = int(length * (1 - mask_ratio))

        noise = torch.rand(batch_size, length, device=x.device)

        # 对产生的数据按从小到大排序，将此顺序作为shuffle的结果
        # 输出的是排序后的索引
        id_shuffle = torch.argsort(noise, dim=1)

        # 还原索引
        id_unshuffle = torch.argsort(id_shuffle, dim=1)

        # 未被mask的部分索引
        id_keep = id_shuffle[:, :length_keep]
        x_mask = torch.gather(x, dim=1, index=id_keep.unsqueeze(-1).repeat(1, 1, dim))

        # 生成二值矩阵，0表示未被mask
        mask = torch.ones([batch_size, length], device=x.device)
        mask[:, :length_keep] = 0

        # decoder输入时还原顺序
        mask = torch.gather(mask, dim=1, index=id_unshuffle)

        return x_mask, mask, id_unshuffle

    # 前向过程
    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        # 位置编码跳过第一个class token的位置
        x = x + self.pos_embed[:, 1:, :]

        x, mask, id_unshuffle = self.random_mask(x, mask_ratio)

        # 添加class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # 在batch维度扩充
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 仅对mask部分使用encoder
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, id_unshuffle

    def forward_decoder(self, x, id_unshuffle):
        # 和encoder前向过程类似
        x = self.decoder_embed(x)

        # 恢复输入顺序，并将mask部分加入到序列中
        mask_tokens = self.mask_token.repeat(x.shape[0], id_unshuffle.shape[1] + 1 - x.shape[1], 1)
        # 跳过class token
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=id_unshuffle.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # 加上class token
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # 加上位置编码
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # 输出维度等于patch像素个数
        x = self.decoder_pred(x)

        # 重建移除class token
        x = x[:, 1:, :]

        return x

    # MSE loss
    def forward_loss(self, imgs, pred, mask):
        '''这里的pred 等价于decoder的x'''
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            # 均值
            mean = target.mean(dim=-1, keepdim=True)
            # 方差
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5

        # loss的计算
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        '''mae总体前向过程'''
        latent, mask, id_unshuffle = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, id_unshuffle)
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask


def mae_vit_base_patch16(**kwargs):
    model = Mae(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = Mae(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = Mae(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
