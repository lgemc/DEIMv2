"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DINOv3 (https://github.com/facebookresearch/dinov3)

Copyright (c) Meta Platforms, Inc. and affiliates.

This software may be used and distributed in accordance with
the terms of the DINOv3 License Agreement.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer


class SpatialPriorModulev2(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()

        # 1/4
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        # 1/8
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(2 * inplanes),
            ]
        )
        # 1/16
        self.conv3 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )
        # 1/32
        self.conv4 = nn.Sequential(
            *[
                nn.GELU(),
                nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(4 * inplanes),
            ]
        )

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)     # 1/8
        c3 = self.conv3(c2)     # 1/16
        c4 = self.conv4(c3)     # 1/32

        return c2, c3, c4


@register()
class DINOv3STAs(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_sta=True,
        conv_inplane=16,
        hidden_dim=None,
        dtype='float32',
    ):
        super(DINOv3STAs, self).__init__()
        self._backbone_dtype = self._parse_dtype(dtype)
        if 'dinov3' in name:
            checkpoint = None
            requires_local_cls_norm = False
            requires_cls_norm = False
            if weights_path is not None:
                print(f'Loading DINOv3 checkpoint from {weights_path}...')
                checkpoint = torch.load(weights_path, map_location='cpu')
                requires_local_cls_norm = any(k.startswith('local_cls_norm') for k in checkpoint.keys())
                requires_cls_norm = any(k.startswith('cls_norm') for k in checkpoint.keys())

            self.dinov3 = torch.hub.load('./dinov3', name, source='local', pretrained=False)

            if requires_cls_norm and getattr(self.dinov3, 'cls_norm', None) is None:
                self.dinov3.cls_norm = copy.deepcopy(self.dinov3.norm)
                self.dinov3.untie_cls_and_patch_norms = True

            if requires_local_cls_norm and getattr(self.dinov3, 'local_cls_norm', None) is None:
                self.dinov3.local_cls_norm = copy.deepcopy(self.dinov3.norm)
                self.dinov3.untie_global_and_local_cls_norm = True

            if checkpoint is not None:
                self.dinov3.load_state_dict(checkpoint, strict=True)
            while len(self.dinov3.blocks) != (interaction_indexes[-1] + 1):
                del self.dinov3.blocks[-1]
            del self.dinov3.head
        else:
            self.dinov3 =  VisionTransformer(embed_dim=embed_dim, num_heads=num_heads)
            if weights_path is not None:
                print(f'Loading ckpt from {weights_path}...')
                checkpoint = torch.load(weights_path)
                self.dinov3._model.load_state_dict(checkpoint)
            else:
                print('Training ViT-Tiny from scratch!')

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)
        else:
            if self._backbone_dtype != torch.float32:
                print('[DINOv3STAs] Requested reduced precision dtype is ignored when finetune=True.')
                self._backbone_dtype = torch.float32

        if self._backbone_dtype != torch.float32:
            print(f'Casting frozen DINOv3 backbone to {self._backbone_dtype} for reduced memory footprint.')
            self.dinov3.to(dtype=self._backbone_dtype)

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def forward(self, x):
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        backbone_in = x if x.dtype == self._backbone_dtype else x.to(self._backbone_dtype)

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                backbone_in, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(backbone_in)

        if len(all_layers) == 1:    # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]
        
        proj_dtype = self.convs[0].weight.dtype
        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            if sem_feat.dtype != proj_dtype:
                sem_feat = sem_feat.to(dtype=proj_dtype)
            sem_feats.append(sem_feat)

        # fusion
        fused_feats = []

        if self.use_sta:
            sta_input = x if x.dtype == proj_dtype else x.to(dtype=proj_dtype)
            detail_feats = self.sta(sta_input)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                if detail_feat.dtype != proj_dtype:
                    detail_feat = detail_feat.to(dtype=proj_dtype)
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4

    @staticmethod
    def _parse_dtype(dtype):
        if isinstance(dtype, torch.dtype):
            return dtype
        if dtype is None:
            return torch.float32

        if not isinstance(dtype, str):
            raise TypeError(f'Unsupported dtype specification: {dtype}')

        key = dtype.lower()
        mapping = {
            'float32': torch.float32,
            'fp32': torch.float32,
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }
        if key not in mapping:
            raise ValueError(f'Unknown dtype {dtype}. Supported: {list(mapping.keys())}')
        return mapping[key]
