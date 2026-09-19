"""
BEVEncoder -- shared bird's-eye-view feature extractor.

Paper role: DRiF trains a single shared BEV feature F_t that feeds THREE
parallel heads (static map segmentation, dynamic risk field, planning),
rather than three independently-encoded branches. This module produces
that shared F_t.

Reconstruction default (backbone width/depth not specified in the paper):
a small conv stem + two downsampling stages + a self-attention bottleneck.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU. Basic building block for the stem/downsampler."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SelfAttentionBottleneck(nn.Module):
    """Multi-head self-attention over the flattened spatial grid of the BEV
    feature map, giving every cell a global receptive field before the three
    heads branch off. Reconstruction default: standard pre-norm MHSA block
    with a small MLP, applied once at the bottleneck resolution.

    Input / Output: [B, C, H, W] -> [B, C, H, W]  (shape preserving)
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out

        tokens = tokens + self.mlp(self.norm2(tokens))

        out = tokens.transpose(1, 2).reshape(B, C, H, W)
        return out


class BEVEncoder(nn.Module):
    """Shared BEV encoder: conv stem -> 2x downsample -> self-attention bottleneck.

    Input:  bev_grid   # Shape: [B, C_in, G, G]   rasterized BEV occupancy/lane/motion channels
    Output: f_t         # Shape: [B, C_bottleneck, G/4, G/4]   shared feature map fed to all 3 heads
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 32,
        bottleneck_channels: int = 128,
        attn_heads: int = 4,
    ):
        super().__init__()
        self.stem = ConvBNAct(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Downsample x2: G -> G/2
        self.down1 = ConvBNAct(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)
        # Downsample x2 again: G/2 -> G/4
        self.down2 = ConvBNAct(base_channels * 2, bottleneck_channels, kernel_size=3, stride=2, padding=1)

        self.bottleneck = SelfAttentionBottleneck(bottleneck_channels, num_heads=attn_heads)

    def forward(self, bev_grid: torch.Tensor) -> torch.Tensor:
        x = self.stem(bev_grid)      # [B, base, G, G]
        x = self.down1(x)             # [B, base*2, G/2, G/2]
        x = self.down2(x)             # [B, bottleneck, G/4, G/4]
        f_t = self.bottleneck(x)      # [B, bottleneck, G/4, G/4]
        return f_t
