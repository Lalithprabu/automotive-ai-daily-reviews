"""
Visual Attention Module (VAM) — carried over from the base TrajFusionNet
(arXiv:2508.19866, verified from its full text this cycle). Two parallel
large-kernel-attention (LKA) CNN branches: one sees the first observed frame
overlaid with the OBSERVED bounding-box trajectory, the other sees the last
observed frame overlaid with the PREDICTED bounding-box trajectory (from
SAM's trajectory transformer). Outputs are concatenated and projected.

The base paper uses a pretrained VAN-B2 backbone (Guo et al., 2022,
"Visual Attention Network") per branch. This module reimplements VAN's core
mechanic — large-kernel attention, i.e. a depthwise conv + depthwise-dilated
conv + 1x1 conv approximating a large receptive-field spatial attention map
that reweights the feature map — from scratch in a lightweight form, so the
repo is self-contained and trainable without downloading external
ImageNet-pretrained weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class VAMConfig:
    in_channels: int = 3        # RGB frame with bbox overlay drawn on it
    base_channels: int = 32
    num_stages: int = 3         # downsampling stages, channels double each stage
    proj_dim: int = 40
    dropout: float = 0.1


class LargeKernelAttention(nn.Module):
    """LKA(X) = Conv_1x1( DWDilatedConv( DWConv(X) ) );  output = X * LKA(X)

    Approximates a large receptive field (depthwise 5x5 + depthwise-dilated
    7x7 @ dilation 3 covers ~23x23) with far fewer parameters than a dense
    large-kernel or a full self-attention map — the mechanism VAN uses in
    place of standard ViT self-attention for vision backbones.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dw_dilated_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=9, groups=channels, dilation=3
        )
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        attn = self.dw_conv(x)
        attn = self.dw_dilated_conv(attn)
        attn = self.pw_conv(attn)
        return x * attn  # element-wise spatial re-weighting, same shape as input


class LKABlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.lka = LargeKernelAttention(channels)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

        self.norm2 = nn.BatchNorm2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj_in(self.norm1(x)))
        h = self.lka(h)
        h = self.proj_out(h)
        x = x + h                       # residual around the attention sub-block
        x = x + self.ffn(self.norm2(x))  # residual around the FFN sub-block
        return x


class VisualAttentionBranch(nn.Module):
    """One VAN-style backbone: patchify-downsample -> stack of LKABlocks ->
    global-average-pool, repeated `num_stages` times with doubling channels."""

    def __init__(self, cfg: VAMConfig):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.base_channels, kernel_size=4, stride=4),
            nn.BatchNorm2d(cfg.base_channels),
        )
        stages = []
        ch = cfg.base_channels
        for _ in range(cfg.num_stages):
            stages.append(LKABlock(ch, cfg.dropout))
            stages.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2),  # downsample + widen
                nn.BatchNorm2d(ch * 2),
            ))
            ch *= 2
        self.stages = nn.Sequential(*stages)
        self.out_channels = ch
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)  # [B, out_channels]
        return x


class VisualAttentionModule(nn.Module):
    def __init__(self, cfg: VAMConfig | None = None):
        super().__init__()
        self.cfg = cfg or VAMConfig()
        self.observed_branch = VisualAttentionBranch(self.cfg)
        self.predicted_branch = VisualAttentionBranch(self.cfg)
        fused_dim = self.observed_branch.out_channels + self.predicted_branch.out_channels
        self.proj = nn.Linear(fused_dim, self.cfg.proj_dim)

    def forward(self, observed_frame: torch.Tensor, predicted_frame: torch.Tensor) -> torch.Tensor:
        # observed_frame / predicted_frame: [B, 3, H, W] (bbox overlays pre-rendered)
        f_obs = self.observed_branch(observed_frame)        # [B, C]
        f_pred = self.predicted_branch(predicted_frame)      # [B, C]
        fused = torch.cat([f_obs, f_pred], dim=-1)            # [B, 2C]
        return self.proj(fused)                               # [B, proj_dim]
