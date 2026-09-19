"""
Shared-weight convolutional camera backbone with a Bi-Level Routing
Attention (BRA) block applied at the final feature resolution.

Each of the 6 camera views is passed through the SAME backbone (weight
sharing across views, standard for multi-view BEV detectors) to produce a
(C, fH, fW) feature map. BRA is applied as a residual attention block on
top of the plain conv features -- this is the "routed backbone" half of
the architecture (the other half, SparseSpatialCrossAttention, does the
BEV lifting and lives in src/model.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import BiLevelRoutingAttention


class ConvBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        use_bra: bool = True,
        bra_region_grid: int = 2,
        bra_topk: int = 2,
        bra_heads: int = 4,
    ):
        super().__init__()
        C = base_channels
        # 3 stride-2 stages: image_size -> image_size/8 (e.g. 64 -> 8).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
        self.use_bra = use_bra
        if use_bra:
            self.bra = BiLevelRoutingAttention(
                dim=C, region_grid=bra_region_grid, topk=bra_topk, num_heads=bra_heads
            )
        self.out_channels = C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        feat = self.stem(x)  # (B, C, H/8, W/8)
        if self.use_bra:
            routed = self.bra(feat)
            feat = feat + routed  # residual: BRA refines, doesn't replace, conv features
        return feat
