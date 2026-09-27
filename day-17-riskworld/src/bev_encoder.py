"""BEV feature encoder.

RECONSTRUCTION NOTE: the paper's abstract only says visual BEV features are
combined with the risk/actor context -- it does not specify a backbone. This
module is this repo's own default: a small stride-preserving CNN (no explicit
"ResNet-style backbone" per se, just stacked conv+BN+ReLU blocks with a
residual skip) chosen to be CPU-trainable in minutes while keeping the H x W
grid resolution intact end-to-end so that later modules (risk field, flow,
occupancy) can stay pixel-aligned with the input BEV raster.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, padding="same" so H, W never change."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BEVEncoder(nn.Module):
    """Encodes a rasterized multi-channel BEV scene into a dense feature map.

    Input:  (B, C_in, H, W)  -- e.g. C_in=4 channels: road mask, lane mask,
            static obstacle mask, current-frame agent footprint raster.
    Output: (B, C_feat, H, W) -- same spatial resolution as the input grid, so
            every downstream module (actor scatter, risk field, flow field)
            operates on pixel-aligned tensors.
    """

    def __init__(self, in_channels: int, feat_channels: int):
        super().__init__()
        self.stem = ConvBlock(in_channels, feat_channels, kernel_size=3)
        self.block1 = ConvBlock(feat_channels, feat_channels, kernel_size=3)
        self.block2 = ConvBlock(feat_channels, feat_channels, kernel_size=3)
        # 1x1 projection for the residual skip is a no-op here since channel
        # counts already match, kept explicit for clarity / future resizing.
        self.skip_proj = nn.Conv2d(feat_channels, feat_channels, kernel_size=1)

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev: (B, C_in, H, W) float tensor.
        Returns:
            (B, C_feat, H, W) BEV feature map.
        """
        x = self.stem(bev)              # (B, C_feat, H, W)
        h = self.block1(x)              # (B, C_feat, H, W)
        h = self.block2(h)              # (B, C_feat, H, W)
        out = h + self.skip_proj(x)     # residual connection, (B, C_feat, H, W)
        return out
