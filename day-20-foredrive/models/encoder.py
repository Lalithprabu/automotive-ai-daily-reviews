"""
Visual encoder — the "shared online encoder" in ForeDrive's terminology.

Per the abstract: "planning gradients update the shared online encoder, while
stop-gradient routing trains the latent predictor with forecasting losses
only." This module IS that shared online encoder: a small CNN that turns a
front-view raster (here, our synthetic ego-centric occupancy proxy, see
src/data/synthetic_dataset.py) into a grid of visual tokens.

RECONSTRUCTION DISCLOSURE: the ForeDrive paper does not publish the exact
encoder architecture (backbone, depth, channel widths) at the abstract level
we have access to (see SOURCING.md). This is a small from-scratch CNN sized
for a 64x64 synthetic raster and CPU training, standing in for whatever
visual backbone the real paper uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisualEncoder(nn.Module):
    """Encodes a (B, C_in, H, W) raster into a grid of (B, N, token_dim) tokens.

    Architecture: a 4-stage strided CNN downsamples H x W -> tokens_per_side x
    tokens_per_side, doubling channels at each stage, followed by a 1x1
    projection to `token_dim`. The resulting spatial grid is flattened into a
    token sequence (N = tokens_per_side ** 2), matching the "grid of visual
    tokens" framing used by ViT-style / DiT-style conditioning.
    """

    def __init__(self, in_channels: int, raster_size: int, tokens_per_side: int, token_dim: int):
        super().__init__()
        assert raster_size % tokens_per_side == 0, (
            f"raster_size ({raster_size}) must be divisible by tokens_per_side ({tokens_per_side})"
        )
        self.tokens_per_side = tokens_per_side
        self.token_dim = token_dim
        num_stages = int(torch.log2(torch.tensor(raster_size // tokens_per_side)).item())
        assert 2 ** num_stages == raster_size // tokens_per_side, (
            "raster_size / tokens_per_side must be a power of 2 for this simple strided-conv stack"
        )

        channels = [in_channels]
        c = 16
        for _ in range(num_stages):
            channels.append(c)
            c *= 2

        layers = []
        for i in range(num_stages):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(num_groups=min(4, channels[i + 1]), num_channels=channels[i + 1]))
            layers.append(nn.GELU())
        self.stem = nn.Sequential(*layers)

        # 1x1 conv projects the final CNN feature map to the token embedding dim
        # used everywhere downstream (world model, fusion, DiT conditioning).
        self.to_tokens = nn.Conv2d(channels[-1], token_dim, kernel_size=1)

    def forward(self, raster: torch.Tensor) -> torch.Tensor:
        """
        raster: (B, C_in, raster_size, raster_size)
        returns: (B, N, token_dim) where N = tokens_per_side ** 2
        """
        b = raster.shape[0]
        feat = self.stem(raster)                       # (B, C_last, tokens_per_side, tokens_per_side)
        tokens = self.to_tokens(feat)                    # (B, token_dim, tokens_per_side, tokens_per_side)
        tokens = tokens.flatten(2).transpose(1, 2)         # (B, N, token_dim)
        assert tokens.shape[1] == self.tokens_per_side ** 2
        return tokens
