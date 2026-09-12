"""
SurroundViewTokenizer -- turns 6 synchronized camera frames into one token
sequence with camera-identity and spatial-position embeddings.
"""
from typing import Tuple

import torch
import torch.nn as nn


class SurroundViewTokenizer(nn.Module):
    """
    Splits each of N_CAMS camera images into non-overlapping patches and
    projects them into the model's token dimension, then adds a learned
    camera-identity embedding and a learned spatial-position embedding so
    the transformer can distinguish "front-left, patch (2,3)" from
    "rear-right, patch (2,3)".
    """

    def __init__(
        self,
        n_cams: int = 6,
        img_size: Tuple[int, int] = (224, 400),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.n_cams = n_cams
        h, w = img_size
        assert h % patch_size == 0 and w % patch_size == 0, \
            "img_size must be divisible by patch_size"
        self.grid_h, self.grid_w = h // patch_size, w // patch_size
        self.n_patches_per_cam = self.grid_h * self.grid_w

        # Conv2d with stride == kernel_size is a standard "patchify" op:
        # it turns each patch_size x patch_size pixel block into one token.
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Learned embeddings, one row per camera / per spatial location.
        self.camera_embed = nn.Parameter(torch.zeros(1, n_cams, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.n_patches_per_cam, embed_dim)
        )
        nn.init.trunc_normal_(self.camera_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, N_cams, C, H, W]
        b, n, c, h, w = images.shape
        assert n == self.n_cams, f"expected {self.n_cams} cameras, got {n}"

        x = images.view(b * n, c, h, w)                  # [B*N, C, H, W]
        x = self.patch_embed(x)                           # [B*N, D, Hp, Wp]
        d = x.shape[1]
        x = x.flatten(2).transpose(1, 2)                  # [B*N, P, D]  P = Hp*Wp
        x = x.view(b, n, self.n_patches_per_cam, d)        # [B, N, P, D]

        x = x + self.camera_embed + self.pos_embed         # broadcast add
        x = x.reshape(b, n * self.n_patches_per_cam, d)    # [B, N*P, D]
        return x
