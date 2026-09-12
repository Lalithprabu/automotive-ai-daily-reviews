"""
Minimal per-camera preprocessing transforms for surround-view clips.

Kept dependency-free (no torchvision requirement) so the reference repo
runs with just `torch` installed. Swap in torchvision.transforms.v2 if you
already depend on it in your training stack.
"""
from typing import Sequence

import torch
import torch.nn.functional as F


def resize_and_normalize(
    images: torch.Tensor,
    target_size: Sequence[int] = (224, 400),
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    images: [N_cams, C, H, W] or [B, N_cams, C, H, W], float in [0, 1]
    Returns: same rank, resized to target_size and channel-normalized.
    """
    orig_shape = images.shape
    flat = images.reshape(-1, *orig_shape[-3:])              # [-, C, H, W]
    flat = F.interpolate(flat, size=tuple(target_size), mode="bilinear", align_corners=False)

    mean_t = torch.tensor(mean, device=images.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=images.device).view(1, -1, 1, 1)
    flat = (flat - mean_t) / std_t

    return flat.reshape(*orig_shape[:-2], *target_size)
