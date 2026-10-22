"""ADE / FDE / Miss-Rate in metres (the paper's three metrics)."""
from __future__ import annotations

import numpy as np
import torch

from .data import SCALE


@torch.no_grad()
def predict(model, x: torch.Tensor, bs: int = 512) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + bs]) for i in range(0, len(x), bs)])


def trajectory_metrics(pred: torch.Tensor, gt: torch.Tensor, mr_step: int = 3, mr_thresh_m: float = 1.0):
    """pred, gt: (N, HOR, 2) in SCALE units (canonical frame; L2 is rotation-invariant).

    mr_step=3 -> t = 2.0 s (dt = 0.5 s).  Returns dict of floats in metres / fraction.
    """
    err = (pred - gt).norm(dim=-1) * SCALE  # (N, HOR) metres
    return {
        "ADE": err.mean().item(),
        "FDE": err[:, -1].mean().item(),
        "MR@2s/1m": (err[:, mr_step] > mr_thresh_m).float().mean().item(),
    }


def per_style_ade(pred, gt, style: np.ndarray, names):
    err = ((pred - gt).norm(dim=-1) * SCALE).mean(1).numpy()
    return {n: float(err[style == i].mean()) for i, n in enumerate(names) if (style == i).any()}
