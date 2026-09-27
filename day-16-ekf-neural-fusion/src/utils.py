"""Shared utilities: reproducibility seeding + ADE/FDE metrics."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average Displacement Error: mean L2 distance between predicted and
    ground-truth positions, averaged over all timesteps (and batch, if a
    batch dimension is given).

    pred, gt: [..., T, 2]
    returns: scalar float
    """
    diff = pred - gt                                    # [..., T, 2]
    dist = np.linalg.norm(diff, axis=-1)                 # [..., T]
    return float(dist.mean())


def fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """Final Displacement Error: L2 distance at the LAST timestep only,
    averaged over the batch dimension if present.

    pred, gt: [..., T, 2]
    returns: scalar float
    """
    diff = pred[..., -1, :] - gt[..., -1, :]             # [..., 2]
    dist = np.linalg.norm(diff, axis=-1)                 # [...]
    return float(dist.mean())


def ade_fde(pred: np.ndarray, gt: np.ndarray):
    return ade(pred, gt), fde(pred, gt)
