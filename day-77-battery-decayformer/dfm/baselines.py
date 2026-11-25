"""Non-learned baseline: fit the early capacity fade with a physics-style law and extrapolate to EOL.

    SOH(n) ~= 1 - a * sqrt(n) - b * n      (SEI + linear LAM), a, b >= 0, least squares on the first k cycles

This is the classic "extrapolate the early curve" approach. It has no way to see a knee coming.
"""
from __future__ import annotations

import numpy as np

from .data import F_MU, F_SD, N_MAX


def capacity_from_x(x: np.ndarray) -> np.ndarray:
    """Normalised feature tensor (M, K, F) -> measured capacity / SOH (M, K)."""
    return x[..., 0] * F_SD[0] + F_MU[0]


def sqrt_fit(x: np.ndarray, k: int, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns SOH predictions on `grid` (M, G) and lifetimes (M,) from the first k cycles."""
    q = capacity_from_x(x)[:, :k]
    n = np.arange(1, k + 1, dtype=float)
    A = np.stack([np.sqrt(n), n], -1)
    preds, lives = [], []
    nn_ = np.arange(0, N_MAX + 1, dtype=float)
    for qi in q:
        coef, *_ = np.linalg.lstsq(A, 1.0 - qi, rcond=None)
        coef = np.maximum(coef, 0.0)
        if coef.sum() == 0:
            coef = np.array([1e-4, 0.0])
        f = lambda c: 1.0 - coef[0] * np.sqrt(c) - coef[1] * c
        preds.append(f(grid))
        s = f(nn_)
        idx = np.nonzero(s <= 0.8)[0]
        lives.append(float(nn_[idx[0]]) if len(idx) else float(N_MAX))
    return np.asarray(preds), np.asarray(lives)
