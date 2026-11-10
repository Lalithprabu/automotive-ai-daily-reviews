"""Center-distance detection metrics (BEV, greedy one-to-one matching)."""
from __future__ import annotations

import numpy as np


def match_greedy(pred: np.ndarray, gt: np.ndarray, thr: float):
    """Greedy nearest matching.  Returns (hit_pred_mask, hit_gt_mask)."""
    hp = np.zeros(len(pred), bool)
    hg = np.zeros(len(gt), bool)
    if len(pred) == 0 or len(gt) == 0:
        return hp, hg
    d = np.linalg.norm(pred[:, None] - gt[None], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for i, j in order:
        if d[i, j] > thr:
            break
        if not hp[i] and not hg[j]:
            hp[i] = hg[j] = True
    return hp, hg


def fuse(ego: np.ndarray, road_corrected: np.ndarray, dedupe: float = 1.5) -> np.ndarray:
    """Union of ego boxes and roadside boxes, dropping roadside boxes duplicating an ego box."""
    if len(road_corrected) == 0:
        return ego
    if len(ego) == 0:
        return road_corrected
    d = np.linalg.norm(road_corrected[:, None] - ego[None], axis=2).min(1)
    return np.vstack([ego, road_corrected[d > dedupe]])


def prf(tp: int, fp: int, fn: int):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return p, r, f
