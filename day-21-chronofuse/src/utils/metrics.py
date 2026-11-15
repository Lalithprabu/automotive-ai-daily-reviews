"""
Lightweight evaluation utilities: heatmap peak decoding + a simplified
"center-distance recovery" metric.

DISCLOSURE: the source paper reports sAP (a COCO-style spatio-temporal
Average Precision metric) on real benchmarks. Reproducing sAP faithfully
requires the paper's exact matching/IoU protocol, which was not
recoverable from the abstract alone (see SOURCING.md). This module instead
implements a simpler, self-contained metric — mean center-pixel distance
error between predicted and true future object positions, and the
percentage of the "naive / no-compensation" error that ChronoFuse
recovers — reported honestly as this repo's own stand-in, never conflated
with the paper's sAP numbers.
"""

from typing import List

import torch
import torch.nn.functional as F

from src.utils.losses import STRIDE


def _nms_peaks(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keeps only local-maximum pixels (standard CenterNet max-pool NMS).

    heatmap: [num_classes, H, W] (already sigmoid-activated probabilities).
    """
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(heatmap.unsqueeze(0), kernel, stride=1, padding=pad).squeeze(0)
    keep = (pooled == heatmap).float()
    return heatmap * keep


def decode_centers(heatmap_logits: torch.Tensor, offset: torch.Tensor, num_per_class: List[int]):
    """Decodes the top-K predicted centers per class from one sample's
    heatmap/offset predictions.

    Args:
        heatmap_logits: [num_classes, H, W]
        offset: [2, H, W]
        num_per_class: how many peaks to keep for each class index (e.g.
            the true agent count per class, for this repo's simplified
            "matched" evaluation protocol).

    Returns:
        List of (class_id, x, y) in full-resolution pixel coordinates.
    """
    probs = torch.sigmoid(heatmap_logits)
    detections = []
    for cls, k in enumerate(num_per_class):
        if k <= 0:
            continue
        peaks = _nms_peaks(probs[cls].unsqueeze(0)).squeeze(0)
        flat = peaks.flatten()
        k = min(k, flat.numel())
        topk_vals, topk_idx = torch.topk(flat, k)
        h, w = peaks.shape
        for idx, val in zip(topk_idx.tolist(), topk_vals.tolist()):
            if val <= 0:
                continue
            iy, ix = idx // w, idx % w
            dx, dy = offset[:, iy, ix].tolist()
            x = (ix + dx) * STRIDE
            y = (iy + dy) * STRIDE
            detections.append((cls, x, y))
    return detections


def _greedy_match_distance(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> float:
    """Greedy nearest-neighbor matching between two small point sets,
    returning the mean matched distance (pixels). Falls back to a large
    penalty distance for any unmatched (missed) ground-truth point."""
    if gt_xy.shape[0] == 0:
        return 0.0
    if pred_xy.shape[0] == 0:
        return 64.0  # penalty for "no detections at all"

    dists = torch.cdist(gt_xy, pred_xy)  # [num_gt, num_pred]
    used_pred = set()
    total = 0.0
    for i in range(gt_xy.shape[0]):
        row = dists[i].clone()
        for j in used_pred:
            row[j] = float("inf")
        j_best = int(torch.argmin(row).item())
        if torch.isinf(row[j_best]):
            total += 64.0
        else:
            total += row[j_best].item()
            used_pred.add(j_best)
    return total / gt_xy.shape[0]


def center_distance_error(heatmap_logits: torch.Tensor, offset: torch.Tensor,
                           gt_centers: torch.Tensor, gt_classes: torch.Tensor) -> float:
    """Mean matched center-distance error (pixels) for one sample."""
    num_per_class = [int((gt_classes == c).sum().item()) for c in range(heatmap_logits.shape[0])]
    dets = decode_centers(heatmap_logits, offset, num_per_class)
    pred_xy = torch.tensor([[x, y] for _, x, y in dets], dtype=torch.float32) if dets else torch.zeros(0, 2)
    return _greedy_match_distance(pred_xy, gt_centers)


def stale_baseline_error(stale_centers: torch.Tensor, future_centers: torch.Tensor) -> float:
    """The 'old way' error: reporting the last-observed position with zero
    latency compensation, evaluated against where the object actually is
    once the output is available."""
    if future_centers.shape[0] == 0:
        return 0.0
    return torch.norm(stale_centers - future_centers, dim=-1).mean().item()
