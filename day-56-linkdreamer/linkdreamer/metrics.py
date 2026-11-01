"""Peak-picking decode and detection metrics on BEV centre heatmaps."""
import numpy as np
import torch
import torch.nn.functional as F

from .synthetic import G, CELL, XMIN, YMIN

RADIUS = 3.0     # matching radius [m]


def decode(logits, thr=0.3):
    """logits [B,1,G,G] -> list (len B) of [(x_m, y_m, score)] after 3x3 NMS."""
    p = torch.sigmoid(logits)
    keep = (p == F.max_pool2d(p, 3, 1, 1)) & (p > thr)
    out = []
    for b in range(p.shape[0]):
        ij = keep[b, 0].nonzero()
        out.append([(XMIN + (int(i) + .5) * CELL, YMIN + (int(j) + .5) * CELL, float(p[b, 0, i, j])) for i, j in ij])
    return out


def match(dets, gt):
    """Greedy score-ordered matching.  -> list of (score, is_tp), n_missed"""
    used, res = set(), []
    for x, y, s in sorted(dets, key=lambda d: -d[2]):
        best, bd = None, RADIUS
        for k, (gx, gy) in enumerate(gt):
            d = np.hypot(x - gx, y - gy)
            if k not in used and d < bd:
                best, bd = k, d
        if best is not None:
            used.add(best)
        res.append((s, best is not None))
    return res, len(gt) - len(used)


def ap_f1(records, n_gt):
    """records: list of (score, tp). AP = area under the precision-recall curve (all-point); F1 at the decode threshold."""
    if n_gt == 0:
        return float("nan"), float("nan")
    r = sorted(records, key=lambda t: -t[0])
    tp = np.cumsum([x for _, x in r]) if r else np.zeros(1)
    fp = np.cumsum([not x for _, x in r]) if r else np.zeros(1)
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    t, f = (tp[-1], fp[-1]) if r else (0, 0)
    return ap, float(2 * t / max(2 * t + f + (n_gt - t), 1))


def heat_loss(logits, tgt, reduction="none"):
    """Class-balanced BCE on the centre heatmap; per-sample mean if reduction=='none'."""
    l = F.binary_cross_entropy_with_logits(logits, tgt[:, None], reduction="none", pos_weight=torch.tensor(6.0))
    return l.flatten(1).mean(1) if reduction == "none" else l.mean()
