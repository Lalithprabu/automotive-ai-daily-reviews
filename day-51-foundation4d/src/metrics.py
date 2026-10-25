import torch
import torch.nn.functional as F
from .synthetic import MAX_D


def abs_rel(pred, gt):
    """Mean |pred-gt|/gt over non-sky pixels at the last frame. pred/gt [B,V,T,H,W]."""
    m = gt[:, :, -1] < MAX_D - 1e-3
    return ((pred[:, :, -1] - gt[:, :, -1]).abs() / gt[:, :, -1])[m].mean().item()


def peaks(heat_prob, thr):
    """3x3 NMS peaks -> list per sample of (ix,iy) cell indices."""
    mx = F.max_pool2d(heat_prob, 3, 1, 1)
    keep = (heat_prob == mx) & (heat_prob > thr)
    return [keep[b, 0].nonzero()[:, [1, 0]] for b in range(heat_prob.shape[0])]


def det_f1(heat_logit, heat_gt, thr, tol=1.5):
    """Greedy match predicted peaks to GT peaks (heat==1) within `tol` cells. Returns (p, r, f1)."""
    pp = peaks(torch.sigmoid(heat_logit), thr)
    gp = [(heat_gt[b, 0] == 1.0).nonzero()[:, [1, 0]] for b in range(heat_gt.shape[0])]
    tp = fp = fn = 0
    for P, G in zip(pp, gp):
        used = set()
        for p in P.float():
            if len(G) == 0:
                fp += 1; continue
            d = (G.float() - p).norm(dim=1)
            for j in used:
                d[j] = 1e9
            j = int(d.argmin())
            if d[j] <= tol:
                tp += 1; used.add(j)
            else:
                fp += 1
        fn += len(G) - len(used)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def best_threshold(heat_logit, heat_gt):
    """Post-hoc F1-maximising threshold on held-out data (Day 14 convention)."""
    best = (0.0, 0.5, (0, 0, 0))
    for thr in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        r = det_f1(heat_logit, heat_gt, thr)
        if r[2] > best[0]:
            best = (r[2], thr, r)
    return best[1], best[2]


def occ_iou(occ_logit, occ_gt, thr=0.5):
    p = torch.sigmoid(occ_logit) > thr
    g = occ_gt > 0.5
    return ((p & g).sum() / (p | g).sum().clamp(min=1)).item()
