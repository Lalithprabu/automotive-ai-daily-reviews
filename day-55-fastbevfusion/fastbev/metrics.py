import numpy as np
import torch
import torch.nn.functional as F
from .synthetic import CELL, cell_xy


def decode(logits, thr=0.3):
    """Peak picking (3x3 max-pool NMS). Returns per-sample list of (fwd_m, lat_m, score)."""
    p = torch.sigmoid(logits)
    keep = (p == F.max_pool2d(p, 3, 1, 1)) & (p > thr)
    f, l = cell_xy(); out = []
    for b in range(p.shape[0]):
        idx = keep[b, 0].nonzero()
        out.append([(float(f[i, j]), float(l[i, j]), float(p[b, 0, i, j])) for i, j in idx.tolist()])
    return out


def match(dets, objs, tol=2.0 * CELL):
    """Greedy score-ordered matching. Returns [(score, is_tp)], n_missed GT indices."""
    used, res = set(), []
    for f, l, s in sorted(dets, key=lambda d: -d[2]):
        best, bd = None, tol
        for k, (of, ol, _) in enumerate(objs):
            d = np.hypot(f - of, l - ol)
            if k not in used and d < bd: best, bd = k, d
        if best is not None: used.add(best)
        res.append((s, best is not None))
    return res, [k for k in range(len(objs)) if k not in used]


@torch.no_grad()
def evaluate(model, L, I, O, yaw=None, thr=0.3, bs=256):
    """AP (all score levels), precision/recall/F1 at thr, recall on near (<24 m) and far (>=24 m) GT."""
    model.eval(); recs, n_gt = [], 0
    for s in range(0, len(L), bs):
        y = None if yaw is None else yaw[s:s + bs]
        dets = decode(model(L[s:s + bs], I[s:s + bs], y), 0.05)
        for d, o in zip(dets, O[s:s + bs]):
            r, _ = match(d, o); recs += r; n_gt += len(o)
    tp = fp = 0; near = [0, 0]; far = [0, 0]
    for s in range(0, len(L), bs):
        y = None if yaw is None else yaw[s:s + bs]
        dets = decode(model(L[s:s + bs], I[s:s + bs], y), thr)
        for d, o in zip(dets, O[s:s + bs]):
            r, miss = match(d, o); tp += sum(t for _, t in r); fp += sum(not t for _, t in r)
            for k, (f, l, _) in enumerate(o):
                b = near if np.hypot(f, l) < 24 else far
                b[1] += 1; b[0] += k not in miss
    recs.sort(key=lambda r: -r[0]); c = np.cumsum([t for _, t in recs]); fpc = np.cumsum([not t for _, t in recs])
    rec = c / max(n_gt, 1); prec = c / np.maximum(c + fpc, 1)
    ap = float(np.sum((rec[1:] - rec[:-1]) * np.maximum.accumulate(prec[::-1])[::-1][1:])) if len(rec) > 1 else 0.0
    ap += float(rec[0] * prec[0]) if len(rec) else 0
    P = tp / max(tp + fp, 1); R = tp / max(n_gt, 1)
    return dict(AP=ap, P=P, R=R, F1=2 * P * R / max(P + R, 1e-9), near_R=near[0] / max(near[1], 1), far_R=far[0] / max(far[1], 1))
