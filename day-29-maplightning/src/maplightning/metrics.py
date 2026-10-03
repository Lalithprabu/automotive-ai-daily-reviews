import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .data import make_batch, CLS_NONE


@torch.no_grad()
def evaluate(model, n=256, s=0.0, seed=0, bs=64, needs_cam=False):
    """Returns dict(err_m, f1_1m). Kept predictions = non-'none' argmax; matched to GT polylines
    (same class only) by minimum mean point error; TP if error < 1.0 m."""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    errs, tp, npred, ngt = [], 0, 0, 0
    for _ in range(0, n, bs):
        b = make_batch(bs, s, gen)
        logits, pts = model(b["img"])
        pred_cls = logits.argmax(-1)
        for i in range(bs):
            keep = (pred_cls[i] != CLS_NONE).nonzero().squeeze(-1)
            g = b["present"][i].nonzero().squeeze(-1)
            npred += len(keep); ngt += len(g)
            if len(keep) == 0 or len(g) == 0:
                continue
            d = (pts[i][keep][:, None] - b["polys"][i][g][None]).norm(dim=-1).mean(-1)   # (K,G) metres
            bad = pred_cls[i][keep][:, None] != b["cls"][i][g][None]
            cost = (d + 1e3 * bad.float()).numpy()
            qi, gi = linear_sum_assignment(cost)
            for q, k in zip(qi, gi):
                if cost[q, k] < 1e3:
                    errs.append(float(d[q, k]))
                    tp += int(d[q, k] < 1.0)
    prec, rec = tp / max(npred, 1), tp / max(ngt, 1)
    return dict(err_m=float(np.mean(errs)) if errs else float("nan"), f1_1m=2 * prec * rec / max(prec + rec, 1e-9),
                precision=prec, recall=rec)
