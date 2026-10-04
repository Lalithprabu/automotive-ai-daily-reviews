import numpy as np
import torch
import torch.nn.functional as F


def focal_loss(logit, target, alpha=2, beta=4):
    """CenterNet focal loss; positives are pixels == 1.0 (centre forced to exactly 1.0 in data.heatmap)."""
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 1.0).float()
    neg = 1 - pos
    pl = -(torch.log(p) * (1 - p) ** alpha * pos).sum()
    nl = -(torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg).sum()
    return (pl + nl) / pos.sum().clamp(min=1)


def extract_peaks(prob, thr=0.05, topk=20):
    """prob (B,1,H,W) -> list of (K,3) arrays [x,y,score] via 3x3 NMS."""
    mx = F.max_pool2d(prob, 3, 1, 1)
    keep = (prob == mx) & (prob > thr)
    out = []
    for b in range(prob.shape[0]):
        ys, xs = torch.nonzero(keep[b, 0], as_tuple=True)
        s = prob[b, 0, ys, xs]
        o = torch.argsort(s, descending=True)[:topk]
        out.append(torch.stack([xs[o].float(), ys[o].float(), s[o]], 1).cpu().numpy())
    return out


def average_precision(peaks, gts, valids, tau, subset=None):
    """Center-distance AP. gts (B,N,2), valids (B,N); subset optional (B,N) bool restricts GT set."""
    recs, npos = [], 0
    for b, pk in enumerate(peaks):
        m = valids[b].numpy()
        g = gts[b].numpy()[m]
        insub = np.ones(len(g), bool) if subset is None else subset[b].numpy()[m]
        npos += int(insub.sum())
        used = np.zeros(len(g), bool)
        for x, y, s in pk:                      # peaks sorted by score
            if len(g):
                d = np.hypot(g[:, 0] - x, g[:, 1] - y)
                d[used] = 1e9
                j = int(d.argmin())
                if d[j] <= tau:
                    used[j] = True
                    if insub[j]:
                        recs.append((s, 1))     # TP on a GT inside the evaluated subset
                    continue                    # match to a GT outside the subset: ignored (neither TP nor FP)
            recs.append((s, 0))
    if npos == 0:
        return float("nan")
    recs.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([1 - r[1] for r in recs])
    rec = tp / npos; prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
