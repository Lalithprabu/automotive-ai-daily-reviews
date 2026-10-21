import torch
import torch.nn.functional as F
from .data import CELL, HALF, GRID


def decode(logits, thr):
    p = torch.sigmoid(logits)
    peak = (p == F.max_pool2d(p, 3, 1, 1)) & (p > thr)
    return peak[:, 0]  # [B,H,W] bool


def gt_cells(pos, mask):
    col = ((pos[..., 0] + HALF) / CELL).floor().long().clamp(0, GRID - 1)
    row = ((pos[..., 1] + HALF) / CELL).floor().long().clamp(0, GRID - 1)
    return row, col


@torch.no_grad()
def detection_stats(logits, pos, mask, thr, radius=1.5):
    """Greedy-free per-GT/pred nearest matching within `radius` cells. Returns dict of counts for
    all / blind-zone (x>=0) objects, plus summed localisation error (cells) of matched GTs."""
    pk = decode(logits, thr)
    B = pk.shape[0]
    row, col = gt_cells(pos, mask)
    tp = fn = fp = tp_b = fn_b = 0; err = 0.0
    for b in range(B):
        pr = pk[b].nonzero().float()           # [P,2] (row, col)
        used = torch.zeros(len(pr), dtype=torch.bool)
        for n in range(pos.shape[1]):
            if mask[b, n] == 0:
                continue
            blind = pos[b, n, 0] >= 0
            g = torch.tensor([float(row[b, n]), float(col[b, n])])
            hit = False
            if len(pr):
                d = (pr - g).norm(dim=1); d[used] = 1e9
                j = int(d.argmin())
                if d[j] <= radius:
                    used[j] = True; hit = True; err += float(d[j])
            if hit:
                tp += 1; tp_b += int(blind)
            else:
                fn += 1; fn_b += int(blind)
        fp += int((~used).sum())
    return dict(tp=tp, fn=fn, fp=fp, tp_b=tp_b, fn_b=fn_b, err=err)


def summarize(s):
    prec = s["tp"] / max(s["tp"] + s["fp"], 1); rec = s["tp"] / max(s["tp"] + s["fn"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(precision=prec, recall=rec, f1=f1, blind_recall=s["tp_b"] / max(s["tp_b"] + s["fn_b"], 1),
                loc_err_cells=s["err"] / max(s["tp"], 1))
