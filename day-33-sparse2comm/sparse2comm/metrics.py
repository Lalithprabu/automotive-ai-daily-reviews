import torch
import torch.nn.functional as F


def extract_peaks(logits, thr=0.05):
    """3x3 max-pool NMS. Returns per-sample list of (x, y, score) tensors."""
    p = torch.sigmoid(logits)
    keep = (F.max_pool2d(p, 3, 1, 1) == p) & (p > thr)
    res = []
    for b in range(p.shape[0]):
        ys, xs = torch.nonzero(keep[b, 0], as_tuple=True)
        res.append(torch.stack([xs.float(), ys.float(), p[b, 0, ys, xs]], 1))
    return res


def average_precision(preds, gt_pos, gt_valid, radius):
    """Centre-distance AP (radius in cells) - an analogue of AP@IoU, NOT the paper's metric."""
    recs, n_gt = [], int(gt_valid.sum().item())
    for b, pk in enumerate(preds):
        gts = gt_pos[b][gt_valid[b] > 0]
        used = torch.zeros(len(gts), dtype=torch.bool)
        for i in torch.argsort(-pk[:, 2]) if len(pk) else []:
            hit = False
            if len(gts):
                d = (gts - pk[i, :2]).norm(dim=1); d[used] = 1e9
                j = int(torch.argmin(d))
                if d[j] <= radius:
                    used[j] = True; hit = True
            recs.append((float(pk[i, 2]), hit))
    if n_gt == 0 or not recs:
        return 0.0
    recs.sort(key=lambda t: -t[0])
    tp = torch.tensor([r[1] for r in recs], dtype=torch.float32).cumsum(0)
    prec = tp / torch.arange(1, len(recs) + 1)
    rec = tp / n_gt
    mrec = torch.cat([torch.zeros(1), rec, torch.ones(1)]); mpre = torch.cat([torch.zeros(1), prec, torch.zeros(1)])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = torch.nonzero(mrec[1:] != mrec[:-1]).flatten()
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


@torch.no_grad()
def evaluate(model, cfg_d, n_scenes=512, seed=999, radii=(1.0, 2.0), bs=128, **kw):
    from .data import sample_scene, make_batch
    model.eval()
    g = torch.Generator().manual_seed(seed)
    P, G, V, offerr = [], [], [], []
    for _ in range(0, n_scenes, bs):
        sc = sample_scene(bs, cfg_d, g)
        b = make_batch(sc, cfg_d, g, **kw)
        out = model(b)
        P += extract_peaks(out["logits"]); G.append(b["pos"]); V.append(b["valid"])
        if "offset" in out:
            offerr.append((out["offset"] - b["offset"]).norm(dim=1))
    G, V = torch.cat(G), torch.cat(V)
    res = {f"AP@r{r:g}": average_precision(P, G, V, r) for r in radii}
    if offerr:
        res["offset_err"] = float(torch.cat(offerr).mean())
    return res
