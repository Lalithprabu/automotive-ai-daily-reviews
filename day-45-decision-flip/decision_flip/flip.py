"""Mean-FLIP scoring, guarded Otsu survivor selection, and physical compilation of a smaller bank."""
import copy
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def mean_flip_scores(model, calib_feats, chunk=1):
    """Score each bank row by the mean number of voxel argmax decisions that change when it is removed.

    Label-free: uses only calibration *inputs*. Exact (not approximate) because the readout is additive:
    mass_without_i = mass - contrib_i, so each leave-one-out decision is one subtraction + argmax.
    Returns (N,) float tensor.
    """
    model.eval(); bank = model.bank
    N = bank.n; total = torch.zeros(N)
    for i in range(0, len(calib_feats), chunk):
        f = calib_feats[i:i + chunk]
        m, c = bank.mass(f)                                                   # m (B,V,C), c (B,N,V,C)
        m0 = m.clone(); m0[..., 0] += F.softplus(bank.empty_bias)
        base = (m0).argmax(-1)                                                # (B,V)
        loo = (m0.unsqueeze(1) - c)                                           # (B,N,V,C) leave-one-out mass
        flips = (loo.argmax(-1) != base.unsqueeze(1)).float().sum(-1)         # (B,N) voxels flipped
        total += flips.sum(0)
    return total / len(calib_feats)


def guarded_otsu(scores, min_sep=0.35, eps=1.0e-2, max_valley=0.5, bins=20):
    """Pick a keep-threshold on log-scores with Otsu; ABSTAIN (keep all) if the split isn't supported.

    Guard: normalised between-class variance (eta) must exceed `min_sep` and both classes must be non-trivial.
    Returns (keep_mask bool (N,), info dict).
    """
    x = torch.log(scores + eps)
    xs = x.sort().values
    best, thr = -1.0, None
    var_tot = x.var(unbiased=False).item() + 1e-12
    cands = (xs[1:] + xs[:-1]) / 2
    for t in cands.unique():
        lo, hi = x[x <= t], x[x > t]
        if len(lo) < 2 or len(hi) < 2:
            continue
        w0, w1 = len(lo) / len(x), len(hi) / len(x)
        sb = w0 * w1 * (lo.mean() - hi.mean()).item() ** 2
        if sb > best:
            best, thr = sb, t.item()
    eta = best / var_tot if thr is not None else 0.0
    valley = 1.0
    if thr is not None:
        # Valley guard: a true two-population split has a density dip at the threshold (a uniform spread does not,
        # yet still scores eta~0.75, so eta alone is not a sufficient guard).
        h = torch.histc(x, bins=bins, min=x.min().item(), max=x.max().item() + 1e-9); edges = torch.linspace(x.min().item(), x.max().item() + 1e-9, bins + 1)
        tb = min(max(int(torch.bucketize(torch.tensor(thr), edges)) - 1, 0), bins - 1)
        dens_thr = h[max(tb - 1, 0):tb + 2].mean(); peak = min(h[:tb].max() if tb > 0 else h.max(), h[tb + 1:].max() if tb < bins - 1 else h.max())
        valley = float(dens_thr / (peak + 1e-9))
    info = {"eta": eta, "thr": thr, "valley": valley}
    if thr is None or eta < min_sep or valley > max_valley:
        return torch.ones_like(scores, dtype=torch.bool), {**info, "abstained": True}
    return x > thr, {**info, "abstained": False}


def compile_pruned(model, keep_mask):
    """Physically build a smaller model: slice per-row params; shared MLP and empty bias are kept."""
    new = copy.deepcopy(model); b = new.bank; idx = keep_mask.nonzero().squeeze(-1)
    for name in ("mean", "log_scale", "opa", "cls"):
        p = getattr(b, name)
        setattr(b, name, torch.nn.Parameter(p.data[idx].clone()))
    return new


def topk_mask(scores, k, largest=True):
    idx = scores.topk(k, largest=largest).indices
    m = torch.zeros_like(scores, dtype=torch.bool); m[idx] = True
    return m
