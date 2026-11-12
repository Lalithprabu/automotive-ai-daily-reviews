"""Training losses and the detection decoder."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import wrap_t


def focal_heat(pred_logit: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0):
    """CenterNet penalty-reduced focal loss.  pred_logit, gt: (B, K, R, A)."""
    p = pred_logit.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pos_l = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_l = -((1 - gt) ** beta) * (p ** alpha) * torch.log(1 - p) * (1 - pos)
    return (pos_l.sum() + neg_l.sum()) / pos.sum().clamp(min=1.0)


def ramod_loss(out: dict, heat, reg, mask, ego_v, vmax: float, w: dict | None = None,
               ego_sup: bool = True, wrap_sup: bool = True):
    """
    reg channels: 0 log_h, 1 log_w, 2 off_r, 3 off_a, 4 v_meas, 5 k_idx (k+2), 6 v_fwd, 7 v_rad (true)
    Loss = focal(heat) + L1(size) + L1(off)
         + raw hypothesis:       circular-L1(v_raw, v_meas) + L1(f_raw, v_fwd)
         + unwrapped hypothesis: L1(v_unw, v_rad)           + L1(f_unw, v_fwd)
         + BCE(p_wrap, k != 0)                                   [off for the "no wrapping loss" ablation]
         + ego: CE over the speed grid + L1(v_e)                 [off for the "no ego supervision" ablation]
    All object terms are taken at GT centre cells.
    """
    w = w or dict(heat=1.0, size=0.2, off=1.0, v=1.0, f=0.3, unw=0.5, wrap=0.5, ego=0.5)
    m = mask.bool()                                                                   # (B, R, A)
    n = m.sum().clamp(min=1)
    L = dict(heat=focal_heat(out["heat"], heat))
    def at(x):                                                                        # (B, C, R, A) -> (N, C)
        return x.permute(0, 2, 3, 1)[m]
    r = reg.permute(0, 2, 3, 1)[m]                                                    # (N, 8)
    L["size"] = F.l1_loss(at(out["size"]), r[:, 0:2], reduction="sum") / n
    L["off"] = F.l1_loss(at(out["off"]), r[:, 2:4], reduction="sum") / n
    L["v"] = wrap_t(out["v_raw"][m] - r[:, 4], vmax).abs().sum() / n                 # circular: wrap-aware
    wrapped = (r[:, 5] != 2).float()
    if out["wrap"] is not None:
        # each hypothesis' forward velocity is trained where that hypothesis is the true one
        L["f"] = (((out["f_raw"][m] - r[:, 6]).abs() * (1 - wrapped)).sum() +
                  ((out["f_unw"][m] - r[:, 6]).abs() * wrapped).sum()) / n
        L["unw"] = (out["v_unw"][m] - r[:, 7]).abs().sum() / n
        if wrap_sup:
            L["wrap"] = F.binary_cross_entropy_with_logits(out["wrap"][m], wrapped, reduction="sum") / n
    else:
        L["f"] = (out["f_raw"][m] - r[:, 6]).abs().sum() / n
    if "v_e" in out and ego_sup:
        grid_v = torch.linspace(0.0, 8.5, out["ego_logits"].shape[-1], device=ego_v.device)
        tgt = (ego_v[..., None] - grid_v).abs().argmin(-1)                             # (B, T)
        L["ego"] = F.cross_entropy(out["ego_logits"].flatten(0, 1), tgt.flatten()) + \
            F.l1_loss(out["v_e"], ego_v)
    total = sum(w.get(k, 1.0) * v for k, v in L.items())
    return total, {k: float(v.detach()) for k, v in L.items()}


@torch.no_grad()
def decode(out: dict, vmax: float, topk: int = 20, thresh: float = 0.15, wrap_head: bool = True):
    """Heat-map peaks -> list (per batch item) of detections:
    dict(cls, score, box[r0,a0,r1,a1] in bins, p_wrap, wrapped, v_rad, v_fwd, v_raw, v_unw, center)."""
    heat = out["heat"].sigmoid()
    B, K, R, A = heat.shape
    peak = (heat == F.max_pool2d(heat, 3, 1, 1)).float() * heat
    sc, idx = peak.reshape(B, -1).topk(topk)
    dets = []
    for b in range(B):
        lst = []
        for s, i in zip(sc[b].tolist(), idx[b].tolist()):
            if s < thresh:
                break
            c, rem = divmod(i, R * A)
            ri, ai = divmod(rem, A)
            lh, lw = out["size"][b, :, ri, ai].exp().tolist()
            orr, oa = out["off"][b, :, ri, ai].tolist()
            cr, ca = ri + orr, ai + oa
            v_raw, f_raw = float(out["v_raw"][b, ri, ai]), float(out["f_raw"][b, ri, ai])
            v_unw, f_unw = float(out["v_unw"][b, ri, ai]), float(out["f_unw"][b, ri, ai])
            p_wrap = float(out["wrap"][b, ri, ai].sigmoid()) if wrap_head else 0.0
            use_unw = p_wrap >= 0.5                                                   # paper's inference rule
            lst.append(dict(cls=c, score=s, box=[cr - lh / 2, ca - lw / 2, cr + lh / 2, ca + lw / 2],
                            p_wrap=p_wrap, wrapped=use_unw, v_raw=v_raw, v_unw=v_unw,
                            v_rad=v_unw if use_unw else v_raw, v_fwd=f_unw if use_unw else f_raw,
                            center=(cr, ca)))
        dets.append(lst)
    return dets
