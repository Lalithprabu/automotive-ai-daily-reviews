"""Losses, margin tokens, TTC-margin guard and selection (torch, batched)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import tokens as TK
from . import world as W

SCALE = torch.tensor([20.0, 3.5])


def upsample_t(fut: torch.Tensor, cur: torch.Tensor, sub: int = 2) -> torch.Tensor:
    """(..., H, 2) future + (..., 2) present -> (..., H*sub, 2) linear interpolation."""
    full = torch.cat([cur[..., None, :], fut], -2)
    a, b = full[..., :-1, :], full[..., 1:, :]
    w = torch.arange(1, sub + 1, dtype=fut.dtype) / sub                      # (sub,)
    out = a[..., None, :] * (1 - w)[:, None] + b[..., None, :] * w[:, None]  # (..., H, sub, 2)
    return out.flatten(-3, -2)


def clearance_t(traj: torch.Tensor, act_fut: torch.Tensor, b: dict):
    """traj (B, K, H, 2), act_fut (B, N, H, 2) -> min clearance (B, K) and argmin time in s (B, K)."""
    ego = upsample_t(traj, torch.zeros_like(traj[..., 0, :]))               # (B, K, T, 2)
    act = upsample_t(act_fut, b["hist"][:, :, -1, :2])                       # (B, N, T, 2)
    d = (ego[:, :, None] - act[:, None]).norm(dim=-1)                       # (B, K, N, T)
    d = d - (W.R_EGO + b["radius"])[:, None, :, None]
    d = d.masked_fill(~b["valid"][:, None, :, None], 1e3)
    m, idx = d.flatten(2).min(-1)
    t = (idx % d.shape[-1] + 1).float() * (W.DT / 2)
    return m, t


def cv_projection(b: dict) -> torch.Tensor:
    """Constant-velocity actor projection from the PRESENT state (deployable) -> (B, N, H, 2)."""
    cur, vel = b["hist"][:, :, -1, :2], b["hist"][:, :, -1, 2:]
    return cur[:, :, None] + vel[:, :, None] * torch.as_tensor(W.T_FUT, dtype=cur.dtype)[None, None, :, None]


def margin_tokens(base_traj: torch.Tensor, b: dict):
    """Teacher margin queries (one per base candidate) from the LOGGED future.  -> E (B,K,4), C (B,K), R (B,K)."""
    m, t = clearance_t(base_traj, b["fut"], b)
    E = torch.stack([m.clamp(-1, 5) / 5, t / 4, ((m > 0) & (m < 1.0)).float(), (m < 0).float()], -1)
    return E, (m < 2.0).float(), (m < 0).float()


def teacher_tokens(base_traj, b):
    Em, Cm, Rm = margin_tokens(base_traj, b)
    return torch.cat([b["tE"], Em], 1), torch.cat([b["tC"], Cm], 1), torch.cat([b["tR"], Rm], 1)


# --------------------------------------------------------------------------- base / imitation loss
def wta_loss(traj, logit, expert):
    """Winner-take-all imitation: regress the closest candidate, classify it.  (Stand-in for L_diff.)"""
    err = ((traj - expert[:, None]) / SCALE).abs().mean((-1, -2))           # (B, K)
    k = err.argmin(-1)
    reg = F.smooth_l1_loss(traj[torch.arange(len(k)), k] / SCALE, expert / SCALE)
    return reg + 0.5 * F.cross_entropy(logit, k), k


def distill_loss(pred, teacher, l_reg=1.0, l_conf=0.5, l_risk=0.5):
    """Paper: masked sum over queries of Smooth-L1(e) + BCE(c) + BCE(r), teacher is stop-gradient."""
    e, c, r = pred
    tE, tC, tR = (x.detach() for x in teacher)
    mask = tC                                                                 # e / r supervised where valid
    reg = (F.smooth_l1_loss(e, tE, reduction="none").sum(-1) * mask).sum() / mask.sum().clamp(min=1)
    conf = F.binary_cross_entropy_with_logits(c, tC)
    risk = (F.binary_cross_entropy_with_logits(r, tR, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
    return l_reg * reg + l_conf * conf + l_risk * risk


# --------------------------------------------------------------------------- preservation + guard
def preserve_loss(tf, tb, b, r_margin):
    """L_preserve = sum_k w_k rho(τ_f^k - τ_b^k).

    w_k high for good base candidates (close to expert AND collision-free under the logged future) and low
    for candidates tied to high-confidence interaction tokens (here: the margin-risk token of candidate k).
    """
    with torch.no_grad():
        ade = ((tb - b["expert"][:, None]).norm(dim=-1)).mean(-1)            # (B, K)
        m, _ = clearance_t(tb, b["fut"], b)
        q = torch.exp(-ade / 2.0) * (m > 0).float()
        w = q * (1 - r_margin)
    rho = F.smooth_l1_loss(tf / SCALE, tb / SCALE, reduction="none").mean((-1, -2))
    return (w * rho).sum(-1).mean()


def guard_loss(tf, tb, b, eps_b=1.0, eps_g=0.75):
    """L_guard = sum_k 1[m(τ_b^k) > ε_b] * max(0, ε_g - m(τ_f^k)),  m = clearance vs CV projection."""
    cv = cv_projection(b)
    with torch.no_grad():
        mb, _ = clearance_t(tb, cv, b)
    mf, _ = clearance_t(tf, cv, b)
    return ((mb > eps_b).float() * F.relu(eps_g - mf)).sum(-1).mean()


@torch.no_grad()
def apply_guard(tf, tb, b, eps_b=1.0, eps_g=0.75):
    """Inference rule: revert any candidate that turned a CV-safe base candidate into a near-contact one."""
    cv = cv_projection(b)
    mb, _ = clearance_t(tb, cv, b)
    mf, _ = clearance_t(tf, cv, b)
    bad = (mb > eps_b) & (mf < eps_g)                                         # (B, K)
    return torch.where(bad[..., None, None], tb, tf), bad
