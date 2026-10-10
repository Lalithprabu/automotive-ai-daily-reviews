"""Synthetic ego-reactive collision scenarios (this repo's own stand-in for the paper's CARLA data).

Scene: ego drives along +x toward a conflict point at the origin; another agent drives along +y
through the same point. The other agent REACTS to the ego (brakes with a hidden compliance level
if the ego looks like it will win the conflict) -- so collision depends on the ego plan AND on a
latent the observer never sees. Observation noise grows with a scene "regime"
(clear < dense < occluded), which makes any single global calibration wrong somewhere.
"""
import math
import torch

REGIMES = ["clear", "dense", "occluded"]
REGIME_NOISE = torch.tensor([0.04, 0.14, 0.40])  # relative obs. noise per regime
DT, T = 0.1, 60  # 6 s horizon


def rollout(p: dict):
    """Vectorised ground-truth simulator. p: dict of (N,) tensors. Returns (xe, yo) each (N, T): ego x / other y."""
    N = p["v0"].shape[0]
    t = torch.arange(T, dtype=torch.float32) * DT  # (T,)
    # --- ego: v(t) = v0 + a t until t_b, then decel b; clipped at [0, 16] ---
    v = p["v0"][:, None] + p["a"][:, None] * t[None]
    v = v.clamp(0, 16)
    v_at_tb = (p["v0"] + p["a"] * p["t_b"]).clamp(0, 16)
    after = t[None] > p["t_b"][:, None]
    v_brake = (v_at_tb[:, None] - p["b"][:, None] * (t[None] - p["t_b"][:, None])).clamp(min=0)
    v = torch.where(after, v_brake, v)
    xe = -p["D_e"][:, None] + torch.cumsum(v, 1) * DT  # ego x position (m); conflict at 0
    # --- other agent: constant speed u0 along y, brakes (hidden compliance c) if ego is closer in time ---
    ttc_e = p["D_e"] / p["v0"].clamp(min=1.0)
    ttc_o = p["d0"] / p["u0"].clamp(min=1.0)
    yields = (ttc_e + 0.5 < ttc_o) | (xe[:, 10] > -0.5 * p["D_e"])  # sees ego committing early
    react_t = 0.6
    dec = torch.where(yields, p["c"], torch.zeros_like(p["c"]))
    u = p["u0"][:, None] - dec[:, None] * (t[None] - react_t).clamp(min=0)
    u = u.clamp(min=0)
    yo = -p["d0"][:, None] + torch.cumsum(u, 1) * DT
    return xe, yo


def simulate_collisions(p: dict) -> torch.Tensor:
    xe, yo = rollout(p)
    return torch.sqrt(xe ** 2 + yo ** 2).min(1).values < 3.0  # bool (N,)


def make_dataset(n: int, regime_mix, seed: int):
    g = torch.Generator().manual_seed(seed)
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g)
    mix = torch.tensor(regime_mix, dtype=torch.float32)
    regime = torch.multinomial(mix, n, replacement=True, generator=g)
    p = dict(
        v0=r(6, 14), a=r(-1.0, 2.5), t_b=r(0.5, 5.0), b=r(1.0, 6.0), D_e=r(25, 55),
        d0=r(20, 50), u0=r(6, 14), c=r(0.5, 6.0),  # c: hidden compliance, NEVER observed
    )
    y = simulate_collisions(p).long()
    clean = torch.stack([p["v0"], p["a"], p["t_b"], p["b"], p["D_e"], p["d0"], p["u0"]], 1)
    scale = torch.tensor([14., 2.5, 5., 6., 55., 50., 14.])
    sigma = REGIME_NOISE[regime][:, None]
    noisy = clean / scale + sigma * torch.randn(n, 7, generator=g)  # regime-dependent obs noise
    onehot = torch.nn.functional.one_hot(regime, 3).float()
    x = torch.cat([noisy, onehot], 1)  # (N, 10) model input
    # scene embedding used for LOCALIZATION: regime one-hot + (noisy) scene-only features (D_e, d0, u0)
    emb = torch.cat([onehot, 0.5 * noisy[:, [4, 5, 6]]], 1)  # (N, 6)
    return dict(x=x, y=y, regime=regime, emb=emb, params=p, clean=clean)
