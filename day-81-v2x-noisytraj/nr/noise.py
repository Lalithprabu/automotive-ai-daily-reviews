"""Realistic-ish object-state noise for sensor-tracked and V2X-received agents (my construction).

The abstract of arXiv 2611.28128 says real deployments see noisy position, heading and speed estimates, worse for
V2X-received states.  It gives no noise model, so this one is mine.  Noise level s (0 = clean ... 4 = severe) scales
every term linearly; temporally correlated (AR(1)) errors are used because white noise is unrealistically easy to
average away.

    sensor-tracked agent   position  AR(1) rho 0.7, sigma 0.20 s m per axis
                           heading   AR(1) rho 0.7, sigma 1.5 s deg
                           speed     AR(1) rho 0.7, sigma 0.25 s m/s
    V2X-received agent     position  GNSS-like slow bias, AR(1) rho 0.97, sigma 0.45 s m per axis + white 0.10 s m
                           heading   AR(1) rho 0.9, sigma 3 s deg;  speed AR(1) rho 0.9, sigma 0.35 s m/s
                           packet loss: each message lost with prob min(0.1 s, 0.5) -> previous value is held
                           ("fresh" flag = 0); the t0 message itself can be stale, so the latest state is old
    Heading is re-wrapped to (-pi, pi].
"""
from __future__ import annotations

import numpy as np
import torch

SENS = dict(rho=0.7, pos=0.20, head=np.deg2rad(1.5), spd=0.25)
V2X = dict(rho_pos=0.97, pos=0.45, white=0.10, rho=0.9, head=np.deg2rad(3.0), spd=0.35, drop=0.10)


def _ar1(shape, rho, sigma, gen):
    """Stationary AR(1) noise along the last axis: e_t = rho e_{t-1} + sqrt(1 - rho^2) w_t, std = sigma."""
    w = torch.randn(shape, generator=gen)
    e = torch.empty(shape)
    e[..., 0] = w[..., 0]
    c = (1 - rho ** 2) ** 0.5
    for t in range(1, shape[-1]):
        e[..., t] = rho * e[..., t - 1] + c * w[..., t]
    return e * sigma


def add_noise(hist: torch.Tensor, v2x: torch.Tensor, level: torch.Tensor | float,
              gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """hist (B, N, T, 4) clean [x, y, heading, v]; v2x (B, N) bool; level scalar or (B,) tensor.

    Returns noisy (B, N, T, 4) and fresh (B, N, T) float (1 = a new measurement arrived this step).
    """
    B, N, T, _ = hist.shape
    s = torch.as_tensor(level, dtype=torch.float32).reshape(-1, 1, 1).expand(B, 1, 1)
    sh = (B, N, T)
    v = v2x[..., None].float()                                                    # (B, N, 1)
    # sensor-tracked terms
    ex = _ar1(sh, SENS["rho"], SENS["pos"], gen)
    ey = _ar1(sh, SENS["rho"], SENS["pos"], gen)
    eh = _ar1(sh, SENS["rho"], SENS["head"], gen)
    ev = _ar1(sh, SENS["rho"], SENS["spd"], gen)
    # V2X terms
    bx = _ar1(sh, V2X["rho_pos"], V2X["pos"], gen) + torch.randn(sh, generator=gen) * V2X["white"]
    by = _ar1(sh, V2X["rho_pos"], V2X["pos"], gen) + torch.randn(sh, generator=gen) * V2X["white"]
    bh = _ar1(sh, V2X["rho"], V2X["head"], gen)
    bv = _ar1(sh, V2X["rho"], V2X["spd"], gen)
    e = torch.stack([(1 - v) * ex + v * bx, (1 - v) * ey + v * by, (1 - v) * eh + v * bh, (1 - v) * ev + v * bv], -1)
    noisy = hist + e * s[..., None]
    noisy[..., 2] = torch.atan2(torch.sin(noisy[..., 2]), torch.cos(noisy[..., 2]))
    noisy[..., 3] = noisy[..., 3].clamp(min=0)
    # packet loss for V2X agents: zero-order hold of the last received message
    p = (0.1 * s).clamp(max=0.5) * v                                              # (B, N, 1)
    lost = torch.rand(sh, generator=gen) < p
    lost[..., 0] = False                                                           # the first message always arrives
    fresh = (~lost).float()
    idx = torch.arange(T).expand(sh) * (~lost)                                     # index of last received message
    idx = torch.cummax(idx, -1).values
    noisy = noisy.gather(2, idx[..., None].expand(-1, -1, -1, 4))
    return noisy, fresh


def staleness(fresh: torch.Tensor) -> torch.Tensor:
    """Steps since the last received message at every step (B, N, T)."""
    T = fresh.shape[-1]
    t = torch.arange(T, dtype=torch.float32)
    last = torch.cummax(t * fresh, -1).values
    return t - last


if __name__ == "__main__":
    from .scene import T_H, make_dataset
    d = make_dataset(400, 1)
    h = torch.from_numpy(d["states"][:, :, :T_H])
    v2x = torch.from_numpy(d["v2x"])
    g = torch.Generator().manual_seed(0)
    for s in (0, 1, 2, 4):
        n, f = add_noise(h, v2x, s, g)
        err = (n[..., :2] - h[..., :2]).norm(dim=-1)[..., -1]
        print(f"level {s}: t0 pos error sensor {err[~v2x].mean():.2f} m, v2x {err[v2x].mean():.2f} m, "
              f"lost msgs {(1 - f[v2x]).mean():.2f}")
