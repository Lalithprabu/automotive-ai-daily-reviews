import torch
from .data import CAR_RADIUS

def ade_fde(pred, gt):
    d = (pred - gt).norm(dim=-1)                 # (B, H)
    return d.mean(1), d[:, -1]

def collision(pred, lead, thresh=2 * CAR_RADIUS):
    """Collision if the predicted ego centre comes within `thresh` m of the lead centre at the same step."""
    return ((pred - lead).norm(dim=-1) < thresh).any(1)

def near_far_ade(pred, gt, split=4):
    d = (pred - gt).norm(dim=-1)
    return d[:, :split].mean(1), d[:, split:].mean(1)


def stale_momentum_extrapolation(hist, horizon=12, dt=0.5):
    """'Physics old way': persist the observed speed/acceleration/curvature trend, ignoring the scene.

    Uses only the history tensor (features: x/20, y/20, cos, sin, v/10, ...).  Constant acceleration
    (clipped at v>=0) along a constant-curvature arc.  This is exactly the failure mode a reset gate
    is meant to cure: when a lead vehicle appears the trend is outdated, but this keeps extrapolating.
    """
    import torch
    v = hist[..., 4] * 10.0                                  # (B,T)
    a = (v[:, -1] - v[:, -4]) / (3 * dt)
    th_prev = torch.atan2(hist[:, -2, 3], hist[:, -2, 2])
    s_prev = -(hist[:, -2, :2] * 20.0).norm(dim=-1).clamp(min=1e-3)
    kappa = th_prev / s_prev
    vk, s, out = v[:, -1], torch.zeros_like(v[:, -1]), []
    for _ in range(horizon):
        v_new = (vk + a * dt).clamp(min=0.0)
        s = s + 0.5 * (vk + v_new) * dt; vk = v_new
        small = kappa.abs() < 1e-6; k = torch.where(small, torch.ones_like(kappa), kappa)
        x = torch.where(small, s, torch.sin(k * s) / k); y = torch.where(small, torch.zeros_like(s), (1 - torch.cos(k * s)) / k)
        out.append(torch.stack([x, y], -1))
    return torch.stack(out, 1)
