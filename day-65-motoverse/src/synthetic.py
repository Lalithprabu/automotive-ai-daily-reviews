"""Synthetic ego-rider scenes (MotoVerse-style *toy* generator, NOT the real benchmark).

World frame: x forward, y left, metres. The ego rider (a motorised two-wheeler) performs one of four
maneuvers; surrounding agents weave with weak lane structure. Everything is vectorised torch.
"""
import math
import torch

DT = 0.2          # s per frame
T_OBS = 30        # observable frames (6 s)
H = 10            # future frames to forecast (2 s)
T_ALL = T_OBS + H
K = 4             # surrounding agents
MANEUVERS = ["straight", "left", "right", "stop"]


def make_batch(B, gen=None, device="cpu"):
    g = dict(generator=gen)
    r = lambda *s: torch.rand(*s, **g)
    n = lambda *s: torch.randn(*s, **g)
    m = torch.randint(0, 4, (B,), **g)                              # maneuver label
    tc = torch.randint(18, 26, (B,), **g).float()[:, None]          # frame of the maneuver "event"
    t = torch.arange(T_ALL).float()[None]                           # (1,T)
    d = torch.where(m == 1, 1.0, torch.where(m == 2, -1.0, 0.0))[:, None]   # +left / -right
    is_stop = (m == 3).float()[:, None]
    v0 = 4 + 4 * r(B, 1)
    A = 0.5 + 0.4 * r(B, 1)
    # --- yaw rate: weak early precursor (drift) + gaussian bump at the event + dynamic ego sway ---------
    sway = 0.14 * torch.sin(0.9 * t + 6.28 * r(B, 1)) * (0.5 + r(B, 1))
    omega = d * (0.06 * t / tc).clamp(max=0.1) + d * A * torch.exp(-0.5 * ((t - tc) / 2.5) ** 2) + sway
    # --- speed: gentle early decel for stop/turn, strong at the event --------------------------------
    sig = torch.sigmoid((t - tc) / 1.5)
    v = v0 * (1 - is_stop * (0.25 * (t / tc).clamp(max=1) + 0.75 * sig)) * (1 - 0.3 * d.abs() * sig)
    yaw = torch.cumsum(omega * DT, 1) - omega * DT                  # yaw_t = sum_{s<t} w_s dt
    ex = torch.cumsum(v * torch.cos(yaw) * DT, 1) - v * torch.cos(yaw) * DT
    ey = torch.cumsum(v * torch.sin(yaw) * DT, 1) - v * torch.sin(yaw) * DT
    ego_pos = torch.stack([ex, ey], -1)                             # (B,T,2)
    # --- surrounding agents: weaving constant-speed movers --------------------------------------------
    ax0 = 5 + 30 * r(B, K); ay0 = -8 + 16 * r(B, K)
    head0 = 0.25 * n(B, K) + math.pi * (r(B, K) < 0.25).float()
    sp = 3 + 6 * r(B, K)
    ph = 6.28 * r(B, K)
    hd = head0[:, None] + 0.3 * torch.sin(0.5 * t[:, :, None] + ph[:, None])   # (B,T,K)
    ax = ax0[:, None] + torch.cumsum(sp[:, None] * torch.cos(hd) * DT, 1)
    ay = ay0[:, None] + torch.cumsum(sp[:, None] * torch.sin(hd) * DT, 1)
    agents_w = torch.stack([ax, ay], -1)                            # (B,T,K,2)
    ego_feat = torch.stack([v + 0.05 * n(B, T_ALL), omega + 0.02 * n(B, T_ALL)], -1)[:, :T_OBS]
    b = dict(ego=ego_feat, ego_pos=ego_pos, ego_yaw=yaw, agents_w=agents_w, label=m)
    b["agents_ego"] = to_ego_all(b)[:, :T_OBS] + 0.05 * n(B, T_OBS, K, 2)   # noisy ego-frame measurements
    return {k: x.to(device) for k, x in b.items()}


def _rot(p, yaw):
    """Rotate vectors p (..., 2) by -yaw (world -> body). yaw broadcastable to p[..., 0]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * p[..., 0] + s * p[..., 1], -s * p[..., 0] + c * p[..., 1]], -1)


def to_ego_all(b):
    """Every agent position expressed in the ego frame of its own frame index. (B,T,K,2)"""
    rel = b["agents_w"] - b["ego_pos"][:, :, None]
    return _rot(rel, b["ego_yaw"][:, :, None])


def to_frame(b, world_xy, idx):
    """Express world points (B,N,2) in the ego frame at frame index idx (B,)."""
    ar = torch.arange(world_xy.shape[0])
    pos, yaw = b["ego_pos"][ar, idx], b["ego_yaw"][ar, idx]
    return _rot(world_xy - pos[:, None], yaw[:, None])


def future_target(b, L):
    """Agent futures (B,K,H,2) in the ego frame of the last observed frame L-1 (L: (B,) long)."""
    B = L.shape[0]
    ar = torch.arange(B)
    idx = L[:, None] + torch.arange(H)[None]                        # (B,H) frames L..L+H-1
    fut = b["agents_w"][ar[:, None], idx]                           # (B,H,K,2)
    flat = fut.permute(0, 2, 1, 3).reshape(B, K * H, 2)
    return to_frame(b, flat, L - 1).view(B, K, H, 2)
