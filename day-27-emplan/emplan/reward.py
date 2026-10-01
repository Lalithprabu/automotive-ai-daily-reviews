"""Rule-based reward (the 'unpaired preference' signal). Differentiability is NOT needed:
the reward only labels candidates good/bad, it never carries gradients.

Scene vector layout (SCENE_DIM=9):
 0 ego speed v0 | 1 lead dist d1 | 2 lead speed v1 | 3 left-vehicle present | 4 left dist d2
 5 left speed v2 | 6 right-vehicle present | 7 right dist d3 | 8 right speed v3
Lanes are y in {-3.5, 0, 3.5}; road edge at |y|=5.25. Lead vehicle is always in the ego lane (y=0).
"""
import torch

SCENE_DIM = 9
CAR_L, CAR_W = 4.6, 1.9


def _obstacles(scene):
    # returns list of (present[B], x0[B], y[float], v[B])
    B = scene.shape[0]
    ones = torch.ones(B, device=scene.device)
    return [
        (ones, scene[:, 1], 0.0, scene[:, 2]),
        (scene[:, 3], scene[:, 4], 3.5, scene[:, 5]),
        (scene[:, 6], scene[:, 7], -3.5, scene[:, 8]),
    ]


def rule_terms(scene, traj, dt=0.5):
    """scene (B,9); traj (B,K,T,2) in ego frame, waypoint t at time (t+1)*dt.
    Returns dict of (B,K) tensors: collision(0/1), offroad(0/1), progress, jerk."""
    B, K, T, _ = traj.shape
    t = (torch.arange(T, device=traj.device, dtype=traj.dtype) + 1) * dt
    coll = torch.zeros(B, K, device=traj.device)
    for present, x0, y, v in _obstacles(scene):
        ox = x0[:, None] + v[:, None] * t[None]                      # (B,T)
        dx = (traj[..., 0] - ox[:, None]).abs()                      # (B,K,T)
        dy = (traj[..., 1] - y).abs()
        hit = ((dx < CAR_L + 0.5) & (dy < CAR_W + 0.3)).any(-1).float()
        coll = torch.maximum(coll, hit * present[:, None])
    off = (traj[..., 1].abs() > 5.25).any(-1).float()
    full = torch.cat([torch.zeros(B, K, 1, 2, device=traj.device), traj], 2)
    vel = (full[:, :, 1:] - full[:, :, :-1]) / dt
    acc = (vel[:, :, 1:] - vel[:, :, :-1]) / dt
    jerk = ((acc[:, :, 1:] - acc[:, :, :-1]) / dt).pow(2).sum(-1).mean(-1).sqrt()
    progress = traj[..., -1, 0] / (scene[:, 0:1] * T * dt * 1.2 + 1e-3)
    backwards = (vel[..., 0] < -0.1).any(-1).float()
    return dict(collision=coll, offroad=off, progress=progress.clamp(0, 1.2), jerk=jerk, backwards=backwards)


def rule_reward(scene, traj, dt=0.5):
    """Scalar reward (B,K): collisions/off-road dominate, then progress, then comfort."""
    r = rule_terms(scene, traj, dt)
    return (-10.0 * r["collision"] - 10.0 * r["offroad"] - 5.0 * r["backwards"]
            + 3.0 * r["progress"] - 0.05 * r["jerk"])
