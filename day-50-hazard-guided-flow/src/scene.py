"""Synthetic driving scene + exact obstacle geometry shared by data, energy, sim and eval."""
import math
import numpy as np
import torch

H, DT = 20, 0.25
X_SCALE, Y_SCALE = 50.0, 5.0          # normalisation of waypoint coordinates (metres -> ~unit range)
EGO_FRONT, EGO_HALF_W = 3.0, 1.0       # ego front overhang from the reference point, half width
OBS_LEN, OBS_HALF_W = 4.5, 1.0         # static obstacle footprint
A_COMFORT, A_MAX = 3.0, 7.0            # m/s^2

def inflated_box(d, y0=0.0):
    """Obstacle footprint Minkowski-inflated by the ego footprint -> a point-vs-box test for waypoints.
    Returns (xmin, xmax, ymin, ymax) tensors/floats. Works for scalars or tensors."""
    return d - EGO_FRONT, d + OBS_LEN + 1.0, y0 - OBS_HALF_W - EGO_HALF_W, y0 + OBS_HALF_W + EGO_HALF_W

def box_sdf(x, y, box):
    """Signed distance (m) from points to an axis-aligned box. Negative inside. Differentiable."""
    xmin, xmax, ymin, ymax = box
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    hx, hy = (xmax - xmin) / 2, (ymax - ymin) / 2
    qx, qy = (x - cx).abs() - hx, (y - cy).abs() - hy
    outside = torch.sqrt(torch.clamp(qx, min=0) ** 2 + torch.clamp(qy, min=0) ** 2 + 1e-9)
    inside = torch.clamp(torch.maximum(qx, qy), max=0)
    return outside + inside

def decode(traj_n):
    """(B, H*2) normalised -> (B, H, 2) metres."""
    t = traj_n.view(traj_n.shape[0], -1, 2)
    return torch.stack([t[..., 0] * X_SCALE, t[..., 1] * Y_SCALE], -1)

def encode(traj_m):
    return torch.stack([traj_m[..., 0] / X_SCALE, traj_m[..., 1] / Y_SCALE], -1).flatten(1)

def make_context(v_hist, d, present):
    """Context vector (B, 7): 4 past speeds, hazard flag, hazard distance, speed-dependent time-to-hazard.
    This is the 'hazard is encoded in the representation' part of the paper's story:
    the information is present, the generator just under-uses it."""
    present = present.float()
    dn = torch.where(present > 0, d / 60.0, torch.zeros_like(d))
    ttc = torch.where(present > 0, torch.clamp(d / (v_hist[:, -1] + 0.5), max=10.0) / 10.0, torch.zeros_like(d))
    return torch.cat([v_hist / 15.0, present[:, None], dn[:, None], ttc[:, None]], 1)

def stopping_distance(v, a=A_COMFORT):
    return v * v / (2 * a)

def demo_trajectory(v0, d, present, rng):
    """Expert demonstration: cruise with small accel noise; if a hazard is present AND needs it, brake at ~A_COMFORT
    to stop EGO_FRONT+2 m before the obstacle. Returns (H,2) metres."""
    v, s, out = v0, 0.0, []
    acc_noise = rng.normal(0, 0.15)
    for k in range(H):
        a = acc_noise
        if present:
            gap = d - EGO_FRONT - 2.0 - s
            need = stopping_distance(v, A_COMFORT)
            if gap < need * 1.15 + 1.0:
                a = -min(A_MAX, max(0.5, v * v / (2 * max(gap, 0.5))))
        v = max(0.0, v + a * DT)
        s += v * DT
        out.append((s, rng.normal(0, 0.03) * (1 + k / H)))
    return np.array(out, dtype=np.float32)

def sample_scene(rng, hazard):
    v0 = float(rng.uniform(4, 14))
    vh = np.clip(v0 + rng.normal(0, 0.1, 4).cumsum() * 0.2, 0, 16).astype(np.float32)
    vh[-1] = v0
    d = float(rng.uniform(6, 55)) if hazard else 0.0
    return v0, vh, d
