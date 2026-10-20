"""Synthetic pedestrian-crossing scenes (this repo's own generator; NOT JAAD/PIE).

Each scene is 48 frames (5 Hz-ish). Crossers enter the road at frame 40. Three
partially-informative, independently-noisy cue channels carry the intent:
  pose   : torso lean toward the road + head turn + slowing gait   (ramps up from frame ~12)
  ego    : approaching vehicle decelerates (yielding)               (ramps up from frame ~12)
  visual : 6-d per-frame scene vector (curb distance, crosswalk flag, vehicle gap, noise x3)
Non-crossers get "decoy" cues of weaker strength, so every single modality is ambiguous.
"""
from __future__ import annotations
import numpy as np
import torch
from .model import NUM_JOINTS

TEMPLATE = np.array([
    (0, 1.70), (0, 1.50), (0.2, 1.45), (0.25, 1.15), (0.25, 0.9), (-0.2, 1.45), (-0.25, 1.15),
    (-0.25, 0.9), (0.1, 0.95), (0.1, 0.5), (0.1, 0.05), (-0.1, 0.95), (-0.1, 0.5), (-0.1, 0.05)],
    dtype=np.float32)


def make_scene(rng: np.random.Generator, intent: int, T: int = 48) -> dict:
    t = np.arange(T)
    r = np.clip((t - 12) / 24.0, 0, 1)                       # cue ramp
    cross = intent == 1
    cue_p = rng.uniform(0.2, 1.0) if cross else rng.uniform(0.0, 0.75)
    cue_e = rng.uniform(0.2, 1.0) if cross else rng.uniform(0.0, 0.75)
    cue_v = rng.uniform(0.2, 1.0) if cross else rng.uniform(0.0, 0.75)

    # --- pedestrian global motion: approach curb at x=0 from the left
    v0 = rng.uniform(0.9, 1.3)
    speed = v0 * (1 - 0.55 * cue_p * r)                      # slows when intending to cross
    x_hip = -4.0 + np.cumsum(speed) * 0.2
    if cross:
        x_hip = np.minimum(x_hip, -rng.uniform(0.3, 0.9))
        x_hip[40:] = x_hip[39] + np.arange(1, T - 39) * 0.2 * 1.2   # steps into road
    elif rng.random() < 0.5:                                  # walks along sidewalk, keeps distance
        x_hip = np.minimum(x_hip, -rng.uniform(0.6, 2.2))
    else:                                                     # stops at curb (decoy)
        x_hip = np.minimum(x_hip, -rng.uniform(0.3, 0.9))
    sp = np.gradient(x_hip) / 0.2                             # realised speed (m/s)
    # --- skeleton
    phase = np.cumsum(2 * np.pi * 0.9 * np.clip(sp, 0, 2) * 0.2)
    P = np.tile(TEMPLATE, (T, 1, 1))                          # (T,V,2)
    amp = 0.22 * np.clip(sp / 1.1, 0, 1.5)
    for j, sgn in ((9, 1), (10, 1.4), (7, -1), (6, -.6), (12, -1), (13, -1.4), (4, 1), (3, .6)):
        P[:, j, 0] += sgn * amp * np.sin(phase)
    lean = 0.35 * cue_p * r + rng.normal(0, 0.03, T)          # rad, toward road
    turn = 0.15 * cue_p * r + rng.normal(0, 0.01, T)
    P[:, :, 0] += lean[:, None] * P[:, :, 1] * 0.6
    P[:, 0, 0] += turn
    P += rng.normal(0, 0.015, P.shape)
    P[:, :, 0] += x_hip[:, None]
    # --- ego vehicle (speed, accel)
    ve = rng.uniform(8, 14) * (1 - 0.5 * cue_e * r) + rng.normal(0, 0.15, T)
    ego = np.stack([ve / 14.0, np.gradient(ve) * 5 / 14.0], -1).astype(np.float32)
    # --- visual context vector
    curb = (0.0 - x_hip) / 4.0
    cw = np.full(T, float(rng.random() < (0.35 + 0.4 * cue_v if cross else 0.65 - 0.4 * cue_v)))
    gap = np.clip(0.8 - 0.5 * cue_v * r * (1 if cross else 0.5), 0, 1) + rng.normal(0, 0.03, T)
    vis = np.stack([curb, cw, gap, *rng.normal(0, 0.3, (3, T))], -1).astype(np.float32)
    return dict(pose=P.astype(np.float32), ego=ego, vis=vis, intent=intent,
                x_hip=x_hip.astype(np.float32), ve=ve.astype(np.float32))


def window_tensors(scene: dict, end: int, W: int = 16):
    """Pose window ending at frame `end` (exclusive) -> (4,W,V), vis (W,6), ego (W,2).
    Pose x is centred on the window's mean hip x so absolute curb distance does not leak."""
    P = scene["pose"][end - W:end].copy()                     # (W,V,2)
    hip_x = P[:, [8, 11], 0].mean()
    P[:, :, 0] -= hip_x
    vel = np.zeros_like(P); vel[1:] = (P[1:] - P[:-1]) * 5.0
    # global velocity: include hip translation (centering removed position, not motion)
    hx = scene["pose"][end - W:end][:, [8, 11], 0].mean(1)
    vel[1:, :, 0] += ((hx[1:] - hx[:-1]) * 5.0)[:, None]
    pose = np.concatenate([P, vel], -1).transpose(2, 0, 1)    # (4,W,V)
    return pose, scene["vis"][end - W:end], scene["ego"][end - W:end]


def build_split(n: int, seed: int, W=16, T=48, end_range=(16, 40)):
    rng = np.random.default_rng(seed)
    poses, vis, ego, y = [], [], [], []
    for i in range(n):
        s = make_scene(rng, int(i % 2), T)
        end = int(rng.integers(end_range[0], end_range[1] + 1))
        p, v, e = window_tensors(s, end, W)
        poses.append(p); vis.append(v); ego.append(e); y.append(s["intent"])
    f = lambda a: torch.from_numpy(np.stack(a)).float()
    return f(poses), f(vis), f(ego), torch.tensor(y).float()
