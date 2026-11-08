"""Synthetic driving-scene task: context encodes a lead vehicle's recent track
(and a latent manoeuvre); target = its next T waypoints (curved/straight/braking)."""
import numpy as np, torch

T = 16

def make_traj(rng, kind=None, n=T + 8):
    kind = kind if kind is not None else rng.integers(0, 3)
    x0, y0 = rng.uniform(-1, 1, 2)
    v = rng.uniform(0.4, 1.0)
    ang = rng.uniform(-0.3, 0.3)
    pts = []
    x, y, th, sp = x0, y0, ang, v
    for _ in range(n):
        if kind == 1: th += 0.12            # left curve
        if kind == 2: sp *= 0.93            # braking
        x += sp * np.cos(th) * 0.1; y += sp * np.sin(th) * 0.1
        pts.append((x, y))
    return np.array(pts, dtype=np.float32), kind

def ctx_from_hist(hist):
    c = np.zeros((8, 16), np.float32)
    c[:, :2] = hist; c[:, 2:4] = np.diff(hist, axis=0, prepend=hist[:1]) * 5
    return c

def batch(n, seed):
    rng = np.random.default_rng(seed)
    ctxs, tgts = [], []
    for _ in range(n):
        p, k = make_traj(rng)
        hist, fut = p[:8], p[8:]
        c = np.zeros((8, 16), np.float32)
        c[:, :2] = hist; c[:, 2:4] = np.diff(hist, axis=0, prepend=hist[:1]) * 5
        ctxs.append(c)
        tgts.append(fut - hist[-1:])   # positions relative to last observed point
    return torch.tensor(np.stack(ctxs)), torch.tensor(np.stack(tgts))
