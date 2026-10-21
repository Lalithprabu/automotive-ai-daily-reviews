"""Synthetic asynchronous-collaborative-perception world (this repo's own construction).

A collaborator vehicle drives past while 3-6 independent objects move (CTRA-like: accel + yaw rate).
Everything is expressed in the collaborator's CURRENT ("now") frame. The collaborator's message is a
BEV feature map rendered `delay` steps ago in ITS OWN (older) frame, plus k-1 earlier history frames.
The ego vehicle only sees the left half of the grid (x<0); the right half is its blind zone and can
only be detected through the collaborator's (stale) message.
"""
import math
import numpy as np
import torch
from .warp import rigid_warp

C_FEAT = 8
GRID = 32
CELL = 2.0
HALF = GRID * CELL / 2.0   # 32 m
DT = 0.15                  # seconds per step
MAX_DELAY = 5              # 0.75 s
MAX_OBJ = 6
SIGMA = 1.2                # cells


def _signatures(n_cls=3, seed=1234):
    g = np.random.RandomState(seed)
    sig = g.randn(n_cls, C_FEAT)
    sig /= np.linalg.norm(sig, axis=1, keepdims=True)
    return torch.tensor(sig * 1.5, dtype=torch.float32)


SIGS = _signatures()
_cc = (torch.arange(GRID, dtype=torch.float32) + 0.5) * CELL - HALF
YY, XX = torch.meshgrid(_cc, _cc, indexing="ij")  # rows=y, cols=x


def render(pos, cls, mask, noise=0.0, gen=None):
    """pos [B,N,2] metres in the frame being rendered; returns [B,C,H,W]."""
    d2 = (XX[None, None] - pos[..., 0, None, None]) ** 2 + (YY[None, None] - pos[..., 1, None, None]) ** 2
    blob = torch.exp(-d2 / (2 * (SIGMA * CELL) ** 2)) * mask[..., None, None]  # [B,N,H,W]
    sig = SIGS[cls]  # [B,N,C]
    f = torch.einsum("bnc,bnhw->bchw", sig, blob)
    if noise > 0:
        f = f + noise * torch.randn(f.shape, generator=gen)
    return f


def heatmap(pos, mask):
    """Gaussian heat at object centres, exact centre cell forced to 1.0 (CenterNet convention)."""
    d2 = (XX[None, None] - pos[..., 0, None, None]) ** 2 + (YY[None, None] - pos[..., 1, None, None]) ** 2
    h = torch.exp(-d2 / (2 * (0.8 * CELL) ** 2)) * mask[..., None, None]
    heat = h.max(1).values[:, None]
    col = ((pos[..., 0] + HALF) / CELL).floor().long().clamp(0, GRID - 1)
    row = ((pos[..., 1] + HALF) / CELL).floor().long().clamp(0, GRID - 1)
    for b in range(pos.shape[0]):
        for n in range(pos.shape[1]):
            if mask[b, n] > 0:
                heat[b, 0, row[b, n], col[b, n]] = 1.0
    return heat


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def make_batch(B, seed, k_hist=4, delays=None, ego_noise=0.03, collab_noise=0.03):
    """Returns dict of tensors. Deterministic in `seed`."""
    rng = np.random.RandomState(seed)
    gen = torch.Generator().manual_seed(seed)
    if delays is None:
        delays = rng.randint(0, MAX_DELAY + 1, size=B)
    delays = np.asarray(delays)
    T = MAX_DELAY + k_hist  # steps back we need
    pos_t = np.zeros((B, MAX_OBJ, 2)); mask = np.zeros((B, MAX_OBJ)); cls = rng.randint(0, 3, (B, MAX_OBJ))
    paths = np.zeros((B, MAX_OBJ, T, 2))  # object pos at t - j*dt in "now" frame, j=0..T-1
    for b in range(B):
        n = rng.randint(3, MAX_OBJ + 1)
        mask[b, :n] = 1
        for i in range(n):
            p = rng.uniform(-24, 24, 2)
            v = rng.choice([0.0, rng.uniform(1.5, 8.0)], p=[0.1, 0.9])
            th = rng.uniform(-np.pi, np.pi); a = rng.uniform(-2.5, 2.5); w = rng.uniform(-0.7, 0.7)
            paths[b, i, 0] = p
            for j in range(1, T):  # backward integration
                v_prev = v
                p = p - v_prev * np.array([np.cos(th), np.sin(th)]) * DT
                v = max(0.0, v - a * DT); th = th - w * DT
                paths[b, i, j] = p
        pos_t[b] = paths[b, :, 0]
    # collaborator ego-motion (in the "now" frame): poses at t - j*dt
    vc = rng.uniform(4, 10, B); wc = rng.uniform(-0.3, 0.3, B)
    cpos = np.zeros((B, T, 2)); cyaw = np.zeros((B, T))
    for b in range(B):
        p = np.zeros(2); yaw = 0.0
        for j in range(1, T):
            yaw_prev = yaw
            p = p - vc[b] * np.array([np.cos(yaw_prev), np.sin(yaw_prev)]) * DT
            yaw = yaw - wc[b] * DT
            cpos[b, j] = p; cyaw[b, j] = yaw
    # history frames: index i=0 is the most recent stale frame (time t - delay), i>0 older
    hist = torch.zeros(B, k_hist, C_FEAT, GRID, GRID); rel = torch.zeros(B, k_hist, 3)
    cls_t = torch.tensor(cls); mask_t = torch.tensor(mask, dtype=torch.float32)
    for i in range(k_hist):
        pq = np.zeros((B, MAX_OBJ, 2)); r = np.zeros((B, 3))
        for b in range(B):
            j = int(min(delays[b] + i, T - 1))
            Rinv = _rot(-cyaw[b, j])
            pq[b] = (paths[b, :, j] - cpos[b, j]) @ Rinv.T       # object in stale frame
            ts = Rinv @ (np.zeros(2) - cpos[b, j])                 # q_stale = R(dyaw) q_now + ts
            r[b] = [ts[0], ts[1], 0.0 - cyaw[b, j]]                # dyaw = yaw_now - yaw_stale
        hist[:, i] = render(torch.tensor(pq, dtype=torch.float32), cls_t, mask_t, collab_noise, gen)
        rel[:, i] = torch.tensor(r, dtype=torch.float32)
    pos_now = torch.tensor(pos_t, dtype=torch.float32)
    target = render(pos_now, cls_t, mask_t, 0.0)  # noise-free fresh collaborator feature (training target only)
    vis = (pos_now[..., 0] < 0).float() * mask_t
    ego = render(pos_now, cls_t, vis, ego_noise, gen)
    return dict(hist=hist, rel=rel, delay=torch.tensor(delays, dtype=torch.float32), target=target, ego=ego,
                heat=heatmap(pos_now, mask_t), pos=pos_now, mask=mask_t, cls=cls_t,
                paths=torch.tensor(paths, dtype=torch.float32), cpos=torch.tensor(cpos, dtype=torch.float32),
                cyaw=torch.tensor(cyaw, dtype=torch.float32))


def warp_history(hist, rel):
    """Pose-warp every history frame into the current frame: [B,K,C,H,W]."""
    B, K = hist.shape[:2]
    out = rigid_warp(hist.flatten(0, 1), rel.flatten(0, 1), HALF)
    return out.view(B, K, *out.shape[1:])
