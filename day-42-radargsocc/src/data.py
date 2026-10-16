"""Synthetic 4D-radar-tensor -> semantic-occupancy dataset (this repo's own construction, NOT K-Radar).

Grid is Cartesian (X forward, Y lateral, Z up): X=48, Y=32, Z=8 voxels.
Classes: 0 empty, 1 road, 2 building, 3 vegetation, 4 vehicle.
Radar input has 8 channels per voxel, mirroring the paper's sparsified tensor description:
  [p1, p2, p3, d1, d2, d3, mean_power, var_power]
(top-3 power values, their Doppler indices, mean + variance of power along the Doppler axis).
Physics baked in: static returns carry ego-motion Doppler (-v_ego*cos(azimuth)); moving vehicles deviate.
"""
import numpy as np, torch
from torch.utils.data import Dataset

X, Y, Z, NC = 48, 32, 8, 5
CLASS_NAMES = ["empty", "road", "building", "vegetation", "vehicle"]
REFLECT = {1: 0.30, 2: 0.7, 3: 0.55, 4: 0.75}   # mean radar reflectivity per class
NDOP = 16                                      # Doppler bins


def make_scene(rng, snow=0.0):
    occ = np.zeros((X, Y, Z), dtype=np.int64)
    occ[:, :, 0] = 1                                         # road plane
    vel = np.zeros((X, Y, Z), dtype=np.float32)              # radial-ish object speed field
    for _ in range(rng.integers(2, 5)):                      # buildings along sides
        side = rng.choice([0, 1]); x0 = rng.integers(0, X - 10); w = rng.integers(6, 14)
        y0 = rng.integers(0, 4) if side == 0 else Y - rng.integers(4, 8)
        occ[x0:x0 + w, y0:y0 + 4, 1:rng.integers(4, 8)] = 2
    for _ in range(rng.integers(3, 8)):                      # vegetation blobs
        cx, cy = rng.integers(0, X), rng.integers(0, Y); r = rng.integers(1, 3)
        xs, ys = np.ogrid[:X, :Y]
        m = ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r
        zt = rng.integers(2, 4)
        for z in range(1, zt): occ[:, :, z][m & (occ[:, :, z] == 0)] = 3
    for _ in range(rng.integers(1, 4)):                      # vehicles (mostly moving)
        x0, y0 = rng.integers(4, X - 6), rng.integers(8, Y - 8)
        reg = occ[x0:x0 + 4, y0:y0 + 2, 1:3]
        if (reg == 0).all():
            occ[x0:x0 + 4, y0:y0 + 2, 1:3] = 4
            vel[x0:x0 + 4, y0:y0 + 2, 1:3] = rng.uniform(-6, 6) if rng.random() < 0.8 else 0.0
    return occ, vel


def render_radar(occ, vel, rng, snow=0.0, v_ego=8.0):
    """Occupancy -> sparse 8-channel radar tensor (blur + noise + Doppler physics)."""
    refl = np.zeros(occ.shape, np.float32)
    for c, r in REFLECT.items(): refl[occ == c] = r
    # radar point-spread: smear power 1 voxel along range & lateral (angular resolution is coarse)
    p = refl.copy()
    for ax in (0, 1):
        for sh, wgt in ((1, 0.7), (2, 0.4)):
            p = np.maximum(p, wgt * np.roll(refl, sh, ax)); p = np.maximum(p, wgt * np.roll(refl, -sh, ax))
    p = p * rng.lognormal(0, 0.7, p.shape).astype(np.float32)
    p += rng.exponential(0.25, p.shape).astype(np.float32)   # strong clutter / floor (hard mode)
    if snow > 0:                                              # snow: attenuate + drop returns
        p *= (1 - 0.5 * snow); p[rng.random(p.shape) < snow * 0.5] *= 0.1
    xs = np.arange(X)[:, None, None] + 3.0; ys = (np.arange(Y)[None, :, None] - Y / 2)
    cosaz = xs / np.sqrt(xs ** 2 + ys ** 2 + 1e-6)
    dop_v = -v_ego * cosaz + vel * cosaz                      # radial velocity (m/s)
    dop_idx = np.clip(np.round(dop_v / 1.0 + NDOP / 2), 0, NDOP - 1)
    # top-3 returns along Doppler: main peak at dop_idx, two weaker neighbours (leakage)
    p1 = p; p2 = 0.45 * p * rng.uniform(0.6, 1.2, p.shape); p3 = 0.2 * p * rng.uniform(0.6, 1.2, p.shape)
    d1 = dop_idx; d2 = np.clip(dop_idx + rng.choice([-1, 1], p.shape), 0, NDOP - 1)
    d3 = np.clip(dop_idx + rng.choice([-2, 2], p.shape), 0, NDOP - 1)
    mean = (p1 + p2 + p3) / NDOP * 4; var = p1 ** 2 * 0.3
    t = np.stack([p1, p2, p3, d1 / NDOP, d2 / NDOP, d3 / NDOP, mean, var], 0).astype(np.float32)
    keep = (p1 > np.quantile(p1, 0.7))                       # sparsify: keep strongest ~30% cells
    return t * keep[None]


class RadarOccDataset(Dataset):
    def __init__(self, n, seed=0, snow=0.0):
        rng = np.random.default_rng(seed); self.items = []
        for _ in range(n):
            occ, vel = make_scene(rng); self.items.append((render_radar(occ, vel, rng, snow), occ))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        r, o = self.items[i]; return torch.from_numpy(r), torch.from_numpy(o)
