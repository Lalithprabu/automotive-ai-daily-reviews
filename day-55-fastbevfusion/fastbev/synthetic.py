"""Synthetic LiDAR + camera world for a FastBEVFusion-style detector (this repo's own toy, NOT nuScenes).

Geometry (ego at the bottom-centre of the BEV grid, looking +row):
    BEV grid  : G x G cells, CELL metres each.  row i -> forward distance, col j -> lateral offset.
    LiDAR     : sparse points; count per object falls ~ 1/range^2  (far objects get 0-2 points).
                Low clutter + tall 'pole' clutter (object-like height, no vehicle in the camera) are added.
    Camera    : a 1-D "image" strip over the azimuth (W_IMG columns across +-FOV/2).  It has NO depth:
                an object is a bump at its azimuth.  Poster-like distractors (bumps with no 3-D object) are added.
This mirrors the paper's core difficulty: image features are semi-aligned with the LiDAR BEV (right azimuth, unknown depth).
"""
import numpy as np
import torch

G, CELL = 32, 1.5                 # 32 x 32 cells, 1.5 m  ->  48 m forward, +-24 m lateral
FOV = np.deg2rad(140.0)           # camera field of view
W_IMG = 96                        # image columns
C_IMG = 3                         # image channels
OBJ_W = 2.2                       # object width (m)
SIGMA = 1.0                       # heatmap gaussian sigma (cells)


def cell_xy():
    """Metric (forward, lateral) of each BEV cell centre: two [G,G] arrays."""
    i, j = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    return (i + 0.5) * CELL, (j + 0.5 - G / 2) * CELL


def azimuth_map():
    """[G,G] azimuth (rad) of every BEV cell as seen from the ego (0 = straight ahead, + = left)."""
    f, l = cell_xy()
    return np.arctan2(-l, f)


def sample_objects(rng, n=None):
    """Random objects: array [n,3] = (forward m, lateral m, appearance 0..1)."""
    n = n or rng.integers(3, 9)
    fwd = rng.uniform(5, 46, n)
    lat = rng.uniform(-21, 21, n)
    return np.stack([fwd, lat, rng.uniform(0.3, 1.0, n)], 1)


def render(objs, rng, n_clutter=40, n_posters=None, n_poles=None):
    """objs [n,3] -> (lidar [2,G,G], img [C_IMG,W_IMG]).
    lidar ch0 = log1p(point count) / 2 ; ch1 = mean point height (objects tall, clutter low)."""
    lidar_cnt = np.zeros((G, G)); lidar_h = np.zeros((G, G))
    pts = []
    for f, l, _ in objs:
        r = np.hypot(f, l)
        k = rng.poisson(max(0.0, 2600.0 / (r * r)))            # ~26 pts @10 m, ~1.6 @40 m
        for _ in range(k):
            pts.append((f + rng.normal(0, .7), l + rng.normal(0, .5), rng.normal(1.3, .3)))
    for _ in range(n_clutter):
        pts.append((rng.uniform(0, G * CELL), rng.uniform(-G * CELL / 2, G * CELL / 2), abs(rng.normal(0.1, .1))))
    # tall clutter (poles / bushes): LiDAR looks object-like (height ~1.3 m, 1-3 pts) but the camera sees no vehicle there
    for _ in range(n_poles if n_poles is not None else rng.integers(8, 14)):
        pf, pl = rng.uniform(3, G * CELL), rng.uniform(-G * CELL / 2, G * CELL / 2)
        for _ in range(rng.integers(1, 4)):
            pts.append((pf + rng.normal(0, .3), pl + rng.normal(0, .3), rng.normal(1.3, .3)))
    for f, l, h in pts:
        i, j = int(f / CELL), int(l / CELL + G / 2)
        if 0 <= i < G and 0 <= j < G:
            lidar_cnt[i, j] += 1; lidar_h[i, j] += h
    lidar = np.stack([np.log1p(lidar_cnt) / 2, lidar_h / np.maximum(lidar_cnt, 1)]).astype(np.float32)

    img = rng.normal(0, 0.08, (C_IMG, W_IMG)).astype(np.float32)
    bins = (np.arange(W_IMG) + .5) / W_IMG * FOV - FOV / 2              # azimuth of each image column
    def bump(az, width_rad, app):
        m = np.exp(-0.5 * ((bins - az) / max(width_rad / 2, 1e-3)) ** 2)
        img[:] += (m[None] * np.array([app, 1 - app, 0.6])[:, None]).astype(np.float32)
    for f, l, a in objs:
        bump(np.arctan2(-l, f), OBJ_W / np.hypot(f, l), a)
    for _ in range(n_posters if n_posters is not None else rng.integers(1, 4)):        # camera-only distractors
        bump(rng.uniform(-FOV / 2, FOV / 2), rng.uniform(.02, .06), rng.uniform(.3, 1))
    return lidar, img


def heatmap(objs):
    """Gaussian centre heatmap [1,G,G] (CenterNet-style)."""
    f, l = cell_xy()
    hm = np.zeros((G, G), np.float32)
    for of, ol, _ in objs:
        d2 = ((f - of) / CELL) ** 2 + ((l - ol) / CELL) ** 2
        hm = np.maximum(hm, np.exp(-d2 / (2 * SIGMA ** 2)))
    return hm[None]


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    L, I, H, O = [], [], [], []
    for _ in range(n):
        o = sample_objects(rng); l, i = render(o, rng)
        L.append(l); I.append(i); H.append(heatmap(o)); O.append(o)
    return torch.tensor(np.stack(L)), torch.tensor(np.stack(I)), torch.tensor(np.stack(H)), O
