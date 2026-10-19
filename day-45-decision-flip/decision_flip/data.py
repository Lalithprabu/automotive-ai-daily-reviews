"""Synthetic driving-scene generator: semantic voxel occupancy + noisy BEV 'camera' features.

Classes: 0 empty, 1 road, 2 vehicle, 3 building, 4 vegetation. Grid (X,Y,Z) = (16,16,4).
Features (C=6, X, Y): 4 per-class presence (max over z), normalised max height, noise channel.
Random 3x3 patches are zeroed to emulate occlusion, so the bank must use scene context.
"""
import numpy as np
import torch


def make_scene(seed, grid=(16, 16, 4)):
    X, Y, Z = grid
    r = np.random.RandomState(seed)
    occ = np.zeros((X, Y, Z), dtype=np.int64)
    # road corridor(s) at z=0
    y0 = r.randint(3, Y - 7); w = r.randint(3, 5)
    occ[:, y0:y0 + w, 0] = 1
    if r.rand() < 0.5:
        x0 = r.randint(3, X - 7)
        occ[x0:x0 + 3, :, 0] = 1
    road = occ[:, :, 0] == 1
    # vegetation patches (low) off-road
    for _ in range(r.randint(2, 5)):
        x, y = r.randint(0, X - 2), r.randint(0, Y - 2)
        if not road[x:x + 2, y:y + 2].any():
            occ[x:x + 2, y:y + 2, 0:r.randint(1, 3)] = 4
    # buildings (tall) off-road
    for _ in range(r.randint(2, 4)):
        x, y = r.randint(0, X - 3), r.randint(0, Y - 3)
        h = r.randint(2, Z + 1)
        if not road[x:x + 3, y:y + 3].any():
            occ[x:x + 3, y:y + 3, 0:h] = 3
    # vehicles on road, one voxel above the road surface
    for _ in range(r.randint(2, 5)):
        x, y = r.randint(0, X - 2), r.randint(0, Y - 1)
        if road[x:x + 2, y].all() and (occ[x:x + 2, y, 1] == 0).all():
            occ[x:x + 2, y, 1] = 2
    # BEV features
    feat = np.zeros((6, X, Y), dtype=np.float32)
    for k in range(4):
        feat[k] = (occ == k + 1).any(axis=2)
    h = np.where(occ > 0, np.arange(Z)[None, None, :] + 1, 0).max(axis=2)
    feat[4] = h / Z
    feat[5] = r.randn(X, Y) * 0.3
    feat[:5] += r.randn(5, X, Y).astype(np.float32) * 0.1
    for _ in range(2):  # occlusion patches
        x, y = r.randint(0, X - 3), r.randint(0, Y - 3)
        feat[:5, x:x + 3, y:y + 3] = 0.0
    return torch.from_numpy(feat), torch.from_numpy(occ)


def make_split(start_seed, n, grid=(16, 16, 4)):
    fs, os_ = zip(*[make_scene(start_seed + i, grid) for i in range(n)])
    return torch.stack(fs), torch.stack(os_)
