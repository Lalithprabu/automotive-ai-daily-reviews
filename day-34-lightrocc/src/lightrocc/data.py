"""Synthetic camera-scene generator for occupancy forecasting.

A scene = 1-4 vehicles (oriented boxes) moving with constant speed and a small yaw rate, seen by a static
forward camera. We render *camera feature maps* (not RGB) for 2 past frames (t-1, t) and voxelize the GT
box occupancy at t = 0..T_future. Everything here is this repo's own toy simulator (NOT the nuScenes data).

Camera feature channels (C=6):
  0 vehicle presence blob   1 blob*cos(heading)   2 blob*sin(heading)   3 blob*height
  4 static building clutter 5 smooth texture noise
  + ch0 also carries weak static "signboard" decoys (amplitude 0.35, no heading cues) as hard negatives.
Depth is NOT given: it must be inferred from blob size and vertical position (ground-plane cue).
"""
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import voxel_centers

C_IMG = 6
L_RANGE, W_RANGE, H_VEH = (3.8, 5.2), (1.7, 2.1), 1.6


def _blob(h, w, cu, cv, su, sv):
    uu = np.arange(w)[None, :]
    vv = np.arange(h)[:, None]
    return np.exp(-0.5 * (((uu - cu) / su) ** 2 + ((vv - cv) / sv) ** 2))


def sample_scene(rng, cfg):
    """Return dict with per-vehicle states for steps k=-1..T (index 0 is k=-1)."""
    T = cfg["world"]["t_future"]
    n = int(rng.integers(1, cfg["data"]["max_vehicles"] + 1))
    veh = []
    tries = 0
    while len(veh) < n and tries < 200:
        tries += 1
        r = rng.uniform(8.0, 27.0)
        b = rng.uniform(-0.48, 0.48)
        x0, y0 = r * math.cos(b), r * math.sin(b)
        if any(math.hypot(x0 - v["x"][1], y0 - v["y"][1]) < 6.0 for v in veh):
            continue
        speed = 0.0 if rng.random() < 0.2 else rng.uniform(1.0, 5.5)
        yaw = rng.uniform(-math.pi, math.pi)
        yr = rng.uniform(-0.08, 0.08) if speed > 0 else 0.0
        L, W = rng.uniform(*L_RANGE), rng.uniform(*W_RANGE)
        # state at k=0 is (x0,y0,yaw); roll backward one step for k=-1, forward for 1..T
        xs, ys, yaws = np.zeros(T + 2), np.zeros(T + 2), np.zeros(T + 2)
        xs[1], ys[1], yaws[1] = x0, y0, yaw
        xs[0], ys[0], yaws[0] = x0 - speed * 0.5 * math.cos(yaw - yr), y0 - speed * 0.5 * math.sin(yaw - yr), yaw - yr
        for k in range(1, T + 1):
            xs[k + 1] = xs[k] + speed * 0.5 * math.cos(yaws[k])
            ys[k + 1] = ys[k] + speed * 0.5 * math.sin(yaws[k])
            yaws[k + 1] = yaws[k] + yr
        veh.append(dict(x=xs, y=ys, yaw=yaws, L=L, W=W, H=H_VEH, speed=speed))
    return veh


def render_frame(veh, k_idx, cfg, rng_static):
    """Render one 6-channel camera feature map for step index k_idx (0 -> k=-1, 1 -> k=0)."""
    c = cfg["camera"]
    h, w, f = c["h"], c["w"], c["focal"]
    img = np.zeros((C_IMG, h, w), np.float32)
    # static clutter (same for both frames of a scene via rng_static)
    for _ in range(rng_static["n_build"]):
        img[4] += rng_static["build_amp"].pop() * _blob(h, w, *rng_static["build"].pop())
    for (cu, cv, su, sv) in rng_static["sign"]:
        img[0] += 0.35 * _blob(h, w, cu, cv, su, sv)
    img[5] = rng_static["tex"]
    for v in veh:
        x, y, yaw = v["x"][k_idx], v["y"][k_idx], v["yaw"][k_idx]
        if x < 2.0:
            continue
        cu = w / 2.0 - f * y / x
        cv = h / 2.0 + f * (c["cam_height"] - 0.8) / x
        lat = abs(v["L"] * math.sin(yaw)) + abs(v["W"] * math.cos(yaw))
        su, sv = max(0.8, f * lat / x / 2.5), max(0.8, f * v["H"] / x / 2.5)
        b = _blob(h, w, cu, cv, su, sv)
        img[0] += b
        img[1] += b * math.cos(yaw)
        img[2] += b * math.sin(yaw)
        img[3] += b * (v["H"] / 2.0)
    return img


def voxelize_boxes(veh, k_idx, vox):
    """GT occupancy [V] (float 0/1) of the union of oriented boxes at step index k_idx."""
    occ = torch.zeros(vox.shape[0])
    for v in veh:
        dx = vox[:, 0] - v["x"][k_idx]
        dy = vox[:, 1] - v["y"][k_idx]
        yaw = v["yaw"][k_idx]
        lx = math.cos(yaw) * dx + math.sin(yaw) * dy
        ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
        inside = (lx.abs() < v["L"] / 2) & (ly.abs() < v["W"] / 2) & (vox[:, 2] < v["H"])
        occ = torch.maximum(occ, inside.float())
    return occ


def make_scene(seed, cfg, vox):
    rng = np.random.default_rng(seed)
    veh = sample_scene(rng, cfg)
    c = cfg["camera"]
    stat = dict(
        n_build=int(rng.integers(3, 7)),
        build=[(rng.uniform(0, c["w"]), rng.uniform(4, 14), rng.uniform(1.5, 4), rng.uniform(2, 5)) for _ in range(6)],
        build_amp=[rng.uniform(0.4, 1.0) for _ in range(6)],
        sign=[(rng.uniform(0, c["w"]), rng.uniform(5, 14), 1.2, 1.2) for _ in range(int(rng.integers(0, 4)))],
        tex=rng.normal(0, 0.15, (c["h"], c["w"])).astype(np.float32),
    )
    import copy
    frames = []
    for k_idx in (0, 1):
        s = copy.deepcopy(stat)
        fr = render_frame(veh, k_idx, cfg, s)
        fr += rng.normal(0, 0.05, fr.shape).astype(np.float32)
        frames.append(fr)
    img = np.concatenate(frames, 0)  # [12, H, W]
    T = cfg["world"]["t_future"]
    occ = torch.stack([voxelize_boxes(veh, 1 + t, vox) for t in range(T + 1)])  # [T+1, V]
    nmax = cfg["data"]["max_vehicles"]
    cen = np.zeros((nmax, T + 1, 2), np.float32)
    yaw = np.zeros((nmax, T + 1), np.float32)
    size = np.zeros((nmax, 3), np.float32)
    valid = np.zeros(nmax, bool)
    for i, v in enumerate(veh):
        cen[i] = np.stack([v["x"][1:], v["y"][1:]], -1)
        yaw[i] = v["yaw"][1:]
        size[i] = [v["L"], v["W"], v["H"]]
        valid[i] = True
    return dict(img=torch.from_numpy(img), occ=occ, cen=torch.from_numpy(cen), yaw=torch.from_numpy(yaw),
                size=torch.from_numpy(size), valid=torch.from_numpy(valid))


class SceneSet(Dataset):
    def __init__(self, n, seed0, cfg):
        vox = voxel_centers(cfg)
        self.items = [make_scene(seed0 + i, cfg, vox) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
