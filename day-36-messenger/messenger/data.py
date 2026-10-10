"""Synthetic driving scene-flow sequences (this repo's own simulator, NOT nuScenes / Argoverse 2).

Design goal: a task where pairwise two-frame flow is genuinely ambiguous and where the
*history* of flows carries real predictive signal.
  * 2-4 rigid "vehicles" (box outlines) with time-varying acceleration + yaw rate
    (sinusoidal, random phase) -> constant-velocity carry-over of the last flow is wrong
    by a known, non-trivial margin.
  * Every frame RE-SAMPLES points on the outlines (no point correspondences across frames),
    so a two-frame estimator suffers from the aperture problem along straight box edges.
  * Static background points (ego-motion already compensated) have zero flow.
  * Point order is shuffled every frame so index never leaks object identity.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

DT = 0.5          # seconds between frames (nuScenes keyframe-like 2 Hz)
AREA = 40.0       # half-extent of the scene (m)


def _pose_apply(pose, u):
    """pose (x,y,th); u (n,2) object-frame -> world (n,2)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    return u @ R.T + pose[:2]


def _perimeter_points(rng, L, W, n):
    """n uniform samples along a rectangle outline, object frame."""
    per = 2 * (L + W)
    t = rng.uniform(0, per, n)
    u = np.zeros((n, 2))
    for i, ti in enumerate(t):
        if ti < L:            u[i] = (-L / 2 + ti, -W / 2)
        elif ti < L + W:      u[i] = (L / 2, -W / 2 + (ti - L))
        elif ti < 2 * L + W:  u[i] = (L / 2 - (ti - L - W), W / 2)
        else:                 u[i] = (-L / 2, W / 2 - (ti - 2 * L - W))
    return u


def make_sequence(rng, T_obs=6, H=4, N=192, n_per=40, noise=0.05):
    """Returns dict of numpy arrays.
    pts  (T_obs,N,2)   observed points per frame (shuffled, noisy)
    disp (T_obs,N,H,2) clean GT displacement of each point at horizons 1..H (zero for background)
    lab  (T_obs,N)     object index, -1 for background
    poses(T,M,3), sizes(M,2)  for rendering only
    """
    T = T_obs + H
    M = int(rng.integers(2, 5))
    pos = rng.uniform(-25, 25, (M, 2))
    speed = rng.uniform(3, 10, M)
    head = rng.uniform(0, 2 * np.pi, M)
    a0, aamp = rng.uniform(-1.5, 1.5, M), rng.uniform(0.5, 3.0, M)
    aw, aph = rng.uniform(0.5, 1.3, M), rng.uniform(0, 2 * np.pi, M)
    y0, yamp = rng.uniform(-0.25, 0.25, M), rng.uniform(0.0, 0.4, M)
    yw, yph = rng.uniform(0.5, 1.3, M), rng.uniform(0, 2 * np.pi, M)
    sizes = np.stack([rng.uniform(3.5, 5.0, M), rng.uniform(1.7, 2.1, M)], 1)
    poses = np.zeros((T, M, 3))
    for k in range(T):
        poses[k, :, :2], poses[k, :, 2] = pos, head
        acc = a0 + aamp * np.sin(aw * k + aph)
        yaw = y0 + yamp * np.sin(yw * k + yph)
        speed = np.clip(speed + acc * DT, 0.0, 14.0)
        head = head + yaw * DT
        pos = pos + speed[:, None] * np.stack([np.cos(head), np.sin(head)], 1) * DT
    n_bg = N - M * n_per
    pts = np.zeros((T_obs, N, 2), np.float32)
    disp = np.zeros((T_obs, N, H, 2), np.float32)
    lab = np.full((T_obs, N), -1, np.int64)
    for t in range(T_obs):
        P, D, Lb = [], [], []
        for m in range(M):
            u = _perimeter_points(rng, sizes[m, 0], sizes[m, 1], n_per)
            w0 = _pose_apply(poses[t, m], u)
            P.append(w0 + rng.normal(0, noise, w0.shape))
            D.append(np.stack([_pose_apply(poses[t + h, m], u) - w0 for h in range(1, H + 1)], 1))
            Lb.append(np.full(n_per, m))
        P.append(rng.uniform(-AREA, AREA, (n_bg, 2)))
        D.append(np.zeros((n_bg, H, 2)))
        Lb.append(np.full(n_bg, -1))
        P, D, Lb = np.concatenate(P), np.concatenate(D), np.concatenate(Lb)
        perm = rng.permutation(N)
        pts[t], disp[t], lab[t] = P[perm], D[perm], Lb[perm]
    return dict(pts=pts, disp=disp, lab=lab, poses=poses.astype(np.float32), sizes=sizes.astype(np.float32), M=M)


class SceneFlowDataset(Dataset):
    def __init__(self, n, seed, **kw):
        rng = np.random.default_rng(seed)
        self.items = [make_sequence(rng, **kw) for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return dict(pts=torch.from_numpy(it["pts"]), disp=torch.from_numpy(it["disp"]),
                    lab=torch.from_numpy(it["lab"]))


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
