"""Synthetic BEV occupancy world (a TOY stand-in for nuScenes / Occ3D -- NOT the FreeOcc data).

Scene = static discs (parked cars / poles) + moving discs (traffic) in the world frame.
The ego vehicle drives an arc (speed v, yaw-rate w) -- this is the 'planner-intended future motion' that
FreeOcc conditions on.  Occupancy at any time is rasterised in the EGO frame of that time.
Timeline: HIST history frames ending at t=0, then H future frames, DT seconds apart.
"""
import math
import torch
from .geometry import G, CELL, cell_centres, rel_pose

DT, HIST, H = 0.5, 4, 6
N_MOV, N_STAT = 8, 16


def make_scene(B, gen=None, n_frames=HIST + H):
    r = lambda *s: torch.rand(*s, generator=gen)
    v = 3 + 5 * r(B)                                    # ego speed  m/s
    w = (r(B) - 0.5) * 0.9                              # ego yaw rate rad/s (up to ~26 deg/s)
    # world positions of objects scattered around the ego start
    mov_p = (r(B, N_MOV, 2) - 0.5) * 70
    mov_a = r(B, N_MOV) * 6.283
    mov_v = 1 + 5 * r(B, N_MOV)
    mov_r = 1.2 + 1.0 * r(B, N_MOV)
    stat_p = (r(B, N_STAT, 2) - 0.5) * 80
    stat_r = 0.8 + 1.2 * r(B, N_STAT)
    return dict(v=v, w=w, mov_p=mov_p, mov_vel=torch.stack([mov_v * torch.cos(mov_a), mov_v * torch.sin(mov_a)], -1),
                mov_r=mov_r, stat_p=stat_p, stat_r=stat_r)


def ego_pose(sc, t):
    """Ego world pose at times t (T,) -> (B, T, 3).  Arc of constant speed / yaw-rate, starting at the origin."""
    v, w = sc["v"][:, None], sc["w"][:, None]
    t = t[None]
    wt = w * t
    small = w.abs() < 1e-4
    sw = torch.where(small, torch.ones_like(w), w)
    x = torch.where(small, v * t, v / sw * torch.sin(wt))
    y = torch.where(small, torch.zeros_like(wt), v / sw * (1 - torch.cos(wt)))
    return torch.stack([x, y, wt], -1)


def render(sc, t, soft=False):
    """Occupancy (B, T, G, G) in the ego frame of every time in t (T,)."""
    pose = ego_pose(sc, t)                                              # (B,T,3)
    cc = cell_centres()                                                 # (G,G,2)
    c, s = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
    # cell centres in world coords: p + R q          -> (B,T,G,G,2)
    wx = pose[..., 0, None, None] + c[..., None, None] * cc[..., 0] - s[..., None, None] * cc[..., 1]
    wy = pose[..., 1, None, None] + s[..., None, None] * cc[..., 0] + c[..., None, None] * cc[..., 1]
    wxy = torch.stack([wx, wy], -1)
    occ = torch.zeros(wx.shape)
    # movers: p(t) = p0 + vel * t
    mp = sc["mov_p"][:, None] + sc["mov_vel"][:, None] * t[None, :, None, None]       # (B,T,N,2)
    for objs, rad in ((mp, sc["mov_r"]), (sc["stat_p"][:, None].expand(-1, len(t), -1, -1), sc["stat_r"])):
        d = (wxy[:, :, None] - objs[:, :, :, None, None]).norm(dim=-1)                # (B,T,N,G,G)
        hit = (d < rad[:, None, :, None, None]).any(2)
        occ = torch.maximum(occ, hit.float())
    return occ


def make_batch(B, gen=None, jitter=0.0):
    """jitter (s): the history frames are CAPTURED at t+eps (eps~N(0,jitter)) but stamped/odometry-aligned at nominal t."""
    sc = make_scene(B, gen)
    t_all = (torch.arange(HIST + H).float() - (HIST - 1)) * DT           # -1.5 ... 3.0
    occ = render(sc, t_all)                                             # (B,10,G,G)
    if jitter > 0:
        hist_occ = torch.stack([render(sc, t_all[i:i + 1] + jitter * torch.randn(1, generator=gen))[:, 0] for i in range(HIST - 1)], 1)
        occ = torch.cat([hist_occ, occ[:, HIST - 1:]], 1)               # newest frame (t=0) is exact
    pose = ego_pose(sc, t_all)                                          # (B,10,3)
    cur = pose[:, HIST - 1]                                             # frame at t=0
    return dict(sc=sc, occ=occ, pose=pose, t_all=t_all,
                hist_gt=occ[:, :HIST], fut_gt=occ[:, HIST:],
                # planner-intended future poses, relative to the current frame (B,H,3)
                fut_rel=rel_pose(cur[:, None], pose[:, HIST:]),
                # history poses relative to the current frame (odometry)
                hist_rel=rel_pose(cur[:, None], pose[:, :HIST]))


def corrupt_history(hist, gen=None, flip=0.03, drop=0.0):
    """Sensor-ish corruption of the observed history: random voxel flips + whole-frame dropout."""
    B, T = hist.shape[:2]
    flips = (torch.rand(hist.shape, generator=gen) < flip).float()
    x = (hist - flips).abs()
    if drop > 0:
        keep = (torch.rand(B, T, 1, 1, generator=gen) >= drop).float()
        keep[:, -1] = 1.0                                              # newest frame is always available
        x = x * keep
    return x
