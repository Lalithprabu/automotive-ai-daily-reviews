"""Synthetic ego-centric streaming occupancy world with a FAST and a SLOW stand-in predictor.

NOT nuScenes.  Classes: 0 free, 1 static (buildings/parked), 2 dynamic (moving vehicles).
The ego moves (speed, yaw drift) and frames arrive at irregular intervals dt in {0.25,0.5,0.75}s.
FAST predictor stand-in: 3x3-cell block average (loses small objects) + large noise.
SLOW predictor stand-in: sharp, low noise -- but only available for frame 0 (it is 'delayed').
"""
import math
import torch
import torch.nn.functional as F

H = W = 48
RES = 1.0                      # metres per cell
EXTENT = H * RES
K = 3
DT_CHOICES = torch.tensor([0.25, 0.5, 0.75])


def _rects(b, n, dyn, g):
    """Random rectangles [b,n,(cx,cy,yaw,half_l,half_w,vx,vy)] in the world frame."""
    cx = torch.rand(b, n, generator=g) * (100 if not dyn else 80) - 20
    side = (torch.rand(b, n, generator=g) < .5).float() * 2 - 1
    if dyn:
        cy = (torch.rand(b, n, generator=g) * 14 - 7)
        yaw = torch.zeros(b, n); hl = torch.full((b, n), 2.2); hw = torch.full((b, n), 1.0)
        vx = torch.rand(b, n, generator=g) * 9 - 2; vy = (torch.rand(b, n, generator=g) - .5) * 1.0
    else:
        cy = side * (6 + torch.rand(b, n, generator=g) * 14)
        yaw = (torch.rand(b, n, generator=g) - .5) * .4
        hl = 1.5 + torch.rand(b, n, generator=g) * 4; hw = 1.5 + torch.rand(b, n, generator=g) * 3
        vx = vy = torch.zeros(b, n)
    return torch.stack([cx, cy, yaw, hl, hw, vx, vy], -1)


def _cell_coords():
    j = (torch.arange(W) + .5 - W / 2) * RES; i = (torch.arange(H) + .5 - H / 2) * RES
    yy, xx = torch.meshgrid(i, j, indexing="ij")           # x along W, y along H (ego +x = forward)
    return xx, yy


def _render(pose, stat, dyn_pos, dyn):
    """pose [B,3]=(X,Y,th); stat [B,Ns,7]; dyn_pos [B,Nd,2] current centres. -> labels [B,H,W] long."""
    B = pose.shape[0]
    xx, yy = _cell_coords()
    c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
    wx = pose[:, 0, None, None] + c[:, None, None] * xx - s[:, None, None] * yy   # [B,H,W]
    wy = pose[:, 1, None, None] + s[:, None, None] * xx + c[:, None, None] * yy
    lab = torch.zeros(B, H, W, dtype=torch.long)

    def inside(cx, cy, yaw, hl, hw):                       # all [B,N]
        dx = wx[:, None] - cx[:, :, None, None]; dy = wy[:, None] - cy[:, :, None, None]
        cc, ss = torch.cos(yaw)[:, :, None, None], torch.sin(yaw)[:, :, None, None]
        lx = cc * dx + ss * dy; ly = -ss * dx + cc * dy
        return ((lx.abs() <= hl[:, :, None, None]) & (ly.abs() <= hw[:, :, None, None])).any(1)
    lab[inside(stat[..., 0], stat[..., 1], stat[..., 2], stat[..., 3], stat[..., 4])] = 1
    lab[inside(dyn_pos[..., 0], dyn_pos[..., 1], dyn[..., 2], dyn[..., 3], dyn[..., 4])] = 2
    return lab


def fast_logits(lab, g):
    oh = F.one_hot(lab, K).permute(0, 3, 1, 2).float()                          # [B,K,H,W]
    blur = F.avg_pool2d(oh, 3, 1, 1)                                             # loses thin/small objects
    return 3.0 * blur + 1.0 * torch.randn(oh.shape, generator=g)


def slow_logits(lab, g):
    oh = F.one_hot(lab, K).permute(0, 3, 1, 2).float()
    return 4.0 * oh + 0.4 * torch.randn(oh.shape, generator=g)


@torch.no_grad()
def make_batch(b, T=6, seed=None):
    """Returns dict: slow0 [B,K,H,W], fast [B,T+1,K,H,W], ego [B,T,3], dts [B,T], gt [B,T+1,H,W]."""
    g = torch.Generator(); g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()) if seed is None else seed)
    stat, dyn = _rects(b, 14, False, g), _rects(b, 8, True, g)
    dts = DT_CHOICES[torch.randint(0, 3, (b, T), generator=g)]
    speed = 4 + torch.rand(b, generator=g) * 8
    yawrate = (torch.rand(b, generator=g) - .5) * 0.2                            # rad/s
    pose = torch.zeros(b, 3); poses = [pose.clone()]
    dyn_pos = dyn[..., :2].clone(); dpos = [dyn_pos.clone()]
    for k in range(T):
        dt = dts[:, k]; th = pose[:, 2] + yawrate * dt
        pose = torch.stack([pose[:, 0] + speed * dt * torch.cos(th), pose[:, 1] + speed * dt * torch.sin(th), th], 1)
        dyn_pos = dyn_pos + dyn[..., 5:7] * dt[:, None, None]
        poses.append(pose.clone()); dpos.append(dyn_pos.clone())
    gt, ego = [], []
    for k in range(T + 1):
        gt.append(_render(poses[k], stat, dpos[k], dyn))
        if k > 0:                                                                # pose k expressed in frame k-1
            p0, p1 = poses[k - 1], poses[k]
            d = p1[:, :2] - p0[:, :2]; c, s = torch.cos(p0[:, 2]), torch.sin(p0[:, 2])
            ego.append(torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], p1[:, 2] - p0[:, 2]], 1))
    gt = torch.stack(gt, 1)
    fast = torch.stack([fast_logits(gt[:, k], g) for k in range(T + 1)], 1)
    return dict(slow0=slow_logits(gt[:, 0], g), fast=fast, ego=torch.stack(ego, 1), dts=dts, gt=gt)


def miou(pred, gt, classes=(1, 2)):
    """Mean IoU over the given classes. pred/gt: [..., H, W] long. Returns scalar tensor."""
    ious = []
    for c in classes:
        p, t = pred == c, gt == c
        inter, union = (p & t).sum().float(), (p | t).sum().float()
        ious.append(inter / union.clamp(min=1))
    return torch.stack(ious).mean()


def class_iou(pred, gt, c):
    p, t = pred == c, gt == c
    return ((p & t).sum().float() / (p | t).sum().float().clamp(min=1)).item()
