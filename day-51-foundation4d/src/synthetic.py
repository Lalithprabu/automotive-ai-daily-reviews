"""Synthetic 4-camera driving scenes with analytic ground truth (depth, BEV centers, 3D occupancy).

World frame = ego frame at each timestep. X right, Y forward, Z up. Cameras at ego origin, height 1.5 m,
yaw 0/90/180/270 deg. Objects are axis-aligned boxes (height 1.6 m). Rendering is exact ray casting
(ground plane + slab ray/box intersection), so every label is analytically correct.
"""
import math
import torch

CAM_H, BOX_H, MAX_D = 1.5, 1.6, 30.0
BEV_N, BEV_R, N_Z, Z_STEP = 16, 16.0, 4, 0.6   # 16x16 cells of 2 m over [-16,16]^2, 4 z-layers
PALETTE = torch.tensor([[0.9, 0.2, 0.2], [0.2, 0.8, 0.3], [0.95, 0.8, 0.1], [0.8, 0.3, 0.9], [0.1, 0.8, 0.9]])
K_MAX = 5


def make_scene(g: torch.Generator, n_scenes: int):
    """Random scenes. pos0:[S,K,2] rel. to ego at t=0; vel:[S,K,2] rel. velocity (m/frame); half:[S,K,2]; mask:[S,K]."""
    S, K = n_scenes, K_MAX
    ang = torch.rand(S, K, generator=g) * 2 * math.pi
    rad = 5.0 + torch.rand(S, K, generator=g) * 8.0
    pos = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], -1)
    vel = (torch.rand(S, K, 2, generator=g) - 0.5) * 1.0 - torch.tensor([0.0, 1.0]) * 0.0
    swap = torch.rand(S, K, generator=g) > 0.5
    half = torch.where(swap[..., None], torch.tensor([1.0, 2.2]), torch.tensor([2.2, 1.0]))
    n = torch.randint(2, K + 1, (S,), generator=g)
    mask = torch.arange(K)[None] < n[:, None]
    return dict(pos0=pos, vel=vel, half=half, mask=mask)


def cam_rays(H, W, V):
    """Unit-forward-component world rays [V,H*W,3]. Horizontal FOV 90 deg (f = W/2)."""
    f = W / 2.0
    u = (torch.arange(W) + 0.5 - W / 2) / f
    v = (torch.arange(H) + 0.5 - H / 2) / f
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    rays = []
    for k in range(V):
        th = k * math.pi / 2
        fwd = torch.tensor([math.sin(th), math.cos(th), 0.0])
        right = torch.tensor([math.cos(th), -math.sin(th), 0.0])
        up = torch.tensor([0.0, 0.0, 1.0])
        d = fwd[None, None] + uu[..., None] * right + (-vv[..., None]) * up
        rays.append(d.reshape(-1, 3))
    return torch.stack(rays)


def render_frame(scene, t, H, W, V=4, noise=0.02, g=None):
    """Return rgb [S,V,3,H,W], depth [S,V,H,W] (metres, clipped MAX_D), obj-pos [S,K,2] at frame t."""
    S, K = scene["mask"].shape
    pos = scene["pos0"] + scene["vel"] * t                      # [S,K,2] relative to ego
    rays = cam_rays(H, W, V)                                    # [V,P,3]
    P = H * W
    o = torch.tensor([0.0, 0.0, CAM_H])
    d = rays[None, :, :, None, :]                               # [1,V,P,1,3]
    lo = torch.cat([pos - scene["half"], torch.zeros(S, K, 1)], -1)
    hi = torch.cat([pos + scene["half"], torch.full((S, K, 1), BOX_H)], -1)
    lo, hi = lo[:, None, None], hi[:, None, None]               # [S,1,1,K,3]
    dd = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
    t1, t2 = (lo - o) / dd, (hi - o) / dd
    tmin = torch.minimum(t1, t2).amax(-1)
    tmax = torch.maximum(t1, t2).amin(-1)
    hit = (tmax >= tmin.clamp(min=0)) & scene["mask"][:, None, None]
    tbox = torch.where(hit, tmin.clamp(min=0), torch.full_like(tmin, 1e9))      # [S,V,P,K]
    tb, idx = tbox.min(-1)                                                      # nearest box
    dz = rays[None, :, :, 2]
    tg = torch.where(dz < -1e-6, CAM_H / (-dz).clamp(min=1e-6), torch.full_like(dz, 1e9)).expand(S, -1, -1)
    depth = torch.minimum(tb, tg)
    is_box, is_ground = tb < tg, (tg < tb) & (tg < MAX_D)
    depth_c = depth.clamp(max=MAX_D)
    hitpt = o + rays[None] * depth_c[..., None]
    chk = ((hitpt[..., 0].floor() + hitpt[..., 1].floor()) % 2)
    ground = (0.25 + 0.15 * chk)[..., None].expand(-1, -1, -1, 3)
    boxc = PALETTE[idx % 5] * (1.0 / (1.0 + 0.03 * depth_c))[..., None]
    sky = torch.tensor([0.55, 0.7, 0.9]).expand(S, V, P, 3)
    rgb = torch.where(is_box[..., None], boxc, torch.where(is_ground[..., None], ground, sky))
    if noise > 0:
        rgb = rgb + noise * torch.randn(rgb.shape, generator=g)
    rgb = rgb.clamp(0, 1).reshape(S, V, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()
    return rgb, depth_c.reshape(S, V, H, W), pos


def bev_targets(pos, half, mask):
    """Heatmap [S,1,N,N] (gaussian, exact centre cell forced to 1) and occupancy [S,Z,N,N] from box footprints."""
    S, K, _ = pos.shape
    cell = 2 * BEV_R / BEV_N
    ctr = (torch.arange(BEV_N) + 0.5) * cell - BEV_R                       # cell centres
    gy, gx = torch.meshgrid(ctr, ctr, indexing="ij")                       # rows=Y, cols=X
    heat = torch.zeros(S, BEV_N, BEV_N)
    occ_xy = torch.zeros(S, BEV_N, BEV_N)
    for k in range(K):
        m = mask[:, k]
        cx, cy = pos[:, k, 0], pos[:, k, 1]
        d2 = ((gx[None] - cx[:, None, None]) ** 2 + (gy[None] - cy[:, None, None]) ** 2) / cell ** 2
        h = torch.exp(-d2 / (2 * 1.0 ** 2)) * m[:, None, None]
        heat = torch.maximum(heat, h)
        ix = ((cx + BEV_R) / cell).floor().long().clamp(0, BEV_N - 1)
        iy = ((cy + BEV_R) / cell).floor().long().clamp(0, BEV_N - 1)
        inb = (cx.abs() < BEV_R) & (cy.abs() < BEV_R) & m
        heat[inb, iy[inb], ix[inb]] = 1.0                                   # CenterNet convention (Day 21 lesson)
        hx, hy = scene_half(half, k)
        ox = ((gx[None] - cx[:, None, None]).abs() < hx[:, None, None] + cell / 2 - 1e-3)
        oy = ((gy[None] - cy[:, None, None]).abs() < hy[:, None, None] + cell / 2 - 1e-3)
        occ_xy = torch.maximum(occ_xy, (ox & oy & m[:, None, None]).float())
    # centres out of range are dropped from the heatmap target (range limit of the BEV grid)
    zl = (torch.arange(N_Z) * Z_STEP < BOX_H).float()
    occ = occ_xy[:, None] * zl[None, :, None, None]
    return heat[:, None], occ


def scene_half(half, k):
    return half[:, k, 0], half[:, k, 1]


def build_dataset(n, seed, H=24, W=32, V=4, T=2):
    """-> dict of tensors. Frames t=0..T-1; labels at the last frame (t=T-1)."""
    g = torch.Generator().manual_seed(seed)
    sc = make_scene(g, n)
    rgbs, deps = [], []
    for t in range(T):
        r, d, pos = render_frame(sc, float(t), H, W, V, g=g)
        rgbs.append(r), deps.append(d)
    heat, occ = bev_targets(pos, sc["half"], sc["mask"])
    return dict(rgb=torch.stack(rgbs, 2), depth=torch.stack(deps, 2),   # [S,V,T,3,H,W], [S,V,T,H,W]
                heat=heat, occ=occ, pos=pos, mask=sc["mask"], scene=sc)
