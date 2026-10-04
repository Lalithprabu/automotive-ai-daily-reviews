"""Synthetic asynchronous-collaborative-perception data (this repo's own design, NOT the paper's data).

World: HxW BEV grid (cells). A few objects move with constant speed and a small constant turn rate.
 * Ego observes the world at time t but only inside a random disk (the rest is its blind zone).
 * Collaborator sees EVERYTHING, but its message is delayed: it describes the world at t - delta.
 * Both send/hold BEV "feature" maps painted from object states:
     ch0 occupancy blob (sigma 1.5), ch1/ch2 velocity (wide blob, sigma 4), ch3 wide presence blob.
 * The receiver only knows a NOISY delay estimate (clock jitter), supplied as a constant plane.
Target: object-center heatmap at the EGO's time t, for ALL objects (visible or blind).
"""
import math
import torch

H = W = 40
DMAX = 6          # max delay in steps
SIG_OCC, SIG_VEL = 1.5, 4.0


def _grid(device="cpu"):
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                            torch.arange(W, dtype=torch.float32, device=device), indexing="ij")
    return xs, ys


def simulate_objects(B, N=6, T=DMAX + 1, gen=None, speed=(0.2, 0.8), start=(11.0, 29.0)):
    """Returns pos (B,N,T,2) [x,y] in cells, vel (B,N,T,2) per-step velocity, valid (B,N)."""
    g = dict(generator=gen)
    n_obj = torch.randint(2, N + 1, (B,), **g)
    valid = torch.arange(N)[None, :] < n_obj[:, None]
    p0 = start[0] + (start[1] - start[0]) * torch.rand(B, N, 2, **g)
    sp = speed[0] + (speed[1] - speed[0]) * torch.rand(B, N, **g)
    th = 2 * math.pi * torch.rand(B, N, **g)
    om = (torch.rand(B, N, **g) - 0.5) * 0.16            # turn rate rad/step
    pos, vel = [], []
    p = p0
    for _ in range(T):
        v = torch.stack([sp * torch.cos(th), sp * torch.sin(th)], -1)
        pos.append(p); vel.append(v)
        p = p + v
        th = th + om
    return torch.stack(pos, 2), torch.stack(vel, 2), valid


def _blob(px, py, sigma, valid):
    xs, ys = _grid(px.device)
    d2 = (xs[None, None] - px[..., None, None]) ** 2 + (ys[None, None] - py[..., None, None]) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2)) * valid[..., None, None].float()  # (B,N,H,W)


def render_obs(pos, vel, valid, noise=0.05, gen=None):
    """pos/vel: (B,N,2) at one instant -> features (B,4,H,W)."""
    px, py = pos[..., 0], pos[..., 1]
    occ = _blob(px, py, SIG_OCC, valid)
    wide = _blob(px, py, SIG_VEL, valid)
    f0 = occ.amax(1)
    f1 = (wide * vel[..., 0, None, None]).sum(1) / 0.8
    f2 = (wide * vel[..., 1, None, None]).sum(1) / 0.8
    f3 = wide.amax(1)
    f = torch.stack([f0, f1, f2, f3], 1)
    return f + noise * torch.randn(f.shape, generator=gen)


def heatmap(pos, valid):
    """CenterNet-style gaussian heatmap, exact discretised centre forced to 1.0."""
    hm = _blob(pos[..., 0], pos[..., 1], SIG_OCC, valid).amax(1)
    B, N, _ = pos.shape
    ci = pos.round().long().clamp(0, W - 1)
    for n in range(N):
        m = valid[:, n]
        b = torch.nonzero(m).squeeze(1)
        hm[b, ci[b, n, 1], ci[b, n, 0]] = 1.0
    return hm[:, None]


def make_batch(B, gen=None, delta=None, disk_r=14.0, delay_noise=0.7, speed=(0.2, 0.8)):
    pos, vel, valid = simulate_objects(B, gen=gen, speed=speed)
    if delta is None:
        delta = torch.randint(0, DMAX + 1, (B,), generator=gen)
    elif not torch.is_tensor(delta):
        delta = torch.full((B,), int(delta))
    t = DMAX
    ar = torch.arange(B)
    pe, ve = pos[:, :, t], vel[:, :, t]                        # ego-time truth
    pc, vc = pos[ar, :, t - delta], vel[ar, :, t - delta]      # collaborator (delayed) truth
    # ego visibility disk
    cen = 12 + 16 * torch.rand(B, 2, generator=gen)
    xs, ys = _grid()
    disk = (((xs[None] - cen[:, 0, None, None]) ** 2 + (ys[None] - cen[:, 1, None, None]) ** 2) <= disk_r ** 2).float()
    vis_obj = (((pe - cen[:, None]) ** 2).sum(-1) <= disk_r ** 2) & valid
    ego = render_obs(pe, ve, vis_obj, gen=gen) * disk[:, None]
    ego = torch.cat([ego, disk[:, None]], 1)                   # 5 ch (last = visibility mask)
    col = render_obs(pc, vc, valid, gen=gen)
    d_est = (delta.float() + delay_noise * torch.randn(B, generator=gen)).clamp(0, DMAX + 1)
    col = torch.cat([col, (d_est / DMAX)[:, None, None, None].expand(B, 1, H, W)], 1)  # 5 ch
    return dict(ego=ego, col=col, hm=heatmap(pe, valid), pos=pe, valid=valid, vis=vis_obj,
                delta=delta, pos_col=pc, vel=ve, disk=disk, cen=cen)
