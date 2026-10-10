"""Synthetic cooperative-perception BEV scenes (this repo's own simulator, NOT a paper dataset).

World: a HxW BEV grid (1 cell ~ 1 m). 4-7 vehicles move at constant velocity.
* Ego sees vehicles only OUTSIDE a random occluder rectangle  -> cooperation is required to find hidden ones.
* A collaborator (other car / roadside unit) sees every vehicle, but its message is degraded the way the
  Sparse2comm abstract lists: bandwidth (random foreground mask-sampling), packet loss (cells dropped),
  transmission delay (message is d frames old), spatial misalignment (rigid sub-cell translation).
Features are C-channel gaussian blobs carrying a per-class semantic signature (stand-in for BEV features).
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class DataCfg:
    H: int = 32
    W: int = 32
    C: int = 8
    min_obj: int = 4
    max_obj: int = 7
    sigma: float = 1.2        # blob radius (cells)
    fg_thr: float = 0.4       # feature-norm threshold that defines "foreground" cells
    max_delay: int = 3        # frames of transmission delay
    max_offset: float = 1.5   # max rigid misalignment (cells) per axis
    max_loss: float = 0.4     # max per-cell packet-loss probability
    noise: float = 0.05
    flow_sigma: float = 1.5
    occ_min: int = 10
    occ_max: int = 16


def class_signatures(C, n_cls=3, seed=1234):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(n_cls, C, generator=g)
    return 1.5 * s / s.norm(dim=1, keepdim=True)


def _grid(H, W, device=None):
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                            torch.arange(W, dtype=torch.float32, device=device), indexing="ij")
    return xs, ys


def render(pos, sig, valid, H, W, sigma):
    """pos [B,N,2] (x,y); sig [B,N,C]; valid [B,N] -> features [B,C,H,W]."""
    xs, ys = _grid(H, W, pos.device)
    d2 = (xs[None, None] - pos[..., 0, None, None]) ** 2 + (ys[None, None] - pos[..., 1, None, None]) ** 2
    g = torch.exp(-d2 / (2 * sigma ** 2)) * valid[..., None, None]
    return torch.einsum("bnhw,bnc->bchw", g, sig)


def sample_scene(B, cfg: DataCfg, g: torch.Generator):
    N = cfg.max_obj
    n_obj = torch.randint(cfg.min_obj, cfg.max_obj + 1, (B,), generator=g)
    valid = (torch.arange(N)[None] < n_obj[:, None]).float()
    lo, hi = 5.0, cfg.W - 5.0
    pos = lo + (hi - lo) * torch.rand(B, N, 2, generator=g)
    ang = 2 * math.pi * torch.rand(B, N, generator=g)
    spd = 0.3 + 0.8 * torch.rand(B, N, generator=g)
    vel = torch.stack([spd * torch.cos(ang), spd * torch.sin(ang)], -1)
    cls = torch.randint(0, 3, (B, N), generator=g)
    sigs = class_signatures(cfg.C)[cls] + 0.2 * torch.randn(B, N, cfg.C, generator=g)
    ow = torch.randint(cfg.occ_min, cfg.occ_max + 1, (B,), generator=g).float()
    oh = torch.randint(cfg.occ_min, cfg.occ_max + 1, (B,), generator=g).float()
    ox = torch.rand(B, generator=g) * (cfg.W - ow)
    oy = torch.rand(B, generator=g) * (cfg.H - oh)
    return dict(pos=pos, vel=vel, sig=sigs, valid=valid, occ=torch.stack([ox, oy, ow, oh], 1))


def _inside(pos, occ):
    return ((pos[..., 0] >= occ[:, None, 0]) & (pos[..., 0] <= occ[:, None, 0] + occ[:, None, 2]) &
            (pos[..., 1] >= occ[:, None, 1]) & (pos[..., 1] <= occ[:, None, 1] + occ[:, None, 3])).float()


def make_batch(scene, cfg: DataCfg, g: torch.Generator, delay=None, keep_ratio=None, loss_p=None, offset=None):
    """Turn a scene (positions at the *current* time t) into one cooperative-perception sample.
    delay: int / [B] tensor / None(random). keep_ratio, loss_p: float / [B] / None(random). offset: [B,2]/None."""
    B = scene["pos"].shape[0]
    pos, vel, sig, valid, occ = scene["pos"], scene["vel"], scene["sig"], scene["valid"], scene["occ"]

    def _vec(v, lo, hi, integer=False):
        if v is None:
            r = torch.rand(B, generator=g)
            return torch.randint(lo, hi + 1, (B,), generator=g).float() if integer else lo + (hi - lo) * r
        return torch.as_tensor(v, dtype=torch.float32).expand(B).clone()

    d = _vec(delay, 0, cfg.max_delay, integer=True)
    keep = _vec(keep_ratio, 0.15, 0.6)
    lossp = _vec(loss_p, 0.0, cfg.max_loss)
    off = (torch.rand(B, 2, generator=g) * 2 - 1) * cfg.max_offset if offset is None else torch.as_tensor(offset, dtype=torch.float32).expand(B, 2).clone()

    H, W, C, s = cfg.H, cfg.W, cfg.C, cfg.sigma
    ego_vis = valid * (1 - _inside(pos, occ))
    ego = render(pos, sig, ego_vis, H, W, s) + cfg.noise * torch.randn(B, C, H, W, generator=g)

    full, sparse, mask = [], [], []
    for lag in (1, 0):  # frame index 0 = older (t-d-1), 1 = newer (t-d)
        p = pos - vel * (d + lag)[:, None, None] + off[:, None, :]
        f = render(p, sig, valid, H, W, s) + cfg.noise * torch.randn(B, C, H, W, generator=g)
        fg = (f.norm(dim=1, keepdim=True) > cfg.fg_thr).float()
        sel = (torch.rand(B, 1, H, W, generator=g) < keep[:, None, None, None]).float()  # random mask-sampling of foreground
        alive = (torch.rand(B, 1, H, W, generator=g) >= lossp[:, None, None, None]).float()  # packet loss
        m = fg * sel * alive
        full.append(f); sparse.append(f * m); mask.append(m)
    full, sparse, mask = torch.stack(full, 1), torch.stack(sparse, 1), torch.stack(mask, 1)

    # ---- targets at time t, in the ego frame (all valid vehicles, incl. ones ego cannot see)
    xs, ys = _grid(H, W)
    d2 = (xs[None, None] - pos[..., 0, None, None]) ** 2 + (ys[None, None] - pos[..., 1, None, None]) ** 2
    gk = torch.exp(-d2 / (2 * s ** 2)) * valid[..., None, None]
    heat = gk.max(1, keepdim=True)[0]
    ix = pos[..., 0].round().long().clamp(0, W - 1)
    iy = pos[..., 1].round().long().clamp(0, H - 1)
    bi = torch.arange(B)[:, None].expand_as(ix)
    vm = valid > 0
    heat[bi[vm], 0, iy[vm], ix[vm]] = 1.0  # exact discretised centre forced to 1 (CenterNet convention)
    wf = torch.exp(-d2 / (2 * cfg.flow_sigma ** 2)) * valid[..., None, None]
    sw = wf.sum(1, keepdim=True)
    disp = vel * d[:, None, None]
    flow_t = torch.einsum("bnhw,bnc->bchw", wf, disp) / (sw + 1e-6)
    return dict(ego=ego, sparse=sparse, mask=mask, full=full, delay=d, offset=off, keep=keep, lossp=lossp,
                heat=heat, flow_t=flow_t, flow_m=(sw > 0.3).float(), pos=pos, valid=valid, ego_vis=ego_vis,
                bandwidth=mask[:, 1].mean((1, 2, 3)))


def advance(scene, cfg: DataCfg, steps=1):
    """Move vehicles forward (bouncing off the arena edge) - used by simulate.py."""
    pos, vel = scene["pos"].clone(), scene["vel"].clone()
    for _ in range(steps):
        pos = pos + vel
        for ax, lim in ((0, cfg.W - 3.0), (1, cfg.H - 3.0)):
            hit = (pos[..., ax] < 3.0) | (pos[..., ax] > lim)
            vel[..., ax] = torch.where(hit, -vel[..., ax], vel[..., ax])
            pos[..., ax] = pos[..., ax].clamp(3.0, lim)
    out = dict(scene); out["pos"], out["vel"] = pos, vel
    return out
