"""Conditional flow-matching trajectory generator + hazard-conditioned energy guidance (the paper's core idea).

Flow matching:  x_t = (1-t) x_0 + t x_1,  x_0 ~ N(0, I) (noise), x_1 = expert trajectory.
Network predicts the velocity  v_theta(x_t, t, c) ~ x_1 - x_0.
Sampling (Euler):  x_{t+dt} = x_t + dt * v.
Guidance (training-free, weights frozen): v' = v - lambda * grad_{x_t} E(x1_hat),  x1_hat = x_t + (1-t) v
where E is a hazard energy computed from EXACT obstacle geometry (signed distance of every waypoint to the
ego-inflated obstacle box).  This is this repo's reconstruction of the idea described in arXiv:2610.28184.
"""
import math
import torch, torch.nn as nn
import torch.nn.functional as F
LATERAL_DAMP = 0.05
GUIDE_CAP = 2.0
from src.scene import decode, box_sdf, inflated_box, H

class TimeEmbed(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.dim = dim
    def forward(self, t):                                   # t: (B,) -> (B, dim)
        half = self.dim // 2
        f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * f[None] * 100.0
        return torch.cat([a.sin(), a.cos()], 1)

class FlowNet(nn.Module):
    def __init__(self, horizon=H, ctx_dim=7, hidden=192, depth=4):
        super().__init__()
        self.D = horizon * 2
        self.temb = TimeEmbed(64)
        self.inp = nn.Linear(self.D + ctx_dim + 64, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                                   nn.Linear(hidden, hidden)) for _ in range(depth)])
        self.out = nn.Linear(hidden, self.D)
    def forward(self, x, t, c):                             # x (B,D), t (B,), c (B,ctx) -> v (B,D)
        h = self.inp(torch.cat([x, c, self.temb(t)], 1))
        for b in self.blocks: h = h + b(h)                  # residual MLP blocks
        return self.out(h)

def flow_matching_loss(net, x1, c):
    x0 = torch.randn_like(x1); t = torch.rand(x1.shape[0], device=x1.device)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    return F.mse_loss(net(xt, t, c), x1 - x0)

def hazard_energy(traj_n, d, present, margin=0.8):
    """E(tau) = sum_k relu(margin - sdf_k)^2  over waypoints k; sdf from exact inflated obstacle box. (B,) energies.
    Zero when no hazard present."""
    p = decode(traj_n)                                      # (B,H,2) metres
    box = inflated_box(d[:, None])
    sdf = box_sdf(p[..., 0], p[..., 1], box)                # (B,H)
    e = (F.relu(margin - sdf) ** 2).sum(1)
    return e * present.float()

@torch.no_grad()
def _v(net, x, t, c): return net(x, t, c)

def sample(net, c, d=None, present=None, steps=16, guide_lambda=0.0, margin=0.8, generator=None, noise=None):
    """Euler sampler. guide_lambda=0 -> baseline ('old way'). >0 -> hazard-energy guidance ('new way').
    Net weights are never modified. Returns normalised trajectories (B, H*2)."""
    B = c.shape[0]
    x = noise if noise is not None else torch.randn(B, net.D, generator=generator)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt)
        if guide_lambda > 0 and present is not None and present.any():
            with torch.enable_grad():
                xr = x.detach().requires_grad_(True)
                v = net(xr, t, c)
                x1 = xr + (1 - t[:, None]) * v                       # one-step clean-sample estimate
                E = hazard_energy(x1, d, present, margin).sum()
                g, = torch.autograd.grad(E, xr)
            # guidance gets stronger as t -> 1 (x1_hat becomes reliable); gradient scaled to metre-space units
            # lane-keeping prior: damp the lateral component of the gradient (the car should brake, not swerve)
            g = g.view(B, -1, 2) * torch.tensor([1.0, LATERAL_DAMP]); g = g.view(B, -1)
            w = guide_lambda * (0.25 + t[:, None])
            v = v.detach()
            # trust-region on the guidance: never let it exceed GUIDE_CAP x the model's own velocity norm
            # (without this, large energies near the obstacle push samples off the data manifold)
            gn = (w * g).norm(dim=1, keepdim=True) + 1e-8
            scale = torch.clamp(GUIDE_CAP * v.norm(dim=1, keepdim=True) / gn, max=1.0)
            v = v - scale * w * g
        else:
            v = _v(net, x, t, c)
        x = x + dt * v
    return x.detach()
