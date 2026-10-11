"""Frozen-after-training mode-conditioned diffusion trajectory model (EDM-style denoiser, own stand-in backbone).
Diffusion runs on the residual r = future - constant-velocity extrapolation. D(x, sigma) predicts clean r."""
import math
import torch
import torch.nn as nn
from .data import D, COND_DIM, N_MODES


class Denoiser(nn.Module):
    def __init__(self, hidden=256, depth=4, sigma_data=0.5):
        super().__init__()
        self.sd = sigma_data
        layers, i = [], D + COND_DIM + N_MODES + 9
        for _ in range(depth):
            layers += [nn.Linear(i, hidden), nn.SiLU()]
            i = hidden
        layers.append(nn.Linear(hidden, D))
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigma, cond, mode):
        """x:(B,D) noisy residual; sigma:(B,) or float; cond:(B,COND_DIM); mode:(B,N_MODES) -> clean estimate (B,D)."""
        if not torch.is_tensor(sigma):
            sigma = torch.full((x.shape[0],), float(sigma), device=x.device)
        s = sigma.reshape(-1, 1).expand(x.shape[0], 1)
        c_skip = self.sd ** 2 / (s ** 2 + self.sd ** 2)
        c_out = s * self.sd / torch.sqrt(s ** 2 + self.sd ** 2)
        c_in = 1 / torch.sqrt(s ** 2 + self.sd ** 2)
        cn = torch.log(s) / 4
        ks = torch.arange(1, 5, device=x.device).float().reshape(1, -1)
        emb = torch.cat([cn, torch.sin(ks * cn), torch.cos(ks * cn)], 1)       # (B,9)
        f = self.net(torch.cat([c_in * x, cond, mode, emb], 1))                 # (B,D)
        return c_skip * x + c_out * f


def edm_loss(model, r, cond, mode, p_mean=-1.5, p_std=1.5):
    sigma = torch.exp(p_mean + p_std * torch.randn(r.shape[0], device=r.device))
    n = torch.randn_like(r) * sigma[:, None]
    den = model(r + n, sigma, cond, mode)
    w = (sigma ** 2 + model.sd ** 2) / (sigma * model.sd) ** 2
    return (w[:, None] * (den - r) ** 2).mean()


@torch.no_grad()
def sample(model, cond, mode, z=None, steps=16, sigma_max=2.0, sigma_min=0.02):
    """Heun ODE sampler. z=None/zeros -> deterministic 'nominal' (near-modal) trajectory."""
    B = cond.shape[0]
    sig = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), steps))
    sig = torch.cat([sig, torch.zeros(1)])
    x = (torch.zeros(B, D) if z is None else z) * sig[0]
    for i in range(steps):
        d = (x - model(x, sig[i], cond, mode)) / sig[i]
        xn = x + (sig[i + 1] - sig[i]) * d
        if sig[i + 1] > 0:
            d2 = (xn - model(xn, sig[i + 1], cond, mode)) / sig[i + 1]
            xn = x + (sig[i + 1] - sig[i]) * 0.5 * (d + d2)
        x = xn
    return x
