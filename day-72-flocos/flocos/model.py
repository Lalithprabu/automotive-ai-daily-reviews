"""
FloCOS - Predicting Compact Forward Occupancy Sets of Surrounding Vehicles via Conditional Flow Matching
(arXiv 2611.15488).  Toy PyTorch re-implementation of the CORE idea:

    history + context ──► AgentEncoder ──► condition c (B, d)
    x_0 ~ N(0, I)  (B, M, 60) ──► Euler ODE   dx/dt = v_θ(x_t, t, c)  (NFE steps) ──► x_1 = θ_1..θ_M
                     θ_i  = Bernstein coefficients of one half-space convex polygon tube   (geometry.py)
                     O(t) = ∪_i P_i(t)    compact, NON-convex forward occupancy set, 3 s horizon

  Stage 1 (conditional flow matching, rectified-flow path):
        x_t = (1 - t) x_0 + t x_1,   x_1 = tight tube fitted to the realised future,
        L_FM = || v_θ(x_t, t, c) - (x_1 - x_0) ||²
  Stage 2 (set-level losses through the differentiable sampler, as in the abstract):
        L_cov  = E_{p ∈ true footprint} [ relu( softmin_i viol_i(p) + δ ) ]     penalises MISSED occupancy
        L_area = E_t [ area( ∪_i P_i(t) ) ]  (soft Monte-Carlo union)          penalises EXCESS area
        L = L_FM + λ_cov L_cov + λ_area L_area

What follows the paper (abstract-level): union of half-space convex polygons, temporal-basis coefficients,
conditional flow matching over those coefficients, coverage + union-area losses, the union feeding
collision-avoidance constraints of a planner.  Everything else (M, N_H, Bernstein degree, encoder,
velocity network, NFE, loss weights, stage split) is MY choice: the full text was not readable.

Shape legend: B batch, M set components, P history steps (10), H future steps (12), K = 60 coefficients.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as G
from . import world as W

HIST_SCALE = torch.tensor([20.0, 2.0, 25.0, 2.0])


# =========================================================================== condition encoder
class AgentEncoder(nn.Module):
    """(B, P, 4) history [x, y, vx, vy] + (B, 6) context -> (B, d).  GRU over time + context MLP."""

    def __init__(self, d=128):
        super().__init__()
        self.inp = nn.Linear(4, 64)
        self.gru = nn.GRU(64, d, batch_first=True)
        self.ctx = nn.Sequential(nn.Linear(W.CTX_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, b: dict) -> torch.Tensor:
        h = F.gelu(self.inp(b["hist"] / HIST_SCALE))                        # (B, P, 64)
        _, hn = self.gru(h)                                                  # (1, B, d)
        return self.out(torch.cat([hn[0], self.ctx(b["ctx"])], -1))          # (B, d)


def time_embedding(t: torch.Tensor, dim=64) -> torch.Tensor:
    """Sinusoidal embedding of the flow time t in [0, 1].  (...,) -> (..., dim)."""
    f = torch.exp(torch.linspace(0, math.log(1000.0), dim // 2))
    a = t[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class FiLMBlock(nn.Module):
    """Residual MLP block whose LayerNorm output is modulated by (condition, time): h + MLP(γ ⊙ LN(h) + β)."""

    def __init__(self, w, d_cond):
        super().__init__()
        self.ln = nn.LayerNorm(w, elementwise_affine=False)
        self.film = nn.Linear(d_cond, 2 * w)
        self.mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Linear(2 * w, w))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h, cond):
        g, b = self.film(cond).chunk(2, -1)
        return h + self.mlp(self.ln(h) * (1 + g) + b)


class VelocityField(nn.Module):
    """v_θ(x_t, t, c): (..., K) coefficients -> (..., K) velocity.  Shared over the M set components."""

    def __init__(self, d_cond=128, w=256, blocks=4):
        super().__init__()
        self.inp = nn.Linear(G.COEF_DIM, w)
        self.cond = nn.Sequential(nn.Linear(d_cond + 64, w), nn.GELU(), nn.Linear(w, w))
        self.blocks = nn.ModuleList([FiLMBlock(w, w) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, G.COEF_DIM))

    def forward(self, x, t, c):
        cond = self.cond(torch.cat([c, time_embedding(t)], -1))              # (..., w)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(h)


# =========================================================================== FloCOS
class FloCOS(nn.Module):
    def __init__(self, d=128, w=256, blocks=4):
        super().__init__()
        self.enc = AgentEncoder(d)
        self.v = VelocityField(d, w, blocks)

    def fm_loss(self, b: dict) -> torch.Tensor:
        """Conditional flow matching on the coefficient tube of the realised future (rectified-flow path)."""
        x1 = G.encode_tube(b["fut"], b["head"])                              # (B, K)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0])
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1                         # straight-line probability path
        return F.mse_loss(self.v(xt, t, self.enc(b)), x1 - x0)

    def sample(self, b: dict, M=6, nfe=4, noise=None):
        """Euler integration of the learned ODE for M noise draws per agent.  -> θ (B, M, K).

        Differentiable when called under grad (stage 2 back-propagates the set losses through all NFE steps).
        """
        c = self.enc(b)                                                      # (B, d)
        B = c.shape[0]
        x = torch.randn(B, M, G.COEF_DIM) if noise is None else noise
        cM = c[:, None].expand(-1, M, -1)
        dt = 1.0 / nfe
        for k in range(nfe):
            t = torch.full((B, M), k * dt)
            x = x + dt * self.v(x, t, cM)                                    # x_{t+dt} = x_t + dt v_θ
        return x

    def predict_set(self, b: dict, M=6, nfe=4, noise=None):
        """-> centres (B, M, H, 2), supports (B, M, H, N_H): the union of M convex polygons per step."""
        return G.decode(self.sample(b, M, nfe, noise))


# =========================================================================== set-level losses
def set_losses(c, s, fut, head, delta=0.05, tau_min=0.1, soft_tau=0.15, area_pts=128):
    """Coverage loss (missed occupancy) + soft union-area loss (excess area).

    c (B, M, H, 2), s (B, M, H, N_H), fut (B, H, 2), head (B, H)
    """
    p = G.footprint_points(fut, head, 5, 3)                                  # (B, H, 15, 2) true footprint
    v = G.violation(p[:, None], c, s)                                        # (B, M, H, 15)
    v_union = -tau_min * torch.logsumexp(-v / tau_min, 1)                    # smooth min over components
    l_cov = F.relu(v_union + delta).mean()
    area = G.union_area(c, s, area_pts, soft_tau=soft_tau)                   # (B, H) m^2
    return l_cov, area.mean()


# =========================================================================== baselines
class DetReg(nn.Module):
    """Deterministic regression of ONE future trajectory -> one footprint per step (classic under-approximation)."""

    def __init__(self, d=128):
        super().__init__()
        self.enc = AgentEncoder(d)
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, W.H * 2))

    def forward(self, b):
        return (self.head(self.enc(b)).view(-1, W.H, 2) * torch.tensor([25.0, 2.5]))[:, None]   # (B, 1, H, 2)


class MTP(nn.Module):
    """K-mode trajectory predictor (winner-takes-all + mode classification) -> union of K footprints."""

    def __init__(self, K=6, d=128):
        super().__init__()
        self.K = K
        self.enc = AgentEncoder(d)
        self.head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, K * (W.H * 2 + 1)))

    def forward(self, b):
        y = self.head(self.enc(b)).view(-1, self.K, W.H * 2 + 1)
        return y[..., :-1].view(-1, self.K, W.H, 2) * torch.tensor([25.0, 2.5]), y[..., -1]


def traj_heading(traj: torch.Tensor) -> torch.Tensor:
    """(..., H, 2) positions -> (..., H) headings by finite differences from the origin."""
    p = torch.cat([torch.zeros_like(traj[..., :1, :]), traj], -2)
    d = p[..., 1:, :] - p[..., :-1, :]
    return torch.atan2(d[..., 1], d[..., 0].clamp(min=1e-2))


def footprint_set(traj: torch.Tensor):
    """Trajectories (B, M, H, 2) -> rectangle footprints as polygons: c (B, M, H, 2), s (B, M, H, N_H)."""
    return traj, G.rect_support(traj_heading(traj))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
