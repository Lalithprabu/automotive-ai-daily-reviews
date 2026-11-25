"""PyTorch core: an interaction-aware Transformer trajectory predictor, plus the two noise mitigations the paper compares.

Re-implementation of the setting of arXiv 2611.28128 (Salehi, Lusvarghi, Sepulcre, Gozalvez, "Robust Trajectory
Prediction Under Sensing and V2X Noise: Evaluation and Mitigation Strategies"), built from the abstract only.  What
the abstract states:

    (1) Transformer-based interaction-aware predictors are trained on curated data where noise was filtered offline
    (2) under realistic noisy position / heading / speed (worse for V2X-received states) accuracy degrades by up to 6x
    (3) two mitigations: adding INPUT DENOISING modules vs RETRAINING on a NOISE-AUGMENTED dataset
    (4) retraining on noise-augmented data is better and keeps degradation below 1.5x

The paper evaluates five published models; I build ONE generic interaction-aware Transformer (my construction), a
learned input denoiser, and a classical constant-velocity Kalman filter as a second denoiser.

Shapes (B scenes, N = 8 agents, T = 20 history steps, F = 8 features, D = 128, K = 6 modes, H = 30 future steps):

    hist   (B, N, T, 4) [x, y, heading, v] (possibly noisy) + fresh (B, N, T) + v2x flag (B, N)
    feats  (B, N, T, 8) = [x/50, y/5, cos h, sin h, v/30, fresh, staleness/5, is_v2x]
    temporal encoder   per agent: Linear(8 -> D) + learned time embedding -> 2 Transformer layers over T
                       -> agent token = [last step || mean over steps] -> Linear(2D -> D)        (B, N, D)
    social encoder     2 Transformer layers over the N agent tokens (every agent attends to every agent)  (B, N, D)
    decoder            K learned mode queries + agent token -> MLP -> offsets (B, N, K, H, 2) and logits (B, N, K)
    output             pred = anchor + cumsum(offsets),  anchor = last observed (x, y) of the INPUT history
                       -> the model has to learn on its own whether the latest observation can be trusted
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noise import staleness

T_H, T_F, K_MODES = 20, 30, 6


def featurize(hist: torch.Tensor, fresh: torch.Tensor, v2x: torch.Tensor) -> torch.Tensor:
    """(B, N, T, 4), (B, N, T), (B, N) -> (B, N, T, 8)."""
    x, y, h, v = hist.unbind(-1)
    st = staleness(fresh).clamp(max=10) / 5.0
    iv = v2x[..., None].float().expand_as(x)
    return torch.stack([x / 50.0, y / 5.0, torch.cos(h), torch.sin(h), v / 30.0, fresh, st, iv], -1)


class InteractionTransformer(nn.Module):
    """Multi-agent, multi-modal trajectory predictor (temporal Transformer -> social Transformer -> mode decoder)."""

    def __init__(self, d: int = 128, heads: int = 4, k: int = K_MODES, t_h: int = T_H, t_f: int = T_F):
        super().__init__()
        self.k, self.t_f = k, t_f
        self.inp = nn.Linear(8, d)
        self.tpos = nn.Parameter(torch.randn(1, t_h, d) * 0.02)
        lay = lambda: nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(lay(), 2, enable_nested_tensor=False)
        self.pool = nn.Linear(2 * d, d)
        self.social = nn.TransformerEncoder(lay(), 2, enable_nested_tensor=False)
        self.modes = nn.Parameter(torch.randn(1, 1, k, d) * 0.1)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, t_f * 2))
        self.cls = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, hist, fresh, v2x) -> dict:
        B, N, T, _ = hist.shape
        f = featurize(hist, fresh, v2x).reshape(B * N, T, -1)                     # (B N, T, 8)
        z = self.temporal(self.inp(f) + self.tpos)                                # (B N, T, D)
        a = self.pool(torch.cat([z[:, -1], z.mean(1)], -1)).view(B, N, -1)        # (B, N, D) agent tokens
        a = self.social(a)                                                        # (B, N, D) interaction
        q = a[:, :, None] + self.modes                                            # (B, N, K, D)
        off = self.dec(q).view(B, N, self.k, self.t_f, 2)                         # per-step displacements
        # displacement scale: ~30 m/s * 0.1 s = 3 m per step in x, small in y
        off = off * torch.tensor([3.0, 0.3], device=hist.device)
        anchor = hist[:, :, -1, :2]                                               # last observed (x, y)
        pred = anchor[:, :, None, None] + off.cumsum(3)                           # (B, N, K, H, 2)
        logit = self.cls(q)[..., 0]                                               # (B, N, K)
        return dict(pred=pred, logit=logit)


def wta_loss(out: dict, fut: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Winner-takes-all: Huber on the closest mode + cross-entropy that the closest mode gets the highest logit.
    fut (B, N, H, 2) clean ground-truth future."""
    err = (out["pred"] - fut[:, :, None]).norm(dim=-1).mean(-1)                   # (B, N, K) ADE per mode
    best = err.argmin(-1)                                                         # (B, N)
    pb = out["pred"].gather(2, best[..., None, None, None].expand(-1, -1, 1, *out["pred"].shape[3:]))[:, :, 0]
    reg = F.smooth_l1_loss(pb, fut)
    cls = F.cross_entropy(out["logit"].flatten(0, 1), best.flatten())
    return reg + 0.1 * cls, dict(reg=reg.item(), cls=cls.item(), minade=err.min(-1).values.mean().item())


class HistoryDenoiser(nn.Module):
    """Learned INPUT DENOISING module: noisy history -> estimate of the clean history, fed to a frozen predictor.

    Per agent, a 2-layer temporal Transformer reads the same 8 features and predicts a residual correction
    [dx, dy, d cos, d sin, dv] for every step; heading is rebuilt with atan2.  It is per-agent (no social layer),
    the usual shape of a plug-in pre-filter.  Trained with MSE to the clean history (last step weighted x4).
    """

    def __init__(self, d: int = 96, heads: int = 4, t_h: int = T_H):
        super().__init__()
        self.inp = nn.Linear(8, d)
        self.tpos = nn.Parameter(torch.randn(1, t_h, d) * 0.02)
        lay = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, 2, enable_nested_tensor=False)
        self.out = nn.Linear(d, 5)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hist, fresh, v2x):
        B, N, T, _ = hist.shape
        f = featurize(hist, fresh, v2x).reshape(B * N, T, -1)
        r = self.out(self.enc(self.inp(f) + self.tpos)).view(B, N, T, 5)
        x = hist[..., 0] + 2.0 * r[..., 0]
        y = hist[..., 1] + 1.0 * r[..., 1]
        c = torch.cos(hist[..., 2]) + 0.2 * r[..., 2]
        s = torch.sin(hist[..., 2]) + 0.2 * r[..., 3]
        v = (hist[..., 3] + 1.0 * r[..., 4]).clamp(min=0)
        clean = torch.stack([x, y, torch.atan2(s, c), v], -1)
        return clean, torch.ones_like(fresh)                                      # denoised: treat all as fresh


def denoise_loss(den: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    w = torch.ones(den.shape[2], device=den.device)
    w[-1] = 4.0
    dp = ((den[..., :2] - clean[..., :2]) ** 2).sum(-1)
    dh = 1 - torch.cos(den[..., 2] - clean[..., 2])
    dv = (den[..., 3] - clean[..., 3]) ** 2
    return ((dp + 20 * dh + 0.5 * dv) * w).mean()


@torch.no_grad()
def kalman_cv(hist: torch.Tensor, fresh: torch.Tensor, r_pos: float = 0.6, q_acc: float = 2.0, dt: float = 0.1):
    """Classical denoiser: causal constant-velocity Kalman filter per agent on (x, y) with the reported speed and
    heading used as a velocity measurement.  Skips the update when no fresh message arrived.  The noise parameters
    are fixed (tuned once for a medium noise level), as a deployed filter would be.  -> (filtered hist, ones)."""
    B, N, T, _ = hist.shape
    x0, y0, h, v = hist.unbind(-1)
    meas = torch.stack([x0, v * torch.cos(h), y0, v * torch.sin(h)], -1)          # (B, N, T, 4) [px, vx, py, vy]
    Fm = torch.tensor([[1, dt], [0, 1.0]])
    Q = q_acc ** 2 * torch.tensor([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
    R = torch.diag(torch.tensor([r_pos ** 2, 1.0 ** 2]))                           # (pos, vel) measurement noise
    out = torch.zeros_like(hist)
    for ax in range(2):
        m = meas[..., 2 * ax: 2 * ax + 2]                                           # (B, N, T, 2)
        xs = m[:, :, 0].clone()                                                     # state (B, N, 2)
        P = torch.eye(2).expand(B, N, 2, 2) * torch.tensor([r_pos ** 2, 1.0])
        res = [xs]
        for t in range(1, T):
            xs = xs @ Fm.T
            P = Fm @ P @ Fm.T + Q
            S = P + R
            Kg = P @ torch.linalg.inv(S)
            upd = fresh[:, :, t, None]
            xs = xs + upd * (Kg @ (m[:, :, t] - xs)[..., None])[..., 0]
            P = torch.where(upd[..., None] > 0, (torch.eye(2) - Kg) @ P, P)
            res.append(xs)
        res = torch.stack(res, 2)                                                  # (B, N, T, 2)
        out[..., ax] = res[..., 0]
        if ax == 0:
            vx = res[..., 1]
        else:
            vy = res[..., 1]
    out[..., 2] = torch.atan2(vy, vx.clamp(min=0.1))
    out[..., 3] = torch.sqrt(vx ** 2 + vy ** 2)
    return out, torch.ones_like(fresh)


def constant_velocity(hist: torch.Tensor, t_f: int = T_F, dt: float = 0.1) -> torch.Tensor:
    """Kinematic baseline: last (x, y) + reported speed along reported heading.  -> (B, N, 1, H, 2)."""
    x, y, h, v = hist[:, :, -1].unbind(-1)
    t = torch.arange(1, t_f + 1, dtype=hist.dtype) * dt
    px = x[..., None] + (v * torch.cos(h))[..., None] * t
    py = y[..., None] + (v * torch.sin(h))[..., None] * t
    return torch.stack([px, py], -1)[:, :, None]


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m, dn = InteractionTransformer(), HistoryDenoiser()
    h = torch.randn(4, 8, T_H, 4)
    fr = torch.ones(4, 8, T_H)
    vx = torch.rand(4, 8) < 0.5
    o = m(h, fr, vx)
    print("predictor", f"{n_params(m):,}", tuple(o["pred"].shape), tuple(o["logit"].shape))
    d, _ = dn(h, fr, vx)
    print("denoiser", f"{n_params(dn):,}", tuple(d.shape))
    print("kalman", tuple(kalman_cv(h, fr)[0].shape))
