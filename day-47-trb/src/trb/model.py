"""Temporal Residual Bottleneck (TRB) for asynchronous collaborative perception.

Paper-sourced (arXiv:2610.10090): pose-warped anchor B; delay-conditioned xLSTM (m-s-m, k=4 history) for
residual temporal evidence; attention-free gated residual bottleneck (zero-init output, scale s=1.5);
object gate from detector objectness on stale+warped features (union, dilation 4, floor 0.05, detached);
F_hat = B + G * dF; L = L_det + 0.02 * L_app (directional/magnitude/improvement/over-application).
Everything else (channel widths, dilated-conv encoder, BEV size, ego fusion head) is this repo's own default.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .xlstm import XLSTMStack
from .data import C_FEAT, HALF, warp_history


@dataclass
class TRBConfig:
    c: int = C_FEAT          # BEV feature channels
    d: int = 32              # temporal embedding dim (paper: 64)
    heads: int = 4
    layout: str = "msm"
    k_hist: int = 4
    bott: int = 32           # bottleneck channels (paper: 64)
    hidden: int = 64         # bottleneck hidden (paper: 256)
    conf_ch: int = 8
    scale: float = 1.5
    gate_dilate: int = 4
    gate_floor: float = 0.05
    use_gate: bool = True
    max_delay: float = 5.0


def fourier(delay, n=4, max_delay=5.0):
    f = 2.0 ** torch.arange(n, dtype=delay.dtype, device=delay.device) * math.pi / (2 * max_delay)
    a = delay[:, None] * f[None]
    return torch.cat([delay[:, None] / max_delay, torch.sin(a), torch.cos(a)], -1)  # [B, 1+2n]


def dconv(i, o, dil):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=dil, dilation=dil), nn.GELU())


class ObjectnessHead(nn.Module):
    """Tiny detector-objectness head on BEV features (stand-in for the frozen detector's objectness map)."""
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        return self.net(x)  # logits


class TemporalResidualBottleneck(nn.Module):
    def __init__(self, cfg: TRBConfig):
        super().__init__()
        self.cfg = cfg
        c, d = cfg.c, cfg.d
        self.enc = nn.Sequential(dconv(c, d, 1), dconv(d, d, 2), dconv(d, d, 4))     # RF ~ 15 cells
        self.film = nn.Linear(1 + 8, 2 * d)                                           # Fourier(delay) -> gamma,beta
        self.pose = nn.Linear(3, d)                                                   # relative pose token
        self.xlstm = XLSTMStack(d, cfg.layout, cfg.heads)
        self.evid = nn.Sequential(dconv(d, d, 2), dconv(d, d, 1))
        self.pred = nn.Conv2d(d, c, 1)                                                # predicted fresh feature P
        self.conf = nn.Conv2d(d, cfg.conf_ch, 1)
        self.bn_in = nn.Conv2d(d + 2 * c + cfg.conf_ch, cfg.bott, 1)
        self.bn_mid = nn.Sequential(nn.GELU(), nn.Conv2d(cfg.bott, cfg.hidden, 3, padding=2, dilation=2), nn.GELU(),
                                    nn.Conv2d(cfg.hidden, cfg.hidden, 3, padding=1), nn.GELU())
        self.bn_out = nn.Conv2d(cfg.hidden, c, 1)
        nn.init.zeros_(self.bn_out.weight); nn.init.zeros_(self.bn_out.bias)           # zero-init: F_hat == B at init
        self.objectness = ObjectnessHead(c)

    def gate(self, stale, warped):
        cfg = self.cfg
        with torch.no_grad():
            o = torch.maximum(torch.sigmoid(self.objectness(stale)), torch.sigmoid(self.objectness(warped)))
            k = 2 * cfg.gate_dilate + 1
            o = F.max_pool2d(o, k, stride=1, padding=cfg.gate_dilate)
            return o.clamp_min(cfg.gate_floor)

    def forward(self, hist, rel, delay):
        """hist [B,K,C,H,W] stale frames in their own frames; rel [B,K,3]; delay [B] (steps).
        Returns dict(F_hat, B, dF, G, P)."""
        cfg = self.cfg
        Bsz, K, C, H, W = hist.shape
        warped = warp_history(hist, rel)                       # [B,K,C,H,W], all in the NOW frame
        S, Bw = hist[:, 0], warped[:, 0]                       # latest stale (raw) / its pose-warped anchor
        gb = self.film(torch.cat([fourier(delay, 4, cfg.max_delay)], -1))  # [B,2d]
        gamma, beta = gb.chunk(2, -1)
        seq = []
        for i in reversed(range(K)):                           # oldest -> newest
            e = self.enc(warped[:, i])                         # [B,d,H,W]
            e = e * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
            e = e + self.pose(rel[:, i])[:, :, None, None]
            seq.append(e)
        seq = torch.stack(seq, 1)                              # [B,K,d,H,W]
        seq = seq.permute(0, 3, 4, 1, 2).reshape(Bsz * H * W, K, cfg.d)
        h = self.xlstm(seq).view(Bsz, H, W, cfg.d).permute(0, 3, 1, 2)
        E = self.evid(h)                                       # recurrent temporal evidence
        P = self.pred(E)                                       # predicted fresh feature (no recon loss)
        z = torch.cat([E, S - Bw, Bw - P, torch.sigmoid(self.conf(E))], 1)  # evidence + discrepancy + agreement + confidence
        dF = cfg.scale * self.bn_out(self.bn_mid(self.bn_in(z)))
        G = self.gate(S, Bw) if cfg.use_gate else torch.ones_like(Bw[:, :1])
        return dict(F_hat=Bw + G * dF, B=Bw, dF=G * dF, G=G, P=P, S=S)


class AsyncCollabDetector(nn.Module):
    """variant: 'stale' (no compensation) | 'warp' (pose warp only) | 'trb' (full)."""
    def __init__(self, variant="trb", cfg: TRBConfig = None):
        super().__init__()
        self.variant = variant
        self.cfg = cfg or TRBConfig()
        self.trb = TemporalResidualBottleneck(self.cfg) if variant == "trb" else None
        self.objectness = self.trb.objectness if self.trb is not None else ObjectnessHead(self.cfg.c)
        c = self.cfg.c
        self.head = nn.Sequential(nn.Conv2d(2 * c, 48, 3, padding=1), nn.GELU(), nn.Conv2d(48, 48, 3, padding=2, dilation=2),
                                  nn.GELU(), nn.Conv2d(48, 1, 1))

    def collab_feature(self, batch):
        if self.variant == "stale":
            return dict(F_hat=batch["hist"][:, 0], B=batch["hist"][:, 0], dF=None, G=None, P=None, S=batch["hist"][:, 0])
        if self.variant == "warp":
            B = warp_history(batch["hist"][:, :1], batch["rel"][:, :1])[:, 0]
            return dict(F_hat=B, B=B, dF=None, G=None, P=None, S=batch["hist"][:, 0])
        return self.trb(batch["hist"], batch["rel"], batch["delay"])

    def forward(self, batch):
        out = self.collab_feature(batch)
        out["logits"] = self.head(torch.cat([batch["ego"], out["F_hat"]], 1))
        return out
