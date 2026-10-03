"""MomWorld-style momentum-aware latent world model + MoFlow refiner (small-scale reconstruction).

NOT paper code.  The paper (arXiv:2609.33737) describes, at abstract level:
  * extracting motion trends from history and propagating *latent momentum* into the future,
    predicting both configuration and momentum states;
  * a momentum-persistence mechanism, scene-conditioned momentum updates, and a
    scene-adaptive reset gate that suppresses outdated momentum on abrupt scene changes;
  * MoFlow: a momentum-conditioned flow-matching module refining a base trajectory in a few
    integration steps, with horizon-aware residual fusion.
Everything below (dimensions, gate equations, losses) is this project's own reconstruction.

Shape legend: B batch, T history steps, H horizon steps, D latent width.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import F_IN, H as H_DEFAULT, T_HIST, POS_SCALE


@dataclass
class MomWorldConfig:
    d: int = 64
    horizon: int = H_DEFAULT
    f_in: int = F_IN
    use_momentum: bool = True      # False = "old way": single-latent-state rollout, no momentum
    use_reset_gate: bool = True    # ablation switch for the scene-adaptive reset gate
    use_moflow: bool = True        # ablation switch for the flow-matching refiner
    flow_steps: int = 4            # Euler integration steps at inference ("only a few")


class HistoryEncoder(nn.Module):
    """GRU over history -> per-step hidden states h (B,T,D)."""
    def __init__(self, cfg):
        super().__init__()
        self.inp = nn.Linear(cfg.f_in, cfg.d)
        self.gru = nn.GRU(cfg.d, cfg.d, batch_first=True)

    def forward(self, hist):                       # (B,T,F)
        h, _ = self.gru(torch.relu(self.inp(hist)))
        return h                                   # (B,T,D)


class MomentumExtractor(nn.Module):
    """m0 = attention-weighted sum of consecutive latent differences (a learned 'latent velocity')."""
    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
        self.proj = nn.Linear(d, d)

    def forward(self, h):                          # (B,T,D)
        diffs = h[:, 1:] - h[:, :-1]               # (B,T-1,D)
        w = torch.softmax(self.score(diffs).squeeze(-1), dim=1)   # (B,T-1)
        return self.proj((w.unsqueeze(-1) * diffs).sum(1))        # (B,D)


class MomentumWorldModel(nn.Module):
    """Latent rollout carrying BOTH configuration z_k and momentum m_k.

        c      = scene context read from the last history state          (B,D)
        g_k    = tanh(W_g [z_{k-1}, c, e_k])        scene-conditioned momentum update
        keep_k = sigmoid(W_r [m_{k-1}, z_{k-1}, c, e_k])   scene-adaptive reset gate
        m_k    = keep_k * alpha * m_{k-1} + g_k     persistence (alpha in (0,1), learned) + update
        z_k    = z_{k-1} + m_k                      configuration advances by its momentum
    Old-way ablation: z_k = z_{k-1} + MLP([z_{k-1}, c, e_k]); no m, no gate.
    """
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d; self.cfg = cfg
        self.hor_emb = nn.Embedding(cfg.horizon, d)
        self.scene_ctx = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.extract = MomentumExtractor(d)
        self.upd = nn.Linear(3 * d, d)
        self.gate = nn.Linear(4 * d, d)
        nn.init.constant_(self.gate.bias, 2.0)                 # start by trusting momentum
        self.alpha_logit = nn.Parameter(torch.full((d,), 2.0)) # alpha ~ 0.88 at init
        self.plain = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, h):
        B, T, D = h.shape
        z = h[:, -1]
        c = self.scene_ctx(z)
        m = self.extract(h) if self.cfg.use_momentum else torch.zeros_like(z)
        zs, ms, keeps = [], [], []
        alpha = torch.sigmoid(self.alpha_logit)
        for k in range(self.cfg.horizon):
            e = self.hor_emb.weight[k].expand(B, -1)
            if self.cfg.use_momentum:
                g = torch.tanh(self.upd(torch.cat([z, c, e], -1)))
                keep = torch.sigmoid(self.gate(torch.cat([m, z, c, e], -1))) if self.cfg.use_reset_gate \
                    else torch.ones_like(z)
                m = keep * alpha * m + g
                z = z + m
                keeps.append(keep)
            else:
                z = z + self.plain(torch.cat([z, c, e], -1))
            zs.append(z); ms.append(m)
        zs = torch.stack(zs, 1); ms = torch.stack(ms, 1)                       # (B,H,D)
        keeps = torch.stack(keeps, 1) if keeps else torch.ones_like(zs)
        return zs, ms, keeps


class MoFlow(nn.Module):
    """Momentum-conditioned flow-matching refiner over the horizon.

    v_theta(x_s, s | cond): per-step Conv1d stack (kernel 3 across the horizon) on
    [x_s (2), cond (D->32), s]; integrates base -> refined in `flow_steps` Euler steps.
    Horizon-aware residual fusion: out = base + sigmoid(phi_k) * (x_1 - base).
    """
    def __init__(self, cfg):
        super().__init__()
        d, hd = cfg.d, 48
        self.cond = nn.Linear(2 * d, 32)
        self.net = nn.Sequential(
            nn.Conv1d(2 + 32 + 1, hd, 3, padding=1), nn.GELU(),
            nn.Conv1d(hd, hd, 3, padding=1), nn.GELU(),
            nn.Conv1d(hd, 2, 3, padding=1))
        self.phi = nn.Parameter(torch.zeros(cfg.horizon))   # horizon-aware fusion logits

    def velocity(self, x, s, cond):                 # x (B,H,2) cond (B,H,32) s (B,)
        sb = s.view(-1, 1, 1).expand(-1, x.shape[1], 1)
        inp = torch.cat([x, cond, sb], -1).transpose(1, 2)
        return self.net(inp).transpose(1, 2)

    def fusion_weights(self):
        return torch.sigmoid(self.phi)

    def refine(self, base, z, m, steps):
        cond = torch.relu(self.cond(torch.cat([z, m], -1)))
        x = base
        for i in range(steps):
            s = torch.full((base.shape[0],), i / steps, device=base.device)
            x = x + self.velocity(x, s, cond) / steps
        w = self.fusion_weights().view(1, -1, 1)
        return base + w * (x - base)

    def loss(self, base, target, z, m):
        """Flow-matching loss: x_s = base + s (target-base);  v* = target - base  (base detached)."""
        base = base.detach()
        cond = torch.relu(self.cond(torch.cat([z, m], -1)))
        s = torch.rand(base.shape[0], device=base.device)
        x_s = base + s.view(-1, 1, 1) * (target - base)
        return F.mse_loss(self.velocity(x_s, s, cond), target - base)


class MomWorld(nn.Module):
    def __init__(self, cfg: MomWorldConfig | None = None):
        super().__init__()
        self.cfg = cfg or MomWorldConfig()
        d = self.cfg.d
        self.enc = HistoryEncoder(self.cfg)
        self.world = MomentumWorldModel(self.cfg)
        self.wp_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))   # per-step displacement
        self.scene_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))  # future lead-vehicle position
        self.moflow = MoFlow(self.cfg) if self.cfg.use_moflow else None

    def forward(self, hist, refine=True):
        """hist (B,T,F) -> dict(base (B,H,2) metres, plan (B,H,2) metres, scene (B,H,2) metres, z, m, keep)."""
        h = self.enc(hist)                                  # (B,T,D)
        z, m, keep = self.world(h)                          # (B,H,D) each
        base = torch.cumsum(self.wp_head(z), 1) * POS_SCALE # displacement increments -> positions (B,H,2)
        scene = self.scene_head(z) * POS_SCALE * 5          # lead position (ego frame)
        plan = self.moflow.refine(base / POS_SCALE, z, m, self.cfg.flow_steps) * POS_SCALE if (self.moflow and refine) else base
        return dict(base=base, plan=plan, scene=scene, z=z, m=m, keep=keep)

    def compute_loss(self, hist, future, lead_future, w_scene=0.5, w_flow=1.0):
        out = self(hist, refine=False)
        l_base = F.smooth_l1_loss(out["base"] / POS_SCALE, future / POS_SCALE)
        l_scene = F.smooth_l1_loss(out["scene"] / (POS_SCALE * 5), lead_future / (POS_SCALE * 5))
        total = l_base + w_scene * l_scene
        l_flow = torch.zeros(())
        if self.moflow is not None:
            l_flow = self.moflow.loss(out["base"] / POS_SCALE, future / POS_SCALE, out["z"].detach(), out["m"].detach())
            # train the horizon-aware fusion weights phi on the *fused* few-step output (base/z/m detached)
            fused = self.moflow.refine(out["base"].detach() / POS_SCALE, out["z"].detach(), out["m"].detach(), self.cfg.flow_steps)
            l_fuse = F.smooth_l1_loss(fused, future / POS_SCALE)
            total = total + w_flow * l_flow + l_fuse
        return total, dict(base=l_base.item(), scene=l_scene.item(), flow=float(l_flow.detach()))
