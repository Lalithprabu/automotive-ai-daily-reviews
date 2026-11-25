"""PyTorch core: SpecFlow, conditional flow matching in the ACCELERATION SPECTRUM for multi-agent trajectory forecasting.

Re-implementation from the ABSTRACT of arXiv 2611.26390 (Xu, Han, Zhou, "SpecFlow: Flow Matching in the Acceleration
Spectrum for Multi-Agent Trajectory Forecasting").  The abstract names five ingredients:

    (1) the generative state is the DCT coefficients of FUTURE ACCELERATIONS (not positions)
    (2) conditional flow matching transports Gaussian noise to those acceleration spectra
    (3) velocity and position are recovered by FIXED ANALYTICAL INTEGRATION from the last observed state
    (4) a JOINT PHASE-SPACE LOSS supervises acceleration, velocity and position
    (5) an ENDPOINT PREDICTOR proposes scored destinations from the observed context; the top one conditions the
        spectral denoiser, which uses ALTERNATING ATTENTION over hypotheses and agents

Sizes, the integration scheme, loss weights, K, step counts and the encoder are my construction (see SOURCING.md).

Shapes (B scenes, A = 6 agent slots, H hypotheses, T = 12 future steps, dt = 0.4 s, D = 128):

    hist (B, A, 8, 4) -> GRU -> h (B, A, D) -> 2 x social attention with relative-pose keys -> ctx (B, A, D)
    endpoint head:  ctx -> K = 6 candidates e_k (B, A, K, 2) [m, agent frame] + logits (B, A, K)
    state x (B, H, A, 24):  24 = 12 DCT coefficients x 2 axes, normalised per coefficient
        c = DCT_II(a) along time, a_k = (v_k - v_{k-1}) / dt,  v_k = (p_k - p_{k-1}) / dt,  p_0 = 0, v_0 = last obs
    flow matching (rectified path):  x_t = (1 - t) x_0 + t x_1,  x_0 ~ N(0, I),  target velocity u = x_1 - x_0
    denoiser tokens (B, H, A, D) = MLP(x_t) + emb(t) + W ctx + MLP(endpoint)
        repeat 2x:  attention over H (per agent)  ->  attention over A (per hypothesis, padded agents masked)  ->  FFN
    integration (fixed, differentiable, no parameters):
        a = IDCT(c)  (B, H, A, 12, 2);   v_k = v_0 + dt * sum_{j<=k} a_j;   p_k = dt * sum_{j<=k} v_j
    phase-space loss on the one-step estimate x_1_hat = x_t + (1 - t) u_hat:  L1 on a, v and p
    sampling: 10 Euler steps t = 0 -> 1, H = 20 hypotheses, then integrate
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DT, T_OBS, T_PRED

T = T_PRED


def dct_matrix(n: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix C (n, n): c = C @ a, a = C.T @ c."""
    k = torch.arange(n, dtype=torch.float64)[:, None]
    i = torch.arange(n, dtype=torch.float64)[None]
    C = torch.cos(math.pi * (i + 0.5) * k / n) * math.sqrt(2.0 / n)
    C[0] /= math.sqrt(2.0)
    return C.float()


# ============================================================================================ kinematics
class Kinematics(nn.Module):
    """Fixed analytical maps between acceleration spectrum, acceleration, velocity and position (no parameters)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("C", dct_matrix(T), persistent=False)

    def spec_to_acc(self, c: torch.Tensor) -> torch.Tensor:          # (..., T, 2) -> (..., T, 2)
        return torch.einsum("kt,...kd->...td", self.C, c)

    def acc_to_spec(self, a: torch.Tensor) -> torch.Tensor:
        return torch.einsum("kt,...td->...kd", self.C, a)

    @staticmethod
    def integrate(a: torch.Tensor, v0: torch.Tensor):
        """a (..., T, 2), v0 (..., 2) -> v, p (..., T, 2).  Semi-implicit Euler, exact inverse of `differentiate`."""
        v = v0.unsqueeze(-2) + DT * torch.cumsum(a, -2)
        p = DT * torch.cumsum(v, -2)
        return v, p

    @staticmethod
    def differentiate(p: torch.Tensor, v0: torch.Tensor):
        """p (..., T, 2) relative to the last observed position, v0 (..., 2) -> v, a (..., T, 2)."""
        pp = torch.cat([torch.zeros_like(p[..., :1, :]), p], -2)
        v = (pp[..., 1:, :] - pp[..., :-1, :]) / DT
        vv = torch.cat([v0.unsqueeze(-2), v], -2)
        a = (vv[..., 1:, :] - vv[..., :-1, :]) / DT
        return v, a


# ============================================================================================ context encoder
class SocialLayer(nn.Module):
    """Agent-to-agent attention; keys/values carry agent j's relative pose seen from agent i."""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.rel = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h, rel, mask):
        """h (B, A, D), rel (B, A, A, 7), mask (B, A) -> (B, A, D)."""
        B, A, D = h.shape
        x = self.n1(h)
        kv = x[:, None, :, :] + self.rel(rel)                              # (B, A_i, A_j, D)
        q = self.q(x).view(B, A, 1, self.h, self.dk)
        k = self.k(kv).view(B, A, A, self.h, self.dk)
        v = self.v(kv).view(B, A, A, self.h, self.dk)
        att = (q * k).sum(-1) / math.sqrt(self.dk)                         # (B, A_i, A_j, heads)
        att = att.masked_fill(~mask[:, None, :, None], -1e9).softmax(2)
        y = (att[..., None] * v).sum(2).reshape(B, A, D)
        h = h + self.o(y)
        return h + self.ff(self.n2(h))


class ContextEncoder(nn.Module):
    def __init__(self, d: int = 128, layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(4, 64)
        self.kind = nn.Embedding(2, 64)
        self.gru = nn.GRU(64, d, batch_first=True)
        self.social = nn.ModuleList([SocialLayer(d) for _ in range(layers)])

    def forward(self, hist, kind, rel, mask):
        B, A, To, _ = hist.shape
        x = self.inp(hist / torch.tensor([10.0, 2.0, 10.0, 10.0], device=hist.device)) + self.kind(kind)[:, :, None]
        _, h = self.gru(x.reshape(B * A, To, -1))
        h = h[-1].view(B, A, -1)
        for lay in self.social:
            h = lay(h, rel, mask)
        return h


class EndpointHead(nn.Module):
    """K scored destination candidates from the observed context only."""

    def __init__(self, d: int, K: int = 6):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 3 * K))

    def forward(self, ctx):
        o = self.mlp(ctx).view(*ctx.shape[:2], self.K, 3)
        return o[..., :2] * 20.0, o[..., 2]                                # metres (agent frame), logits


# ============================================================================================ denoiser
def t_embed(t: torch.Tensor, d: int) -> torch.Tensor:
    half = d // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * 1000.0 * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


class AlternatingBlock(nn.Module):
    """Self-attention over hypotheses (per agent), then over agents (per hypothesis), then FFN."""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.nh, self.na, self.nf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.hyp = nn.MultiheadAttention(d, heads, batch_first=True)
        self.agt = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, mask):
        B, H, A, D = x.shape
        y = self.nh(x).permute(0, 2, 1, 3).reshape(B * A, H, D)            # tokens = hypotheses of one agent
        y = self.hyp(y, y, y, need_weights=False)[0].view(B, A, H, D).permute(0, 2, 1, 3)
        x = x + y
        y = self.na(x).reshape(B * H, A, D)                                # tokens = agents of one hypothesis
        kpm = (~mask)[:, None].expand(B, H, A).reshape(B * H, A)
        y = self.agt(y, y, y, key_padding_mask=kpm, need_weights=False)[0].view(B, H, A, D)
        x = x + y
        return x + self.ff(self.nf(x))


class SpectralDenoiser(nn.Module):
    def __init__(self, d: int = 128, layers: int = 2, state_dim: int = 2 * T, use_endpoint: bool = True):
        super().__init__()
        self.d = d
        self.inp = nn.Sequential(nn.Linear(state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.tmlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.cproj = nn.Linear(d, d)
        self.ep = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d)) if use_endpoint else None
        self.blocks = nn.ModuleList([AlternatingBlock(d) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, state_dim))

    def forward(self, x, t, ctx, mask, endpoint=None):
        """x (B, H, A, S), t (B,), ctx (B, A, D), mask (B, A), endpoint (B, H, A, 2) -> velocity (B, H, A, S)."""
        z = self.inp(x) + self.tmlp(t_embed(t, self.d))[:, None, None] + self.cproj(ctx)[:, None]
        if self.ep is not None:
            z = z + self.ep(endpoint / 20.0)
        for b in self.blocks:
            z = b(z, mask)
        return self.out(z)


# ============================================================================================ full model
class SpecFlow(nn.Module):
    """state: 'dct' (SpecFlow), 'acc' (acceleration in time, no DCT) or 'pos' (positions frame by frame, old way)."""

    def __init__(self, state: str = "dct", endpoint: bool = True, phase: bool = True, d: int = 128, K: int = 6,
                 h_train: int = 8, w_phase=(0.05, 0.1, 0.1), w_ep: float = 0.1):
        super().__init__()
        self.state, self.use_ep, self.phase, self.h_train = state, endpoint, phase, h_train
        self.w_phase, self.w_ep = w_phase, w_ep
        self.kin = Kinematics()
        self.enc = ContextEncoder(d)
        self.eph = EndpointHead(d, K)
        self.den = SpectralDenoiser(d, use_endpoint=endpoint)
        self.register_buffer("mu", torch.zeros(T, 2))                      # state normalisation (set from data)
        self.register_buffer("sd", torch.ones(T, 2))

    # ---------------------------------------------------------------- state <-> kinematics
    def raw_state(self, p, v0):
        """Ground-truth future p (..., T, 2) -> un-normalised state (..., T, 2)."""
        if self.state == "pos":
            return p
        _, a = self.kin.differentiate(p, v0)
        return self.kin.acc_to_spec(a) if self.state == "dct" else a

    def kinematics(self, x, v0):
        """Normalised state (..., 2T) -> a, v, p (..., T, 2) by fixed integration (or differencing for 'pos')."""
        s = x.unflatten(-1, (T, 2)) * self.sd + self.mu
        if self.state == "pos":
            v, a = self.kin.differentiate(s, v0)
            return a, v, s
        a = self.kin.spec_to_acc(s) if self.state == "dct" else s
        v, p = self.kin.integrate(a, v0)
        return a, v, p

    @torch.no_grad()
    def fit_norm(self, fut, v0, mask):
        s = self.raw_state(fut, v0)[mask]                                  # (N, T, 2)
        self.mu.copy_(s.mean(0))
        self.sd.copy_(s.std(0).clamp(min=1e-3))

    # ---------------------------------------------------------------- training
    def loss(self, b: dict):
        hist, kind, rel, mask, fut, v0 = b["hist"], b["kind"], b["rel"], b["mask"], b["fut"], b["v0"]
        B, A = mask.shape
        H = self.h_train
        ctx = self.enc(hist, kind, rel, mask)
        ep, logit = self.eph(ctx)                                          # (B, A, K, 2), (B, A, K)
        end = fut[:, :, -1]                                                # (B, A, 2)
        dist = (ep - end[:, :, None]).norm(dim=-1)                         # (B, A, K)
        kstar = dist.argmin(-1)
        ep_best = ep.gather(2, kstar[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
        l_ep = (F.smooth_l1_loss(ep_best, end, reduction="none").sum(-1)
                + F.cross_entropy(logit.flatten(0, 1), kstar.flatten(), reduction="none").view(B, A))
        l_ep = (l_ep * mask).sum() / mask.sum()
        x1 = ((self.raw_state(fut, v0) - self.mu) / self.sd).flatten(-2)   # (B, A, 2T)
        x1 = x1[:, None].expand(B, H, A, -1)
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device)
        tt = t[:, None, None, None]
        xt = (1 - tt) * x0 + tt * x1
        cond = ep_best.detach()[:, None].expand(B, H, A, 2) if self.use_ep else None
        u = self.den(xt, t, ctx, mask, cond)
        m = mask[:, None, :, None].float()
        l_fm = (((u - (x1 - x0)) ** 2) * m).sum() / (m.sum() * H * x1.shape[-1])
        loss = l_fm + self.w_ep * l_ep
        parts = dict(fm=float(l_fm.detach()), ep=float(l_ep.detach()))
        if self.phase:
            x1h = xt + (1 - tt) * u
            a, v, p = self.kinematics(x1h, v0[:, None].expand(B, H, A, 2))
            v_gt, a_gt = self.kin.differentiate(fut, v0)
            mm = mask[:, None, :, None, None].float()
            n = mm.sum() * H * T * 2
            la = ((a - a_gt[:, None]).abs() * mm).sum() / n
            lv = ((v - v_gt[:, None]).abs() * mm).sum() / n
            lp = ((p - fut[:, None]).abs() * mm).sum() / n
            loss = loss + self.w_phase[0] * la + self.w_phase[1] * lv + self.w_phase[2] * lp
            parts.update(a=float(la.detach()), v=float(lv.detach()), p=float(lp.detach()))
        return loss, parts

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def sample(self, b: dict, H: int = 20, steps: int = 10, ep_mode: str = "top1", keep_path: bool = False,
               generator: torch.Generator | None = None):
        """Returns dict p (B, H, A, T, 2) [agent frame], a, v, endpoints, logits, cond (B, H, A, 2) and, if keep_path,
        the state at every Euler step (steps+1, B, H, A, 2T)."""
        hist, kind, rel, mask, v0 = b["hist"], b["kind"], b["rel"], b["mask"], b["v0"]
        B, A = mask.shape
        ctx = self.enc(hist, kind, rel, mask)
        ep, logit = self.eph(ctx)
        cond = None
        if self.use_ep:
            if ep_mode == "top1":                                          # the abstract: top candidate conditions
                cond = ep.gather(2, logit.argmax(-1)[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
                cond = cond[:, None].expand(B, H, A, 2)
            else:                                                          # extension: hypotheses spread over K
                pr = logit.softmax(-1)                                     # (B, A, K)
                cdf = pr.cumsum(-1)
                qs = (torch.arange(H, device=pr.device) + 0.5) / H         # stratified quantiles
                idx = (qs[None, None, :, None] > cdf[:, :, None, :]).sum(-1).clamp(max=ep.shape[2] - 1)  # (B, A, H)
                cond = ep.gather(2, idx[..., None].expand(-1, -1, -1, 2)).permute(0, 2, 1, 3)
        x = torch.randn(B, H, A, 2 * T, generator=generator, device=hist.device)
        path = [x] if keep_path else None
        for i in range(steps):
            t = torch.full((B,), i / steps, device=x.device)
            x = x + self.den(x, t, ctx, mask, cond) / steps
            if keep_path:
                path.append(x)
        a, v, p = self.kinematics(x, v0[:, None].expand(B, H, A, 2))
        out = dict(p=p, a=a, v=v, endpoints=ep, logits=logit, cond=cond)
        if keep_path:
            out["path"] = torch.stack(path)
        return out


VARIANTS = {
    "specflow": dict(state="dct", endpoint=True, phase=True),
    "pos_state": dict(state="pos", endpoint=True, phase=True),
    "acc_time": dict(state="acc", endpoint=True, phase=True),
    "no_phase": dict(state="dct", endpoint=True, phase=False),
    "no_endpoint": dict(state="dct", endpoint=False, phase=True),
}


def build(name: str) -> SpecFlow:
    return SpecFlow(**VARIANTS[name])


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    B, A = 4, 6
    b = dict(hist=torch.randn(B, A, T_OBS, 4), kind=torch.randint(0, 2, (B, A)), rel=torch.randn(B, A, A, 7),
             mask=torch.arange(A)[None] < torch.tensor([6, 4, 3, 2])[:, None], fut=torch.randn(B, A, T, 2).cumsum(2),
             v0=torch.randn(B, A, 2))
    for nm in VARIANTS:
        m = build(nm)
        loss, parts = m.loss(b)
        loss.backward()
        out = m.sample(b, H=5, steps=3)
        print(f"{nm:12s} params {n_params(m):,}  loss {float(loss):.3f}  p {tuple(out['p'].shape)}")
