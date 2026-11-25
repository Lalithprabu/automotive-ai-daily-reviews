"""PyTorch core: DecayFormer, a compact hybrid state-space / attention forecaster of battery SOH trajectories.

Re-implementation from the ABSTRACT of arXiv 2611.19116 (Patel, Rao, Ayhan, "A Hybrid State Space-Attention
Architecture for Forecasting the Trajectories of Battery Degradation").  The abstract names four ingredients:

    (1) cycle-level SELECTIVE RECURRENCE with a single scalar state per channel
    (2) ATTENTION whose QUERIES come from the recurrent state
    (3) state-of-charge (SOC) INTERVAL STATISTICS as extra input features
    (4) a MASTER-CURVE HEAD that jointly estimates a scalar lifetime and a monotonically non-increasing SOH profile
        parameterised by NORMALISED lifetime

Everything below the level of those sentences (sizes, gate equations, number of queries, knot count, losses) is my
construction; see SOURCING.md.

Shapes (B cells, K = 100 early cycles, F input channels, D model width):

    x (B, K, F) --Linear--> e (B, K, D)  + sinusoidal cycle-index embedding
    selective recurrence, L layers, per channel d (scalar state h_d):
        delta_t = softplus(W_delta e_t + b)                     (B, K, D)   input-dependent step size
        a_t     = exp(-delta_t * exp(A_log))                    (B, K, D)   in (0, 1): how much of the past to keep
        h_t     = a_t * h_{t-1} + delta_t * (W_b e_t)           (B, D)      ONE scalar state per channel
        y_t     = (W_c e_t) * h_t + D_skip * e_t, gated by SiLU(W_z e_t), residual + LayerNorm
        cycles beyond the observed prefix k are masked: a = 1, input = 0 -> the state is frozen at cycle k
    state-query attention:
        Q = W_q h_k  reshaped to (B, n_q, D)                    n_q queries generated FROM the final state
        K, V = token sequence y (B, K, D), key padding beyond k
        ctx = MultiHeadAttention(Q, K, V) -> (B, n_q * D)
    master-curve head: z = [h_k, ctx] -> MLP ->
        log L_hat = log(1000) + out_0                            scalar lifetime (cycles to SOH 0.8)
        w = out_1..M (knot logits) -> G(u_i) = [0, cumsum(softmax(w))]  on knots u_i = i/M, G(0) = 0, G(1) = 1
        SOH(n) = 1 - 0.2 * G(n / L_hat)    (linear interpolation; beyond u = 1 the last slope is extended)
        softmax >= 0  =>  G non-decreasing  =>  SOH non-increasing BY CONSTRUCTION, and SOH(L_hat) = 0.8 exactly,
        so the lifetime and the curve can never disagree.
    free head ("old way", used by baselines / ablation): z -> MLP -> SOH at 61 absolute cycles 0..3000, unconstrained
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_MAX

LOG_L0 = math.log(1000.0)
FREE_GRID = torch.linspace(0, N_MAX, 61)                 # absolute cycles for the unconstrained head


def sinusoid(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


def interp1d(xq: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Batched linear interpolation with linear EXTRAPOLATION on both ends.
    xq (B, Q) query points, xp (P,) sorted knots shared by the batch, fp (B, P) values -> (B, Q)."""
    P = xp.shape[0]
    idx = torch.searchsorted(xp, xq.contiguous()).clamp(1, P - 1)          # right knot index
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp.gather(1, idx - 1), fp.gather(1, idx)
    w = (xq - x0) / (x1 - x0)
    return f0 + w * (f1 - f0)


# ============================================================================================ (1) recurrence
class SelectiveRecurrence(nn.Module):
    """Mamba-style selective scan with a SCALAR state per channel (state size N = 1)."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, 4 * d)                                    # delta, b, c, z in one matmul
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.5, 8.0, d)))  # per-channel decay rates
        self.D_skip = nn.Parameter(torch.ones(d))
        self.out = nn.Linear(d, d)

    def forward(self, e: torch.Tensor, valid: torch.Tensor):
        """e (B, K, D); valid (B, K) bool.  Returns y (B, K, D), h_final (B, D), gates a (B, K, D)."""
        x = self.norm(e)
        dl, b, c, z = self.proj(x).chunk(4, -1)
        delta = F.softplus(dl - 1.0)                                       # (B, K, D)  step size > 0
        a = torch.exp(-delta * torch.exp(self.A_log))                      # (B, K, D)  retention in (0, 1)
        v = delta * b                                                      # (B, K, D)  write
        m = valid[..., None].float()
        a = a * m + (1 - m)                                                # masked cycles: keep state
        v = v * m                                                          # ... and write nothing
        h = torch.zeros_like(x[:, 0])
        hs = []
        for t in range(x.shape[1]):                                        # K = 100: a plain loop is fine
            h = a[:, t] * h + v[:, t]
            hs.append(h)
        H = torch.stack(hs, 1)                                             # (B, K, D)
        y = (c * H + self.D_skip * x) * F.silu(z)
        return e + self.out(y), h, a


# ============================================================================================ (2) attention
class StateQueryAttention(nn.Module):
    """Cross-attention whose queries are generated from the recurrent summary state."""

    def __init__(self, d: int, n_q: int = 4, heads: int = 4):
        super().__init__()
        self.n_q, self.d = n_q, d
        self.q = nn.Linear(d, n_q * d)
        self.kv_norm = nn.LayerNorm(d)
        self.mha = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, h: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor):
        """h (B, D), tokens (B, K, D), valid (B, K) -> ctx (B, n_q*D), attention weights (B, n_q, K)."""
        B = h.shape[0]
        q = self.q(h).view(B, self.n_q, self.d)                            # (B, n_q, D)
        kv = self.kv_norm(tokens)
        ctx, w = self.mha(q, kv, kv, key_padding_mask=~valid, need_weights=True, average_attn_weights=True)
        return ctx.reshape(B, -1), w


# ============================================================================================ (4) heads
class MasterCurveHead(nn.Module):
    """Scalar lifetime + monotone normalised-lifetime master curve; SOH(L_hat) = 0.8 by construction."""

    def __init__(self, d_in: int, n_knots: int = 16, hidden: int = 128):
        super().__init__()
        self.M = n_knots
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, 1 + n_knots))
        self.register_buffer("u", torch.linspace(0, 1, n_knots + 1), persistent=False)

    def forward(self, z: torch.Tensor):
        o = self.mlp(z)
        log_life = LOG_L0 + o[:, 0]                                        # (B,)
        G = torch.cat([torch.zeros_like(o[:, :1]), torch.cumsum(o[:, 1:].softmax(-1), -1)], -1)   # (B, M+1)
        return dict(log_life=log_life, G=G)

    def soh(self, out: dict, cycles: torch.Tensor) -> torch.Tensor:
        """cycles (B, Q) or (Q,) -> SOH (B, Q)."""
        life = out["log_life"].exp()[:, None]
        if cycles.dim() == 1:
            cycles = cycles[None].expand(life.shape[0], -1)
        u = cycles / life                                                  # normalised lifetime
        return 1.0 - 0.2 * interp1d(u, self.u, out["G"])

    def lifetime(self, out: dict) -> torch.Tensor:
        return out["log_life"].exp().clamp(max=N_MAX)


class FreeHead(nn.Module):
    """Unconstrained SOH at fixed absolute cycles ('old way'); lifetime = first crossing of 0.8."""

    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, len(FREE_GRID)))
        self.register_buffer("grid", FREE_GRID.clone(), persistent=False)
        with torch.no_grad():                                              # start from a gentle linear fade
            self.mlp[-1].bias.copy_(self.grid / 1000.0)

    def forward(self, z: torch.Tensor):
        return dict(soh_grid=1.0 - 0.2 * self.mlp(z))                      # (B, 61)

    def soh(self, out: dict, cycles: torch.Tensor) -> torch.Tensor:
        g = out["soh_grid"]
        if cycles.dim() == 1:
            cycles = cycles[None].expand(g.shape[0], -1)
        return interp1d(cycles, self.grid, g)

    def lifetime(self, out: dict) -> torch.Tensor:
        n = torch.linspace(0, N_MAX, 601, device=out["soh_grid"].device)
        s = self.soh(out, n)                                               # (B, 601)
        below = s <= 0.8
        first = torch.where(below.any(1), below.float().argmax(1), torch.full_like(below[:, 0], 600, dtype=torch.long))
        i1 = first.clamp(min=1)
        s0, s1 = s.gather(1, (i1 - 1)[:, None])[:, 0], s.gather(1, i1[:, None])[:, 0]
        frac = ((s0 - 0.8) / (s0 - s1).clamp(min=1e-6)).clamp(0, 1)
        life = n[i1 - 1] + frac * (n[1] - n[0])
        return torch.where(below.any(1), life, torch.full_like(life, float(N_MAX)))


# ============================================================================================ models
class DecayFormer(nn.Module):
    def __init__(self, n_feat: int, d: int = 64, n_layers: int = 2, n_q: int = 4, heads: int = 4,
                 head: str = "master", attention: bool = True, k_max: int = 100):
        super().__init__()
        self.inp = nn.Linear(n_feat, d)
        self.register_buffer("pe", sinusoid(k_max, d), persistent=False)
        self.rec = nn.ModuleList([SelectiveRecurrence(d) for _ in range(n_layers)])
        self.attn = StateQueryAttention(d, n_q, heads) if attention else None
        d_z = d + (n_q * d if attention else 0)
        self.head = MasterCurveHead(d_z) if head == "master" else FreeHead(d_z)

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        """x (B, K, F), valid (B, K) -> head output dict (+ 'attn' (B, n_q, K), 'gates' (B, K, D) for plots)."""
        e = self.inp(x) + self.pe[: x.shape[1]]
        gates = None
        for layer in self.rec:
            e, h, gates = layer(e, valid)
        z, w = h, None
        if self.attn is not None:
            ctx, w = self.attn(h, e, valid)
            z = torch.cat([h, ctx], -1)
        out = self.head(z)
        out.update(attn=w, gates=gates)
        return out


class TransformerFree(nn.Module):
    """'Scaled' baseline: a bigger vanilla Transformer encoder + unconstrained SOH head."""

    def __init__(self, n_feat: int, d: int = 128, layers: int = 3, heads: int = 4, k_max: int = 100):
        super().__init__()
        self.inp = nn.Linear(n_feat, d)
        self.register_buffer("pe", sinusoid(k_max, d), persistent=False)
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = FreeHead(2 * d, hidden=256)

    def forward(self, x, valid):
        hseq = self.norm(self.enc(self.inp(x) + self.pe[: x.shape[1]], src_key_padding_mask=~valid))
        m = valid[..., None].float()
        mean = (hseq * m).sum(1) / m.sum(1)
        last = hseq.gather(1, (valid.sum(1) - 1).clamp(min=0)[:, None, None].expand(-1, 1, hseq.shape[-1]))[:, 0]
        out = self.head(torch.cat([mean, last], -1))
        out.update(attn=None, gates=None)
        return out


class GRUFree(nn.Module):
    """Classic recurrent baseline: 2-layer GRU + unconstrained SOH head."""

    def __init__(self, n_feat: int, hidden: int = 128):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, 2, batch_first=True, dropout=0.1)
        self.head = FreeHead(hidden)

    def forward(self, x, valid):
        hseq, _ = self.gru(x)
        last = hseq.gather(1, (valid.sum(1) - 1).clamp(min=0)[:, None, None].expand(-1, 1, hseq.shape[-1]))[:, 0]
        out = self.head(last)
        out.update(attn=None, gates=None)
        return out


# ============================================================================================ variants + loss
SCALAR_IDX = list(range(5))
ALL_IDX = list(range(25))
VARIANTS = {
    "decayformer": dict(cls="df", feats=ALL_IDX, head="master", attention=True),
    "df_freehead": dict(cls="df", feats=ALL_IDX, head="free", attention=True),
    "df_nosoc": dict(cls="df", feats=SCALAR_IDX, head="master", attention=True),
    "df_noattn": dict(cls="df", feats=ALL_IDX, head="master", attention=False),
    "transformer": dict(cls="tf", feats=ALL_IDX),
    "gru": dict(cls="gru", feats=ALL_IDX),
}


def build(name: str) -> nn.Module:
    v = VARIANTS[name]
    nf = len(v["feats"])
    if v["cls"] == "df":
        return DecayFormer(nf, head=v["head"], attention=v["attention"])
    if v["cls"] == "tf":
        return TransformerFree(nf)
    return GRUFree(nf)


def trajectory_loss(model: nn.Module, out: dict, traj: torch.Tensor, grid: torch.Tensor, life: torch.Tensor,
                    w_life: float = 0.5) -> tuple[torch.Tensor, dict]:
    """MAPE over the valid (pre-EOL) part of the true trajectory + a log-lifetime term for the master head."""
    valid = ~torch.isnan(traj)
    pred = model.head.soh(out, grid)                                       # (B, G)
    t = torch.nan_to_num(traj, nan=1.0)
    ape = ((pred - t).abs() / t) * valid
    mape = ape.sum() / valid.sum()
    loss = mape
    parts = dict(mape=float(mape.detach()))
    if "log_life" in out:
        ll = (out["log_life"] - life.log()).abs().mean()
        loss = loss + w_life * 0.05 * ll                                   # 0.05: log-error -> MAPE-ish scale
        parts["log_life"] = float(ll.detach())
    return loss, parts


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    B, K = 3, 100
    x = torch.randn(B, K, 25)
    valid = torch.arange(K)[None] < torch.tensor([100, 50, 20])[:, None]
    grid = torch.arange(0, 3001, 10.0)
    for name in VARIANTS:
        m = build(name)
        xi = x[..., VARIANTS[name]["feats"]]
        out = m(xi, valid)
        s = m.head.soh(out, grid)
        L = m.head.lifetime(out)
        print(f"{name:12s} params {n_params(m):8,d}  soh {tuple(s.shape)}  life {L.detach().numpy().round(0)}")
