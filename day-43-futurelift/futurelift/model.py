"""FutureLift: backbone-agnostic proposal refiner built on a heterogeneous *future graph*.

Paper-sourced (arXiv:2610.22215 abstract/body): each proposal is lifted into temporal nodes
(R contiguous intervals per agent-mode, anchored at the interval's predicted endpoint, features from a
GRU over the segment in a local frame); typed edges (history / map / temporal-bidirectional / modal);
L layers of relation-specific multi-head attention with edge features, residual updates, pre-LN FFN;
graph built once, all segments updated simultaneously; heads predict per-interval coordinate
residuals, Laplace scales (ELU+1+0.1) and a confidence residual. Only proposal coords, scores and
scene context are used (no backbone features/queries).
Everything else (widths, depth, edge MLPs, layer ordering) is this repo's own default.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import T_H, T_F, N_AGENTS, N_MAP, SCALE


def _unit(v, eps=1e-6):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


class EdgeAttention(nn.Module):
    """Relation-specific multi-head attention with geometric edge features.

    Edge feature between query node i and key j (all computed in i's local heading frame):
    [dx, dy, dist, cos(dtheta), sin(dtheta)] -> MLP -> additive bias on keys AND values.
    """
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.edge = nn.Sequential(nn.Linear(5, d), nn.ReLU(), nn.Linear(d, 2 * d))

    def forward(self, qf, qp, qd, kf, kp, kd, mask=None):
        # qf (B,N,d) qp (B,N,2) qd (B,N,2) | kf (B,M,d) kp (B,M,2) kd (B,M,2) | mask (B,N,M) True=visible
        B, N, d = qf.shape; M = kf.shape[1]
        rel = kp[:, None] - qp[:, :, None]                               # (B,N,M,2) world-offset
        c, s = qd[..., 0:1], qd[..., 1:2]                                # (B,N,1)
        rx = c * rel[..., 0] + s * rel[..., 1]                           # rotate into query frame
        ry = -s * rel[..., 0] + c * rel[..., 1]
        cos_r = qd[:, :, None, 0] * kd[:, None, :, 0] + qd[:, :, None, 1] * kd[:, None, :, 1]
        sin_r = qd[:, :, None, 0] * kd[:, None, :, 1] - qd[:, :, None, 1] * kd[:, None, :, 0]
        dist = (rx ** 2 + ry ** 2 + 1e-9).sqrt()
        e = self.edge(torch.stack([rx / SCALE, ry / SCALE, dist / SCALE, cos_r, sin_r], -1))
        eb_k, eb_v = e.chunk(2, -1)                                      # (B,N,M,d) each
        q = self.q(qf).view(B, N, self.h, self.dh)
        k = (self.k(kf)[:, None] + eb_k).view(B, N, M, self.h, self.dh)
        v = (self.v(kf)[:, None] + eb_v).view(B, N, M, self.h, self.dh)
        att = torch.einsum("bnhd,bnmhd->bnhm", q, k) / math.sqrt(self.dh)
        if mask is not None:
            att = att.masked_fill(~mask[:, :, None, :], -1e9)
        out = torch.einsum("bnhm,bnmhd->bnhd", att.softmax(-1), v).reshape(B, N, d)
        return self.o(out)


class SceneEncoder(nn.Module):
    """Trained *with* the refiner (paper: QCNet-style encoder, not the backbone's features)."""
    def __init__(self, d):
        super().__init__()
        self.hist = nn.Sequential(nn.Linear(T_H * 4, d), nn.ReLU(), nn.Linear(d, d))
        self.role = nn.Embedding(N_AGENTS, d)
        self.map = nn.Sequential(nn.Linear(4, d), nn.ReLU(), nn.Linear(d, d))
        self.mix = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)

    def forward(self, batch):
        B = batch["hist"].shape[0]
        hf = self.hist(batch["hist"].flatten(2) / 10.0) + self.role.weight[None]      # (B,4,d)
        mf = self.map(batch["map"] / torch.tensor([SCALE, SCALE, 1, 1.], device=hf.device))  # (B,M,d)
        tok = torch.cat([hf, mf], 1)
        pad = torch.cat([~batch["mask"], torch.zeros(B, N_MAP, dtype=torch.bool, device=hf.device)], 1)
        tok = self.mix(tok, src_key_padding_mask=pad)
        h_pos = batch["hist"][:, :, -1, :2]                                           # last position
        h_dir = _unit(batch["hist"][:, :, -1, 2:4] + torch.tensor([1e-3, 0.], device=hf.device))
        return dict(h_feat=tok[:, :N_AGENTS], h_pos=h_pos, h_dir=h_dir, h_mask=batch["mask"],
                    m_feat=tok[:, N_AGENTS:], m_pos=batch["map"][..., :2], m_dir=batch["map"][..., 2:4])


class Lifter(nn.Module):
    """Lift each proposal into R temporal nodes: GRU over the segment in a local frame."""
    def __init__(self, d, R):
        super().__init__()
        self.R, self.L = R, T_F // R
        self.gru = nn.GRU(2, d, batch_first=True)
        self.anchor = nn.Sequential(nn.Linear(4, d), nn.ReLU(), nn.Linear(d, d))
        self.interval = nn.Embedding(R, d)
        self.score = nn.Linear(1, d)

    def forward(self, prop, logits):
        B, K, T, _ = prop.shape
        P = prop.view(B, K, self.R, self.L, 2)
        start = torch.cat([torch.zeros(B, K, 1, 2, device=prop.device), P[:, :, :-1, -1]], 2)   # (B,K,R,2)
        ext = torch.cat([start[:, :, :, None], P], 3)                                          # (B,K,R,L+1,2)
        anchor = P[:, :, :, -1]                                                                # (B,K,R,2)
        dirv = _unit(anchor - start + torch.tensor([1e-3, 0.], device=prop.device))
        rel = ext - anchor[:, :, :, None]
        c, s = dirv[..., 0][..., None], dirv[..., 1][..., None]
        loc = torch.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], -1) / 10.0
        _, hN = self.gru(loc.reshape(B * K * self.R, self.L + 1, 2))
        h = hN[0].view(B, K, self.R, -1)
        h = h + self.anchor(torch.cat([anchor / SCALE, dirv], -1)) + self.interval.weight[None, None]
        h = h + self.score(F.log_softmax(logits, -1)[:, :, None, None]).expand_as(h)
        return h, anchor, dirv


class FutureLiftLayer(nn.Module):
    def __init__(self, d, heads, temporal):
        super().__init__()
        self.temporal = temporal
        self.n = nn.ModuleList(nn.LayerNorm(d) for _ in range(5))
        self.hist, self.map, self.modal = (EdgeAttention(d, heads) for _ in range(3))
        self.temp = EdgeAttention(d, heads) if temporal != "none" else None
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h, pos, dirv, ctx):
        B, K, R, d = h.shape; N = K * R
        fp, fd = pos.reshape(B, N, 2), dirv.reshape(B, N, 2)
        x = self.n[0](h).reshape(B, N, d)
        h = h + self.hist(x, fp, fd, ctx["h_feat"], ctx["h_pos"], ctx["h_dir"],
                          ctx["h_mask"][:, None, :].expand(B, N, -1)).view(B, K, R, d)
        x = self.n[1](h).reshape(B, N, d)
        h = h + self.map(x, fp, fd, ctx["m_feat"], ctx["m_pos"], ctx["m_dir"]).view(B, K, R, d)
        if self.temp is not None:          # temporal edges among the R intervals of ONE proposal
            x = self.n[2](h).reshape(B * K, R, d)
            p, dr = pos.reshape(B * K, R, 2), dirv.reshape(B * K, R, 2)
            m = None
            if self.temporal == "causal":  # old-way: segment r sees only segments <= r
                m = torch.tril(torch.ones(R, R, dtype=torch.bool, device=h.device))[None].expand(B * K, R, R)
            h = h + self.temp(x, p, dr, x, p, dr, m).view(B, K, R, d)   # "bidirectional": m=None
        x = self.n[3](h).permute(0, 2, 1, 3).reshape(B * R, K, d)      # modal edges: modes at same interval
        p = pos.permute(0, 2, 1, 3).reshape(B * R, K, 2); dr = dirv.permute(0, 2, 1, 3).reshape(B * R, K, 2)
        h = h + self.modal(x, p, dr, x, p, dr).view(B, R, K, d).permute(0, 2, 1, 3)
        return h + self.ffn(self.n[4](h))


class FutureLift(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, R=3, temporal="bidirectional"):
        super().__init__()
        assert temporal in ("bidirectional", "causal", "none")
        self.R, self.L = R, T_F // R
        self.enc, self.lift = SceneEncoder(d), Lifter(d, R)
        self.layers = nn.ModuleList(FutureLiftLayer(d, heads, temporal) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, self.L * 4 + 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)   # start as identity refiner

    def forward(self, batch, prop, logits):
        prop, logits = prop.detach(), logits.detach()          # gradients stop at the frozen proposals
        B, K = prop.shape[:2]
        ctx = self.enc(batch)
        h, pos, dirv = self.lift(prop, logits)
        for layer in self.layers:
            h = layer(h, pos, dirv, ctx)                       # graph built once; all segments update together
        o = self.head(self.norm(h))                            # (B,K,R,L*4+1)
        res = o[..., :self.L * 2].reshape(B, K, self.R, self.L, 2) * 5.0          # metres, <=~5 m scale
        b = F.elu(o[..., self.L * 2:self.L * 4].reshape(B, K, self.R, self.L, 2)) + 1.0 + 0.1
        d_logit = o[..., -1].mean(2)                           # (B,K) pooled over intervals
        refined = prop + res.reshape(B, K, -1, 2)
        return refined, b.reshape(B, K, -1, 2), logits + d_logit
