"""
ForeseeDrive - Future-Token Conditioning for Anticipatory Generative Driving (arXiv 2611.14286)
Toy PyTorch re-implementation of the CORE mechanism:

    present scene x_t ──► [frozen] base generator G  ──► K candidates τ_b, logits ℓ        (the old way)
            │                       ▲  ▲  ▲  (adapter injected after every decoder layer l)
            └──► token predictor h_ψ ──► future tokens Ẑ^f = {(e_i, c_i, r_i)}  (actor / route / occupancy / margin)
                                     └─► u = Pool(Attn(E_q(Ẑ^f), E_x(x_t)))                   scene vector
                 z̃_k^(l) = z_k^(l) + α_l · A_l(z_k^(l), u)        bias-free A_l(z, 0) = 0,  α_l zero-init
                 ℓ̃_k     = ℓ_k + b_k(u)                           b_k(0) = 0
            ──► K candidates τ_f (geometry already shaped by the anticipated future) ──► select argmax ℓ̃

Paper facts followed: query-indexed tokens (actor, route, occupancy, TTC-margin) with (e, c, r) per query;
predictor distilled from a logged-future teacher with Smooth-L1(e) + BCE(c) + BCE(r) and a stop-gradient
target; low-confidence tokens suppressed before injection; geometry path + logit path with exact
zero-token no-op; pretrained generator FROZEN, only adapters trained; candidate-preservation loss;
TTC-margin guard (train hinge + inference rejection of safe->unsafe candidates).
My choices: DiffusionDrive is replaced by a deterministic anchor-based K-mode transformer decoder (no
diffusion), all sizes, the exact form of A_l (bias-free FiLM), the attention pooling with bias-free values,
constant-velocity projection for the TTC margin.  Not the authors' code.

Shape legend: B batch, N actors, P history steps, K candidates, H future steps, Q token queries, d width.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tokens as TK
from . import world as W

TOK_IN = TK.E_DIM + 1          # injected token vector v_i = c_i * [e_i, r_i]


def mlp(i, h, o, bias_out=True):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o, bias=bias_out))


def actor_features(b: dict) -> torch.Tensor:
    """(B, N, P, 4) history + type -> (B, N, P*4 + 3) normalised features."""
    h = b["hist"]
    s = torch.tensor([20.0, 5.0, 10.0, 2.0])
    f = (h / s).flatten(2)
    typ = F.one_hot(b["typ"], 2).float()
    ego_v = (b["v_ego"] / 10.0)[:, None, None].expand(-1, h.shape[1], 1)
    return torch.cat([f, typ, ego_v], -1)


# =========================================================================== scene encoder
class SceneEncoder(nn.Module):
    """Actors + ego -> context tokens (B, N+1, d) and a key-padding mask (True = pad)."""

    def __init__(self, d=64, layers=2):
        super().__init__()
        self.act = mlp(W.HIST * 4 + 3, d, d)
        self.ego = mlp(1, d, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.0, batch_first=True,
                                                                    norm_first=True), layers,
                                         enable_nested_tensor=False)

    def forward(self, b: dict):
        a = self.act(actor_features(b))                                      # (B, N, d)
        e = self.ego(b["v_ego"][:, None, None] / 10.0)                       # (B, 1, d)
        x = torch.cat([e, a], 1)                                             # (B, N+1, d)
        pad = torch.cat([torch.zeros_like(b["valid"][:, :1]), ~b["valid"]], 1)
        return self.enc(x, src_key_padding_mask=pad), pad


# =========================================================================== base generator (frozen later)
class BaseGenerator(nn.Module):
    """DiffusionDrive stand-in: K anchor queries -> L transformer-decoder layers -> K trajectories + logits.

    forward(..., cond=None) is the base planner.  With cond=(adapters, alphas, u) the adapter is applied
    after every layer: z <- z + alpha_l * A_l(z, u).
    """

    def __init__(self, anchors: torch.Tensor, d=64, layers=3):
        super().__init__()
        self.register_buffer("anchors", anchors)                             # (K, H, 2)
        self.K, self.L, self.d = anchors.shape[0], layers, d
        self.scene = SceneEncoder(d)
        self.q = mlp(W.H * 2, d, d)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d, 4, 2 * d, 0.0, batch_first=True, norm_first=True)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.off = mlp(d, d, W.H * 2)
        self.logit = mlp(d, d, 1)
        self.register_buffer("scale", torch.tensor([20.0, 3.5]))

    def encode(self, b):
        return self.scene(b)

    def forward(self, b: dict, cond=None, ctx=None):
        mem, pad = ctx if ctx is not None else self.scene(b)
        B = mem.shape[0]
        z = self.q((self.anchors / self.scale).flatten(1))[None].expand(B, -1, -1)       # (B, K, d)
        for l, layer in enumerate(self.layers):
            z = layer(z, mem, memory_key_padding_mask=pad)                   # self-attn(K) + cross-attn(scene)
            if cond is not None:                                             # ---- geometry path (paper eq.)
                adapters, alphas, u = cond
                if adapters is not None:
                    z = z + alphas[l] * adapters[l](z, u)                    # z̃ = z + α_l A_l(z, u)
        z = self.norm(z)
        traj = self.anchors[None] + self.off(z).view(B, self.K, W.H, 2) * self.scale * 0.25   # (B, K, H, 2)
        return traj, self.logit(z)[..., 0], z, mem


# =========================================================================== token predictor h_ψ
class TokenPredictor(nn.Module):
    """Present context (+ base candidates for margin queries) -> one token per query.

    queries: N_A actor slots (id emb + embedding of the ranked actor), N_R route + N_O occupancy (id emb),
             K margin queries (id emb + embedding of base candidate k).
    out: e (B, Q, 4), c_logit (B, Q), r_logit (B, Q)
    """

    def __init__(self, K: int, d=64, layers=2):
        super().__init__()
        self.Q = TK.N_SCENE_Q + K
        self.scene = SceneEncoder(d)
        self.qid = nn.Parameter(torch.randn(self.Q, d) * 0.1)
        self.cand = mlp(W.H * 2, d, d)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 4, 2 * d, 0.0, batch_first=True,
                                                                    norm_first=True), layers)
        self.head = mlp(d, d, TK.E_DIM + 2)

    def forward(self, b: dict, base_traj: torch.Tensor):
        mem, pad = self.scene(b)                                             # (B, N+1, d)
        B = mem.shape[0]
        q = self.qid[None].expand(B, -1, -1).clone()                         # (B, Q, d)
        # actor slot i <- ranked actor token (mem index = actor + 1 because of the ego token)
        r = b["rank"]                                                        # (B, N_A), -1 = empty
        g = torch.gather(mem, 1, (r.clamp(min=0) + 1)[..., None].expand(-1, -1, mem.shape[-1]))
        q[:, :TK.N_A] = q[:, :TK.N_A] + g * (r >= 0)[..., None]
        q[:, TK.N_SCENE_Q:] = q[:, TK.N_SCENE_Q:] + self.cand((base_traj / torch.tensor([20.0, 3.5])).flatten(2))
        y = self.head(self.dec(q, mem, memory_key_padding_mask=pad))         # (B, Q, 6)
        return y[..., :TK.E_DIM], y[..., TK.E_DIM], y[..., TK.E_DIM + 1]


def inject(e, c, r, from_logits: bool, c_thresh: float = 0.5):
    """Build v_i = 1[c_i] * [e_i, r_i]; low-confidence tokens are suppressed (exact zeros)."""
    if from_logits:
        keep = (torch.sigmoid(c) >= c_thresh).float()
        r = torch.sigmoid(r)
    else:
        keep = c
    return keep[..., None] * torch.cat([e, r[..., None]], -1)               # (B, Q, 5)


# =========================================================================== ForeseeDrive conditioning
class TokenPool(nn.Module):
    """u = Pool(Attn(E_q(Z), E_x(x))):  M scene queries attend over the Q injected tokens.

    Values are a BIAS-FREE projection of the token vector, and all-zero tokens are masked out, so
    Z = 0  =>  every value is 0 (or every key masked)  =>  u = 0 exactly.
    """

    def __init__(self, Q: int, d=64, M=4, heads=4):
        super().__init__()
        self.M, self.h, self.dh = M, heads, d // heads
        self.qid = nn.Parameter(torch.randn(Q, d) * 0.1)                    # fixed query identities
        self.k = nn.Linear(TOK_IN, d)                                        # keys may have bias (masked anyway)
        self.v = nn.Linear(TOK_IN, d, bias=False)                            # E_q values: bias-free
        self.sq = nn.Parameter(torch.randn(M, d) * 0.1)
        self.sx = nn.Linear(d, M * d)                                        # E_x(x): scene-dependent queries
        self.o = nn.Linear(M * d, d, bias=False)

    def forward(self, v_tok: torch.Tensor, scene_vec: torch.Tensor):
        B, Q, _ = v_tok.shape
        live = v_tok.abs().sum(-1) > 0                                       # (B, Q)
        k = (self.k(v_tok) + self.qid[None]).view(B, Q, self.h, self.dh).transpose(1, 2)    # (B, h, Q, dh)
        v = self.v(v_tok).view(B, Q, self.h, self.dh).transpose(1, 2)
        q = (self.sq[None] + self.sx(scene_vec).view(B, self.M, -1)).view(B, self.M, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)                 # (B, h, M, Q)
        att = att.masked_fill(~live[:, None, None], float("-inf"))
        w = torch.nan_to_num(att.softmax(-1), nan=0.0)                       # all masked -> zeros
        pooled = (w @ v).transpose(1, 2).reshape(B, -1)                      # (B, M*d)
        return self.o(pooled)                                                # (B, d)


class ZeroGatedAdapter(nn.Module):
    """A(z, u) = (W_a z) ⊙ (W_g u) + W_b u, every map bias-free  =>  A(z, 0) = 0."""

    def __init__(self, d=64):
        super().__init__()
        self.a = nn.Linear(d, d, bias=False)
        self.g = nn.Linear(d, d, bias=False)
        self.b = nn.Linear(d, d, bias=False)

    def forward(self, z, u):                                                 # z (B, K, d), u (B, d)
        return self.a(z) * self.g(u)[:, None] + self.b(u)[:, None]


class ForeseeDrive(nn.Module):
    """Frozen base generator + trainable future-token conditioning (geometry path + logit path)."""

    def __init__(self, base: BaseGenerator, geometry=True, logit=True, d=64):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        Q = TK.N_SCENE_Q + base.K
        self.pool = TokenPool(Q, d)
        self.geometry, self.use_logit = geometry, logit
        self.adapters = nn.ModuleList([ZeroGatedAdapter(d) for _ in range(base.L)]) if geometry else None
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(base.L)])   # zero-init
        self.lw = nn.Linear(d, d, bias=False)                                # b_k(u) = <W u, z_k> / sqrt(d)

    def forward(self, b: dict, v_tok: torch.Tensor, base_out=None):
        self.base.eval()
        with torch.no_grad():
            ctx = self.base.encode(b)                                        # (mem, pad), frozen
            tb, lb, zb, mem = base_out if base_out is not None else self.base(b, ctx=ctx)
        scene_vec = mem[:, 0]                                                # ego token after encoding
        u = self.pool(v_tok, scene_vec)                                      # (B, d), exactly 0 for zero tokens
        if self.geometry:
            tf, lf, zf, _ = self.base(b, cond=(self.adapters, self.alphas, u), ctx=ctx)
        else:
            tf, lf, zf = tb, lb, zb
        if self.use_logit:
            lf = lf + (self.lw(u)[:, None] * zf).sum(-1) / math.sqrt(zf.shape[-1])   # ℓ̃ = ℓ + b_k(u)
        return dict(traj=tf, logit=lf, u=u, base_traj=tb, base_logit=lb)


def n_params(m: nn.Module, trainable=False) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))
