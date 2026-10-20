"""STC-MGNet-style pedestrian crossing-intention model (this repo's reconstruction).

Paper (arXiv:2610.23356, abstract-level): a pose-graph encoder enhanced with
Spatial-Temporal-Channel (STC) attention -- three parallel branches emphasising
joints / frames / feature channels, merged with LEARNABLE fusion weights -- plus a
multi-stage cross-modal channel modulation in which visual and ego-vehicle-motion
cues gate the pose feature channels after each graph-encoding stage.

Tensor convention everywhere: (B, C, T, V) = batch, channels, frames, joints.
All layer sizes, the exact attention formulas and the gate form are this repo's
own defaults, NOT paper-sourced (see SOURCING.md).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_JOINTS = 14
SKELETON_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                  (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (8, 11)]


def normalized_adjacency(V: int = NUM_JOINTS) -> torch.Tensor:
    """Symmetric-normalised adjacency with self loops: D^-1/2 (A+I) D^-1/2."""
    A = torch.eye(V)
    for i, j in SKELETON_EDGES:
        A[i, j] = A[j, i] = 1.0
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]


class STCAttention(nn.Module):
    """Three parallel attention branches (joint / frame / channel) fused by
    learnable weights.   x: (B,C,T,V) -> (B,C,T,V)

    joint  : mean over T -> (B,C,V) -> conv1d over V  -> sigmoid -> (B,1,1,V)
    frame  : mean over V -> (B,C,T) -> conv1d over T  -> sigmoid -> (B,1,T,1)
    channel: mean over T,V -> (B,C) -> bottleneck MLP -> sigmoid  -> (B,C,1,1)
    out = x * (w_j*a_j + w_f*a_f + w_c*a_c),  w = 3*softmax(theta)  (init: all 1)
    """

    def __init__(self, C: int, reduction: int = 4, k: int = 3):
        super().__init__()
        self.joint = nn.Conv1d(C, 1, k, padding=k // 2)
        self.frame = nn.Conv1d(C, 1, k, padding=k // 2)
        self.channel = nn.Sequential(nn.Linear(C, max(C // reduction, 4)), nn.ReLU(),
                                     nn.Linear(max(C // reduction, 4), C))
        self.theta = nn.Parameter(torch.zeros(3))  # learnable fusion logits

    def fusion_weights(self) -> torch.Tensor:
        return 3.0 * torch.softmax(self.theta, dim=0)  # sums to 3, init (1,1,1)

    def forward(self, x):
        a_j = torch.sigmoid(self.joint(x.mean(2)))[:, :, None, :]       # (B,1,1,V)
        a_f = torch.sigmoid(self.frame(x.mean(3)))[:, :, :, None]       # (B,1,T,1)
        a_c = torch.sigmoid(self.channel(x.mean((2, 3))))[:, :, None, None]  # (B,C,1,1)
        w = self.fusion_weights()
        out = x * (w[0] * a_j + w[1] * a_f + w[2] * a_c)
        return out, {"joint": a_j.flatten(1), "frame": a_f.flatten(1), "w": w.detach()}


class CrossModalChannelGate(nn.Module):
    """Modality-specific channel gating: visual and ego embeddings each produce a
    per-channel gate in (0,2); features are multiplied by both. Zero-initialised so
    the module starts as an exact identity (gate = 2*sigmoid(0) = 1)."""

    def __init__(self, C: int, embed_dim: int):
        super().__init__()
        self.g_vis = nn.Linear(embed_dim, C)
        self.g_ego = nn.Linear(embed_dim, C)
        for m in (self.g_vis, self.g_ego):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, v_emb, e_emb):
        gv = 2 * torch.sigmoid(self.g_vis(v_emb))[:, :, None, None]
        ge = 2 * torch.sigmoid(self.g_ego(e_emb))[:, :, None, None]
        return x * gv * ge


class GraphStage(nn.Module):
    """Spatial graph conv (fixed skeleton A + learnable residual B) -> STC attention
    (optional) -> temporal conv -> residual."""

    def __init__(self, cin, cout, k_t, use_stc, dropout):
        super().__init__()
        self.register_buffer("A", normalized_adjacency())
        self.B = nn.Parameter(torch.zeros(NUM_JOINTS, NUM_JOINTS))
        self.proj = nn.Conv2d(cin, cout, 1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.stc = STCAttention(cout) if use_stc else None
        self.tconv = nn.Conv2d(cout, cout, (k_t, 1), padding=(k_t // 2, 0))
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = torch.einsum("bctv,vw->bctw", self.proj(x), self.A + self.B)  # graph conv
        h = F.relu(self.bn1(h))
        info = None
        if self.stc is not None:
            h, info = self.stc(h)
        h = self.drop(self.bn2(self.tconv(h)))
        return F.relu(h + self.res(x)), info


class STCMGNet(nn.Module):
    """pose (B,4,T,V) + visual (B,T,Dv) + ego (B,T,De) -> crossing logit (B,)

    use_stc / use_modulation switch the two paper ideas off, giving the "old way"
    pose-only ST-GCN baseline (False, False)."""

    def __init__(self, in_channels=4, hidden=(32, 64, 64), visual_dim=6, ego_dim=2,
                 embed_dim=32, temporal_kernel=5, dropout=0.2,
                 use_stc=True, use_modulation=True):
        super().__init__()
        self.use_modulation = use_modulation
        chans = [in_channels, *hidden]
        self.stages = nn.ModuleList(
            GraphStage(chans[i], chans[i + 1], temporal_kernel, use_stc, dropout)
            for i in range(len(hidden)))
        self.vis_enc = nn.Sequential(nn.Linear(visual_dim, embed_dim), nn.ReLU())
        self.vis_out = nn.Linear(2 * embed_dim, embed_dim)       # [mean_T ; last frame]
        self.ego_enc = nn.GRU(ego_dim, embed_dim, batch_first=True)
        self.gates = nn.ModuleList(CrossModalChannelGate(h, embed_dim) for h in hidden) \
            if use_modulation else None
        self.head = nn.Sequential(nn.Linear(hidden[-1], 32), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(32, 1))

    def encode_context(self, visual, ego):
        z = self.vis_enc(visual)                                  # (B,T,E)
        v_emb = self.vis_out(torch.cat([z.mean(1), z[:, -1]], -1))  # (B,E)
        _, h = self.ego_enc(ego)                                  # h: (1,B,E)
        return v_emb, h[0]

    def forward(self, pose, visual, ego, return_attn=False):
        x = pose
        v_emb, e_emb = self.encode_context(visual, ego) if self.use_modulation else (None, None)
        attn = None
        for i, st in enumerate(self.stages):
            x, info = st(x)
            if info is not None:
                attn = info                                       # keep the last stage's maps
            if self.use_modulation:
                x = self.gates[i](x, v_emb, e_emb)               # cross-modal modulation
        logit = self.head(x.mean((2, 3))).squeeze(-1)             # (B,)
        return (logit, attn) if return_attn else logit
