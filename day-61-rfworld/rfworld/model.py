"""RFWorld-style rollout-free trajectory evaluator (my reconstruction from the public README; see SOURCING.md).

Idea (README-level): build ONE shared scene memory from camera + LiDAR + ego state + navigation; shape it with
temporal supervision at TRAINING time only; score every candidate trajectory by a SPARSE spatial readout of that
memory (no autoregressive world-model rollout at inference).

Tensor legend: B batch, N anchors, T waypoints, d model dim, M memory queries, H=W=16 memory grid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

XMAX, YMAX = 32.0, 16.0


def cnn(cin, d):
    """64x64 -> 16x16 feature map [B,d,16,16] (stride 4)."""
    return nn.Sequential(nn.Conv2d(cin, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, d, 3, 2, 1), nn.GELU(), nn.Conv2d(d, d, 3, 1, 1))


class SceneMemory(nn.Module):
    """Shared scene memory = (BEV memory map [B,d,16,16], M learned-query tokens [B,M,d]).
    Camera/LiDAR feature maps are fused; ego state + nav command modulate them (FiLM); the queries cross-attend
    to all tokens, so a global summary and a spatial map are both available to the readout."""
    def __init__(s, d=64, M=16, heads=4):
        super().__init__()
        s.cam, s.lid = cnn(3, d), cnn(2, d)
        s.fuse = nn.Sequential(nn.Conv2d(2 * d, d, 1), nn.GELU(), nn.Conv2d(d, d, 3, 1, 1))
        s.film = nn.Linear(3 + 3, 2 * d)                                            # (ego[3], nav[3]) -> (gamma, beta)
        s.pos = nn.Parameter(torch.randn(1, 256, d) * 0.02)                         # learned positional enc for 16x16 tokens
        s.q = nn.Parameter(torch.randn(1, M, d) * 0.1)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        s.ln = nn.LayerNorm(d)

    def forward(s, cam, lidar, ego, nav):
        fm = s.fuse(torch.cat([s.cam(cam), s.lid(lidar)], 1))                       # [B,d,16,16]
        g, b = s.film(torch.cat([ego, nav], -1)).chunk(2, -1)                       # [B,d] each
        fm = fm * (1 + g[:, :, None, None]) + b[:, :, None, None]                   # FiLM conditioning
        tok = fm.flatten(2).transpose(1, 2) + s.pos                                 # [B,256,d]
        mem, _ = s.attn(s.q.expand(len(cam), -1, -1), tok, tok)                     # [B,M,d] cross-attention
        return fm, s.ln(mem)


class SparseReadout(nn.Module):
    """Scores N candidate trajectories WITHOUT rolling anything forward.
    Each waypoint (x,y) bilinearly samples the BEV memory at exactly that location (sparse: N*T lookups, not a dense
    N x H x W pass); waypoint embeddings are concatenated per trajectory, then cross-attend to the memory queries."""
    def __init__(s, d=64, T=6, heads=4):
        super().__init__()
        s.T = T
        s.wp = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        s.step = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())                      # per-waypoint (embedding, sampled memory)
        s.traj = nn.Sequential(nn.Linear(T * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        s.ln = nn.LayerNorm(d)
        s.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 5))         # logits: nc, dac, ttc, ep, comf

    def forward(s, fm, mem, anchors):
        B, N, T, _ = anchors.shape
        gx = -anchors[..., 1] / YMAX                                                # col 0 = y +16  -> x_norm -1
        gy = 1 - anchors[..., 0] / (XMAX / 2)                                       # row 0 = x 32   -> y_norm -1
        grid = torch.stack([gx, gy], -1).view(B, N * T, 1, 2)                       # [B,N*T,1,2]
        samp = F.grid_sample(fm, grid, align_corners=False, padding_mode="zeros")   # [B,d,N*T,1]  sparse lookup
        samp = samp.squeeze(-1).transpose(1, 2).view(B, N, T, -1)                   # [B,N,T,d]
        h = s.step(torch.cat([s.wp(anchors), samp], -1))                            # [B,N,T,d]
        q = s.traj(h.flatten(2))                                                    # [B,N,d] one token per trajectory
        a, _ = s.attn(q, mem, mem)                                                  # trajectory token queries memory
        return s.head(s.ln(q + a))                                                  # [B,N,5]


class TemporalHead(nn.Module):
    """TRAINING-ONLY: predicts future BEV occupancy (+1.5 s, +3 s) from the BEV memory, which pushes dynamics into the
    memory. Dropped at inference -> evaluation is rollout-free."""
    def __init__(s, d=64):
        super().__init__()
        s.net = nn.Sequential(nn.Conv2d(d, d, 3, 1, 1), nn.GELU(), nn.Conv2d(d, 2, 1))

    def forward(s, fm): return s.net(fm)                                            # [B,2,16,16] logits


def compose_pdms(p):
    """p [B,N,5] probabilities (nc,dac,ttc,ep,comf) -> PDMS-style score [B,N] (same formula as the scorer)."""
    nc, dac, ttc, ep, cm = p.unbind(-1)
    return nc * dac * (5 * ep + 5 * ttc + 2 * cm) / 12


class RFWorldToy(nn.Module):
    def __init__(s, d=64, M=16, T=6):
        super().__init__()
        s.mem, s.read, s.temporal = SceneMemory(d, M), SparseReadout(d, T), TemporalHead(d)

    def forward(s, batch, anchors, with_aux=False):
        fm, mem = s.mem(batch["cam"], batch["lidar"], batch["ego"], batch["nav"])
        logits = s.read(fm, mem, anchors)
        out = {"logits": logits, "pdms": compose_pdms(torch.sigmoid(logits))}
        if with_aux: out["fut"] = s.temporal(fm)
        return out


class GlobalAnchorClassifier(nn.Module):
    """'Old way' baseline: pooled global feature -> MLP -> one score per fixed anchor (no per-waypoint readout)."""
    def __init__(s, N=256, d=64):
        super().__init__()
        s.mem = SceneMemory(d)
        s.head = nn.Sequential(nn.Linear(d * 16, 512), nn.GELU(), nn.Linear(512, N * 5))
        s.N = N

    def forward(s, batch, anchors=None, with_aux=False):
        _, mem = s.mem(batch["cam"], batch["lidar"], batch["ego"], batch["nav"])
        logits = s.head(mem.flatten(1)).view(len(mem), s.N, 5)
        return {"logits": logits, "pdms": compose_pdms(torch.sigmoid(logits))}


def loss_fn(out, tgt, fut_tgt=None, w_aux=0.5):
    """Sub-score BCE (nc,dac,ttc,ep,comf) + MSE on composed PDMS (+ temporal aux BCE when present)."""
    t = torch.stack([tgt["t_nc"], tgt["t_dac"], tgt["t_ttc"], tgt["t_ep"], tgt["t_comf"]], -1)
    l = F.binary_cross_entropy_with_logits(out["logits"], t) + 2 * F.mse_loss(out["pdms"], tgt["t_pdms"])
    if fut_tgt is not None and "fut" in out:
        l = l + w_aux * F.binary_cross_entropy_with_logits(out["fut"], fut_tgt)
    return l
