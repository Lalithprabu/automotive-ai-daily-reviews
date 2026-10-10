"""MESSENGER-style memory-enhanced sequential scene-flow forecasting (this repo's reconstruction).

Paper-sourced (abstract level): a sequential pipeline with a MEMORY BUFFER of past flow
estimates + latent states; at each frame the stored flows/states are CORRELATED and RETRIEVED
to INITIALISE the current flow in a next-frame FORECASTING setup; an UNCERTAINTY-AWARE
REWEIGHTING module filters unreliable retrievals to limit error accumulation.
Everything else (dims, kNN/attention form, GRU-free state, chaining) is this repo's own default.

Shapes: B batch, N points per frame, d feature dim, K memory depth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cdist(a, b):
    # exact pairwise distances: the default matmul-based path loses ~1e-2 m at |x|~20 in float32
    return torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")


def gather_pts(x, idx):
    """x (B,M,C), idx (B,N,k) -> (B,N,k,C)"""
    b = torch.arange(x.shape[0], device=x.device).view(-1, 1, 1)
    return x[b, idx]


def knn_interp(q, ref, val, k=3):
    """Inverse-distance interpolation of `val` (defined at `ref`) at query points `q`.
    q (B,N,2), ref (B,M,2), val (B,M,C) -> (B,N,C), nearest-ref distance (B,N)."""
    d = cdist(q, ref)
    dist, idx = d.topk(min(k, ref.shape[1]), dim=-1, largest=False)
    w = 1.0 / (dist + 1e-2)
    w = w / w.sum(-1, keepdim=True)
    return (w.unsqueeze(-1) * gather_pts(val, idx)).sum(2), dist[..., 0]


class LocalEnc(nn.Module):
    """PointNet-style local geometry encoder: kNN relative offsets -> MLP -> max-pool, + position."""

    def __init__(self, d, k=8):
        super().__init__()
        self.k = k
        self.rel = nn.Sequential(nn.Linear(2, d), nn.ReLU(), nn.Linear(d, d))
        self.pos = nn.Linear(2, d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, P):                                   # P (B,N,2)
        idx = cdist(P, P).topk(self.k + 1, dim=-1, largest=False).indices[..., 1:]
        rel = (gather_pts(P, idx) - P.unsqueeze(2)) / 2.0   # (B,N,k,2)
        f = self.rel(rel).max(2).values                     # (B,N,d)
        return self.out(torch.cat([f, self.pos(P / 40.0)], -1))


class PairFlowEstimator(nn.Module):
    """Two-frame (observed) flow: for each point of frame a, soft-correspondence over ALL points
    of frame b (dense attention; N is small) + a latent state h used as memory content.
        score_ij = q_i . k_j / sqrt(d) + MLP(p_j - p_i)
        flow_i   = sum_j softmax_j(score_ij) (p_j - p_i)  + refinement(h_i)"""

    def __init__(self, d=48):
        super().__init__()
        self.enc = LocalEnc(d)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.off = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.state = nn.Sequential(nn.Linear(2 * d + 2, d), nn.Tanh())
        self.refine = nn.Linear(d, 2)
        self.d = d

    def forward(self, Pa, Pb):                              # (B,N,2) each
        fa, fb = self.enc(Pa), self.enc(Pb)
        off = Pb.unsqueeze(1) - Pa.unsqueeze(2)             # (B,N,M,2)  p_j - p_i
        score = torch.einsum("bnd,bmd->bnm", self.q(fa), self.k(fb)) / self.d ** 0.5
        score = score + self.off(off / 8.0).squeeze(-1)
        w = score.softmax(-1)                               # (B,N,M)
        coarse = (w.unsqueeze(-1) * off).sum(2)             # (B,N,2)
        ctx = torch.einsum("bnm,bmd->bnd", w, self.v(fb))
        h = self.state(torch.cat([fa, ctx, coarse / 8.0], -1))
        return coarse + self.refine(h), h


class MemoryForecaster(nn.Module):
    """Next-frame flow FORECAST from a memory of past (points, flow, state) entries.

    1. RETRIEVE: for each age a=1..K, advance that entry's points to the current time by chaining
       interpolated flows of the newer entries, then soft-match them to the current points.
    2. REWEIGHT: a small MLP on (retrieved state, match distance, disagreement with the newest
       flow) gives a per-point confidence g_a in (0,1) that scales that age's contribution.
    3. FORECAST: head predicts a residual on top of the newest retrieved flow (the 'init').
    K=1, use_gate=False  == the 'old way': carry the last pairwise flow forward + learned residual.
    """

    def __init__(self, d=48, K=3, topk=6, use_gate=True):
        super().__init__()
        self.K, self.topk, self.use_gate, self.d = K, topk, use_gate, d
        self.enc = LocalEnc(d)
        self.log_s = nn.Parameter(torch.zeros(()))
        self.gate = nn.Sequential(nn.Linear(d + 2, 32), nn.ReLU(), nn.Linear(32, 1))
        in_dim = d + K * (2 + d + 1 + 1)
        self.head = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.o_delta, self.o_logsig, self.o_h = nn.Linear(128, 2), nn.Linear(128, 1), nn.Linear(128, d)

    def retrieve(self, P_cur, entries):
        """-> list over ages 1..K of dict(f (B,N,2), h (B,N,d), dist (B,N), mask float)."""
        L = len(entries)
        out = []
        for a in range(1, self.K + 1):
            e = L - a
            if e < 0:
                B, N = P_cur.shape[:2]
                z = P_cur.new_zeros
                out.append(dict(f=z(B, N, 2), h=z(B, N, self.d), dist=z(B, N), mask=0.0))
                continue
            pos = entries[e]["P"] + entries[e]["f"]                    # entry's points at time e+1
            for j in range(e + 1, L):                                  # hop forward to current time
                fj, _ = knn_interp(pos, entries[j]["P"], entries[j]["f"])
                pos = pos + fj
            d = cdist(P_cur, pos)                                # (B,N,N)
            dist, idx = d.topk(min(self.topk, d.shape[-1]), dim=-1, largest=False)
            w = (-(dist ** 2) / (2 * torch.exp(2 * self.log_s))).softmax(-1).unsqueeze(-1)
            out.append(dict(f=(w * gather_pts(entries[e]["f"], idx)).sum(2),
                            h=(w * gather_pts(entries[e]["h"], idx)).sum(2),
                            dist=dist[..., 0], mask=1.0))
        return out

    def forward(self, P_cur, entries):
        r = self.retrieve(P_cur, entries)
        feats, gates = [self.enc(P_cur)], []
        for a, ra in enumerate(r):
            if ra["mask"] == 0.0:
                g = P_cur.new_zeros(P_cur.shape[:2])
            elif self.use_gate:
                dis = (ra["f"] - r[0]["f"]).norm(dim=-1)
                g = torch.sigmoid(self.gate(torch.cat([ra["h"], ra["dist"].unsqueeze(-1),
                                                       dis.unsqueeze(-1)], -1)).squeeze(-1))
            else:
                g = P_cur.new_ones(P_cur.shape[:2])
            gates.append(g)
            gg = g.unsqueeze(-1)
            feats += [gg * ra["f"], gg * ra["h"], gg * ra["dist"].unsqueeze(-1), gg]
        z = self.head(torch.cat(feats, -1))
        flow = r[0]["f"] + self.o_delta(z)                             # retrieval-initialised forecast
        return flow, self.o_logsig(z).squeeze(-1), torch.tanh(self.o_h(z)), torch.stack(gates, 0)


def rollout(model, P_t, entries, H):
    """Autoregressive extrapolation with NO new observations. Each predicted flow becomes a new
    memory entry and advances the (virtual) point cloud. Point index i tracks the same physical
    point across steps, so cumulative displacement is directly comparable with GT.
    Returns list of (cum_disp (B,N,2), logsig (B,N), gates (K,B,N))."""
    entries, P, cum, outs = list(entries), P_t, torch.zeros_like(P_t), []
    for _ in range(H):
        flow, logsig, h, gates = model(P, entries)
        cum = cum + flow
        entries.append(dict(P=P, f=flow, h=h))
        P = P + flow
        outs.append((cum, logsig, gates))
    return outs
