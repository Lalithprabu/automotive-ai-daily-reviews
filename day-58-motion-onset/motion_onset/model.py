"""Meta-learned recurrent scene-graph GNN for motion-onset prediction (my reconstruction of the architecture described in the paper summary).

Per frame t:
  nodes  = visible objects (+ one persistent SCENE node linked to every object)
  node in = Linear([pos, vel, vel_valid, is_scene]) + Linear(semantic embedding) + Linear(h_{t-1})        -> [B,M,d]   (M = N+1)
  edges  = kNN (k<=5, <=10 m) among visible objects, scene<->object edges, self loops; edge feat = [unit(dx,dy), dvx, dvy, is_scene_edge]
  2 x GATv2 (4 heads)  ->  per-node GRUCell(h_{t-1})  ->  MLP -> logit -> sigmoid = P(object starts moving within H)
No weight update happens at test time: adaptation lives in the GRU memory ("meta-learning in context", trained episodically).
Tensor legend: B batch, N objects, M=N+1 nodes, d hidden, E=5 edge features, D semantic dim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .synthetic import D


class GATv2(nn.Module):
    """Dense GATv2 (Brody et al.): e_ij = a . LeakyReLU(Ws h_j + Wd h_i + We e_ij); adj[b,i,j]=True means message j -> i."""
    def __init__(s, d=64, heads=4, e_dim=5):
        super().__init__()
        assert d % heads == 0
        s.h, s.dh = heads, d // heads
        s.Ws, s.Wd, s.We = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(e_dim, d, bias=False)
        s.a = nn.Parameter(torch.randn(heads, s.dh) * 0.1)
        s.out, s.norm = nn.Linear(d, d), nn.LayerNorm(d)

    def forward(s, h, e, adj):
        B, M, _ = h.shape
        hs = s.Ws(h).view(B, 1, M, s.h, s.dh)                    # source j        [B,1,M,H,dh]
        hd = s.Wd(h).view(B, M, 1, s.h, s.dh)                    # destination i   [B,M,1,H,dh]
        he = s.We(e).view(B, M, M, s.h, s.dh)                    # edge term       [B,M,M,H,dh]
        sc = (F.leaky_relu(hs + hd + he, 0.2) * s.a).sum(-1)     # attention logits [B,M,M,H]
        sc = sc.masked_fill(~adj[..., None], -1e9)
        att = torch.softmax(sc, dim=2) * adj[..., None]          # normalise over sources j; zero where no edge
        msg = (att[..., None] * hs).sum(2).reshape(B, M, -1)     # [B,M,d]
        return s.norm(h + F.elu(s.out(msg)))                     # residual + norm


class MotionOnsetGNN(nn.Module):
    def __init__(s, d=64, heads=4, k=5, radius=10.0, recurrent=True):
        super().__init__()
        s.d, s.k, s.radius, s.recurrent = d, k, radius, recurrent
        s.node = nn.Linear(5, d)
        s.sem = nn.Linear(D, d)                                  # 384-d DINOv2 in the paper -> 64-d; here a 32-d synthetic embedding
        s.hin = nn.Linear(d, d, bias=False)                      # feed memory into the message passing (my design choice)
        s.gnn = nn.ModuleList([GATv2(d, heads), GATv2(d, heads)])
        s.gru = nn.GRUCell(d, d)
        s.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def graph(s, pos, vel, vis):
        """pos,vel [B,N,2], vis [B,N] -> edge feats [B,M,M,5], adjacency [B,M,M] (adj[b,i,j]: j sends to i)."""
        B, N, _ = pos.shape; M = N + 1
        dpos = pos[:, None] - pos[:, :, None]                    # [B,i,j,2] = pos_j - pos_i
        dist = dpos.norm(dim=-1)
        ok = vis[:, :, None] & vis[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=pos.device)
        dist = dist.masked_fill(~ok, 1e9)
        kk = min(s.k, N - 1)
        kth = dist.topk(kk, dim=-1, largest=False).values[..., -1:]          # distance of the k-th nearest neighbour
        adj_o = ok & (dist <= kth) & (dist < s.radius)                        # kNN within radius [B,N,N]
        adj_o = adj_o | torch.eye(N, dtype=torch.bool, device=pos.device)[None] & vis[:, :, None]   # self loops for visible
        adj = torch.zeros(B, M, M, dtype=torch.bool, device=pos.device)
        adj[:, :N, :N] = adj_o
        adj[:, :N, N] = vis                                                   # scene -> object
        adj[:, N, :N] = vis                                                   # object -> scene
        adj[:, N, N] = True
        e = torch.zeros(B, M, M, 5, device=pos.device)
        e[:, :N, :N, :2] = dpos / dist.clamp(min=1e-3, max=1e8)[..., None].clamp(min=1e-3) * (dist < 1e8)[..., None]   # unit direction
        e[:, :N, :N, 2:4] = vel[:, None] - vel[:, :, None]                    # relative velocity
        e[:, :, N, 4] = 1; e[:, N, :, 4] = 1                                  # scene-edge flag
        return e, adj

    def forward(s, pos, vel, vis, sem, h=None):
        """One frame. pos,vel [B,N,2]; vis [B,N]; sem [B,N,D]; h [B,M,d] -> (prob [B,N], h_new [B,M,d])."""
        B, N, _ = pos.shape; M = N + 1
        if h is None or not s.recurrent: h = pos.new_zeros(B, M, s.d)
        speed_ok = vis.float()
        x = torch.zeros(B, M, 5, device=pos.device)
        x[:, :N, 0:2] = (pos - pos.new_tensor([20., 7.])) / 20.              # normalised position
        x[:, :N, 2:4] = vel; x[:, :N, 4] = 0
        x[:, N, 4] = 1                                                       # is_scene
        x[:, :N] = x[:, :N] * speed_ok[..., None]
        semm = torch.cat([sem * vis[..., None], sem.new_zeros(B, 1, D)], 1)  # scene node has no semantics
        g = s.node(x) + s.sem(semm) + s.hin(h)                               # [B,M,d]
        e, adj = s.graph(pos, vel, vis)
        for layer in s.gnn: g = layer(g, e, adj)
        h_new = s.gru(g.reshape(B * M, -1), h.reshape(B * M, -1)).view(B, M, -1)
        keep = torch.cat([vis, vis.new_ones(B, 1)], 1)[..., None]
        h_new = h_new * keep                                                 # invisible nodes carry no state
        return torch.sigmoid(s.head(h_new[:, :N]).squeeze(-1)), h_new

    def rollout(s, pos, vel, vis, sem, tbptt=24):
        """pos,vel [B,T,N,2], vis [B,T,N] -> probs [B,T,N]. Truncated BPTT: detach memory every `tbptt` frames (paper: 20)."""
        h, out = None, []
        for t in range(pos.shape[1]):
            p, h = s(pos[:, t], vel[:, t], vis[:, t], sem, h)
            out.append(p)
            if s.training and (t + 1) % tbptt == 0: h = h.detach()
        return torch.stack(out, 1)


def masked_bce(p, y, q):
    """Binary cross-entropy over the query set only (static, visible, velocity observed). Proper scoring rule -> calibrated."""
    p = p.clamp(1e-5, 1 - 1e-5)
    l = -(y * p.log() + (1 - y) * (1 - p).log())
    return (l * q).sum() / q.sum().clamp(min=1)
