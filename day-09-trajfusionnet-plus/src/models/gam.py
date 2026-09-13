"""
Graph Attention Module (GAM) — the core novel contribution of TrajFusionNet+
(arXiv:2609.10806, Landry & Akhloufi, submitted Sept 9, 2026).

TrajFusionNet+ extends the original two-branch TrajFusionNet
(arXiv:2508.19866 — Sequence Attention Module + Visual Attention Module)
with a third branch: a pedestrian-centric SCENE GRAPH built from a semantic
segmentation of the observation frame. The pedestrian is the hub node; every
other detected traffic element (vehicle, crosswalk, traffic light, curb,
sidewalk, road surface, ...) becomes a satellite node. Multi-head graph
attention layers propagate relational context ("is a vehicle approaching?",
"is there a marked crosswalk within reach?") into a single pedestrian-centric
embedding that is late-fused with SAM's trajectory encoding and VAM's visual
encoding to predict crossing intention.

SOURCING NOTE (good-faith reconstruction — read before treating this as a
byte-for-byte reproduction of the authors' code):
The TrajFusionNet+ abstract confirms the GAM's *role* ("extracts
pedestrian-centric graphs from segmented scene images and captures the
relational dependencies between pedestrians and traffic elements") but every
attempt to fetch the paper's full HTML/PDF text this cycle
(arxiv.org/html, arxiv.org/pdf, an ar5iv mirror not yet processed for this
very recent ID, and an alphaxiv.org PDF mirror blocked by robots.txt)
returned HTTP 429 or a dead end — the same recurring obstacle logged on
nearly every prior day of this series. The exact node taxonomy, edge
construction rule, GNN layer count, and attention formulation below are a
structural reconstruction following (a) standard Graph Attention Network
(Velickovic et al., 2018) mechanics and (b) the base TrajFusionNet's own
documented conventions (d_model=128, late-fusion-via-dense-layers) wherever
a direct analogue exists in its verified architecture. Treat hyperparameters
here as sensible defaults, not confirmed values from the paper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GAMConfig:
    node_feat_dim: int = 8       # raw per-node feature width, see NodeFeatureEncoder
    d_model: int = 128           # shared embedding width (matches SAM's d_model)
    num_heads: int = 4
    num_layers: int = 2
    proj_dim: int = 40           # output width fed into the late-fusion trunk
    dropout: float = 0.1
    max_nodes: int = 16          # pedestrian hub + up to 15 satellite traffic elements


class NodeFeatureEncoder(nn.Module):
    """Encodes raw per-node scalars into the shared d_model embedding space.

    Each node (the pedestrian hub, or a satellite traffic element extracted
    from the segmentation mask) is described by an 8-dim feature vector:
        [class_one_hot(4): {pedestrian, vehicle, crosswalk, other-static},
         centroid_x, centroid_y,           # normalized to [-1, 1] in image space
         area_ratio,                       # segment area / frame area
         dist_to_pedestrian]               # euclidean, normalized by frame diagonal
    """

    def __init__(self, cfg: GAMConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.node_feat_dim, cfg.d_model // 2),
            nn.LayerNorm(cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.d_model),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        # node_features: [B, N, node_feat_dim]  ->  [B, N, d_model]
        return self.mlp(node_features)


class GraphAttentionLayer(nn.Module):
    """A single multi-head graph attention layer with an additive relative-
    position edge bias, pre-norm residual connections, and a feed-forward
    block — the same "attention block" recipe used throughout TrajFusionNet's
    other two branches (SAM's transformer, VAM's large-kernel attention),
    kept consistent here for architectural coherence.

    Shapes:
        x:         [B, N, D]     node embeddings
        edge_bias: [B, N, N]     precomputed scalar bias per (i, j) pair,
                                  derived from relative pedestrian-centric
                                  geometry (see PedestrianCentricGAM.build_graph)
        adj_mask:  [B, N, N]     1.0 where an edge exists, 0.0 otherwise
                                  (padded/absent nodes are masked out)
    """

    def __init__(self, cfg: GAMConfig):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0, "d_model must divide num_heads"
        self.d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads

        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = nn.Dropout(cfg.dropout)

        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )
        self.ffn_dropout = nn.Dropout(cfg.dropout)

        # Learned scalar gate on the edge bias, so the model can decide how
        # much geometric prior to trust vs. pure content-based attention.
        self.edge_bias_gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, edge_bias: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm1(x)

        q = self.q_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,Dh]
        k = self.k_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,Dh]
        v = self.v_proj(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,Dh]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,N,N]
        scores = scores + self.edge_bias_gate * edge_bias.unsqueeze(1)  # broadcast over heads

        neg_inf = torch.finfo(scores.dtype).min
        mask = adj_mask.unsqueeze(1) > 0  # [B,1,N,N] -> broadcast to [B,H,N,N]
        scores = scores.masked_fill(~mask, neg_inf)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)  # isolated/padded nodes: all-masked row -> softmax(nan) -> 0
        attn = self.attn_dropout(attn)

        out = attn @ v                                    # [B,H,N,Dh]
        out = out.transpose(1, 2).reshape(B, N, D)         # [B,N,D]
        out = self.out_proj(out)
        x = x + out                                        # residual 1

        h2 = self.norm2(x)
        x = x + self.ffn_dropout(self.ffn(h2))              # residual 2 (pre-norm FFN)
        return x


class PedestrianCentricGAM(nn.Module):
    """Full Graph Attention Module: encodes nodes, stacks `num_layers` graph
    attention blocks, and reads out the pedestrian hub node's final embedding
    as the branch's contribution to late fusion.

    Node 0 is ALWAYS the pedestrian hub by construction (see build_graph).
    """

    def __init__(self, cfg: GAMConfig | None = None):
        super().__init__()
        self.cfg = cfg or GAMConfig()
        self.node_encoder = NodeFeatureEncoder(self.cfg)
        self.layers = nn.ModuleList(
            GraphAttentionLayer(self.cfg) for _ in range(self.cfg.num_layers)
        )
        self.readout_norm = nn.LayerNorm(self.cfg.d_model)
        self.proj = nn.Linear(self.cfg.d_model, self.cfg.proj_dim)

    @staticmethod
    def build_graph(node_positions: torch.Tensor, node_classes: torch.Tensor,
                     node_areas: torch.Tensor, valid_mask: torch.Tensor):
        """Builds the pedestrian-centric graph tensors from raw per-node
        scene attributes extracted upstream (e.g. from a segmentation mask +
        connected-components pass — out of scope for this module).

        Args:
            node_positions: [B, N, 2]  normalized (x, y) centroids, node 0 = pedestrian
            node_classes:   [B, N]     long class ids in {0:pedestrian,1:vehicle,
                                        2:crosswalk,3:other-static}
            node_areas:     [B, N]     segment area / frame area, in [0, 1]
            valid_mask:     [B, N]     1.0 for real nodes, 0.0 for padding

        Returns:
            node_features: [B, N, 8]   see NodeFeatureEncoder docstring
            edge_bias:     [B, N, N]   -distance (closer pairs attend more, before learning)
            adj_mask:      [B, N, N]   pedestrian-centric star graph + fully-connected
                                        satellites within a fixed radius, padding excluded
        """
        B, N, _ = node_positions.shape
        one_hot = F.one_hot(node_classes.clamp(min=0, max=3), num_classes=4).float()  # [B,N,4]

        ped_pos = node_positions[:, :1, :]                              # [B,1,2]
        dist_to_ped = torch.linalg.norm(node_positions - ped_pos, dim=-1)  # [B,N]
        dist_to_ped = dist_to_ped / math.sqrt(2.0)                       # normalize by frame diagonal

        node_features = torch.cat(
            [one_hot, node_positions, node_areas.unsqueeze(-1), dist_to_ped.unsqueeze(-1)],
            dim=-1,
        )  # [B, N, 4+2+1+1] = [B, N, 8]

        pairwise_dist = torch.cdist(node_positions, node_positions, p=2)  # [B,N,N]
        edge_bias = -pairwise_dist  # closer pairs get a higher pre-softmax bias

        radius = 0.6  # normalized-coordinate connectivity radius for satellite-satellite edges
        within_radius = (pairwise_dist <= radius).float()
        star_edges = torch.zeros_like(within_radius)
        star_edges[:, 0, :] = 1.0
        star_edges[:, :, 0] = 1.0  # pedestrian hub always connects to (and from) every node
        adj = torch.clamp(within_radius + star_edges, max=1.0)

        pad = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)  # [B,N,N], zero out any padded row/col
        adj_mask = adj * pad
        eye = torch.eye(N, device=adj_mask.device).unsqueeze(0)
        adj_mask = torch.clamp(adj_mask + eye * valid_mask.unsqueeze(1), max=1.0)  # self-loops for valid nodes

        return node_features, edge_bias, adj_mask

    def forward(self, node_features: torch.Tensor, edge_bias: torch.Tensor,
                adj_mask: torch.Tensor) -> torch.Tensor:
        # node_features: [B, N, 8], edge_bias/adj_mask: [B, N, N]
        x = self.node_encoder(node_features)  # [B, N, d_model]
        for layer in self.layers:
            x = layer(x, edge_bias, adj_mask)
        x = self.readout_norm(x)
        pedestrian_hub = x[:, 0, :]           # [B, d_model]  node 0 is always the pedestrian
        return self.proj(pedestrian_hub)      # [B, proj_dim]  ready for late fusion
