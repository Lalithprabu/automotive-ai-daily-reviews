"""
Sparse-BEVNet attention modules.

Reconstructed from arXiv:2609.14185 (Sparse-BEVNet: Bi-Level Routing and
Sparse Spatial Attention based Multi-View BEV 3D Object Detection). The
paper's own published detail is thin (short 4-page CISAT 2026 paper; only
the abstract, headline metrics and author list were recoverable via an
alphaXiv mirror after arXiv rate-limited direct fetches). Internal
dimensions, exact routing formulas and hyperparameters below are this
project's own reconstruction defaults, not paper-sourced numbers.

Three modules, each implementing a distinct sparsity/routing idea used
in the paper's description:

1. BiLevelRoutingAttention (BRA)   - coarse region-level routing BEFORE
   fine-grained token attention, applied inside the camera backbone.
2. CascadedGroupAttention (CGA)    - cascades partial outputs across
   attention heads/groups for "free" head diversity, used in the fusion
   / detection head.
3. SparseSpatialCrossAttention     - the BEV lifting / view-transform
   step: each BEV grid cell gates itself to only its top-k most relevant
   camera views instead of dense sampling across all views.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLevelRoutingAttention(nn.Module):
    """Two-level (region -> token) routed attention over a (B, C, H, W)
    feature map.

    Level 1 (coarse routing): the feature map is partitioned into a
    ``region_grid x region_grid`` set of spatial regions. Each region is
    mean-pooled into a single region token, and a lightweight
    region-to-region affinity (routing) score decides, for every query
    region, which ``topk`` regions in the whole map are worth attending to.

    Level 2 (fine attention): standard multi-head scaled dot-product
    attention is then computed only between a query region's tokens and
    the tokens living inside the top-k routed regions -- never the full
    H*W token grid. This is what makes the mechanism sparse: routing
    happens BEFORE the expensive token-level attention, not as a mask
    applied after it.
    """

    def __init__(self, dim: int, region_grid: int = 2, topk: int = 2, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.region_grid = region_grid
        self.topk = topk
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Fine-grained token projections.
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Coarse region-routing projections (small, cheap: operate on
        # num_regions tokens only).
        self.region_q = nn.Linear(dim, dim)
        self.region_k = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        rg = self.region_grid
        assert H % rg == 0 and W % rg == 0, "H, W must be divisible by region_grid"
        rh, rw = H // rg, W // rg
        num_regions = rg * rg
        tokens_per_region = rh * rw

        # (B, C, H, W) -> (B, num_regions, tokens_per_region, C)
        x_ = x.view(B, C, rg, rh, rg, rw)
        x_ = x_.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, rg, rg, rh, rw, C
        x_ = x_.view(B, num_regions, tokens_per_region, C)

        # ---- Level 1: coarse region routing ----
        region_tokens = x_.mean(dim=2)  # (B, num_regions, C)
        rq = self.region_q(region_tokens)
        rk = self.region_k(region_tokens)
        routing_scores = torch.matmul(rq, rk.transpose(-1, -2)) / (C ** 0.5)  # (B, R, R)
        topk = min(self.topk, num_regions)
        topk_idx = routing_scores.topk(topk, dim=-1).indices  # (B, R, topk)

        # ---- Level 2: fine token attention, restricted to routed regions ----
        # Vectorized over all query regions at once (all B*num_regions "mini
        # attentions" computed as one batched matmul, instead of a Python
        # loop over regions) -- functionally identical to looping over
        # query regions one at a time, just without the per-region overhead.
        Q = num_regions
        x_src = x_.unsqueeze(1).expand(B, Q, num_regions, tokens_per_region, C)  # (B, Q, R, T, C)
        idx_b = topk_idx.view(B, Q, topk, 1, 1).expand(-1, -1, -1, tokens_per_region, C)  # (B, Q, topk, T, C)
        kv_tokens = torch.gather(x_src, 2, idx_b)  # (B, Q, topk, T, C)
        kv_tokens = kv_tokens.reshape(B, Q, topk * tokens_per_region, C)

        q_tokens = x_  # (B, Q, T, C) -- every region is a query region

        q = self.q_proj(q_tokens)  # (B, Q, T, C)
        k = self.k_proj(kv_tokens)  # (B, Q, topk*T, C)
        v = self.v_proj(kv_tokens)

        BQ = B * Q
        q = q.reshape(BQ, tokens_per_region, C)
        k = k.reshape(BQ, topk * tokens_per_region, C)
        v = v.reshape(BQ, topk * tokens_per_region, C)

        q = q.view(BQ, tokens_per_region, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(BQ, topk * tokens_per_region, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(BQ, topk * tokens_per_region, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (BQ, heads, T, topk*T)
        attn = attn.softmax(dim=-1)
        o = torch.matmul(attn, v)  # (BQ, heads, T, head_dim)
        o = o.transpose(1, 2).reshape(BQ, tokens_per_region, C)
        out = self.out_proj(o)  # (BQ, T, C)
        out = out.view(B, Q, tokens_per_region, C)

        # Fold regions back into the (B, C, H, W) grid.
        out = out.view(B, rg, rg, rh, rw, C)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(B, C, H, W)
        return out

    @torch.no_grad()
    def region_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Utility for visualization only: returns a (B, region_grid,
        region_grid) heatmap of how much each region's routed output
        "lit up" (mean squared activation), used to render the BRA
        routing-glow overlay in simulate.py. Not used during training."""
        out = self.forward(x)
        B, C, H, W = out.shape
        rg = self.region_grid
        rh, rw = H // rg, W // rg
        energy = out.view(B, C, rg, rh, rg, rw).pow(2).mean(dim=(1, 3, 5))
        return energy


class CascadedGroupAttention(nn.Module):
    """Cascaded multi-group self-attention over a (B, N, C) token sequence.

    The channel dimension is split into ``num_groups`` equal slices, one
    per "head". Instead of every head attending independently on its own
    slice (which tends to produce redundant heads), each group's input is
    the sum of its own channel slice AND the *previous* group's output.
    This cascading is what the paper calls getting attention diversity
    "for free": later heads are conditioned on what earlier heads already
    found, without adding extra parameters beyond one attention block per
    group.
    """

    def __init__(self, dim: int, num_groups: int = 4):
        super().__init__()
        assert dim % num_groups == 0, "dim must be divisible by num_groups"
        self.dim = dim
        self.num_groups = num_groups
        self.group_dim = dim // num_groups

        self.q_projs = nn.ModuleList([nn.Linear(self.group_dim, self.group_dim) for _ in range(num_groups)])
        self.k_projs = nn.ModuleList([nn.Linear(self.group_dim, self.group_dim) for _ in range(num_groups)])
        self.v_projs = nn.ModuleList([nn.Linear(self.group_dim, self.group_dim) for _ in range(num_groups)])
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        splits = torch.split(x, self.group_dim, dim=-1)  # num_groups tensors of (B, N, group_dim)
        outputs = []
        cascade_input = None
        for g in range(self.num_groups):
            xg = splits[g]
            if cascade_input is not None:
                xg = xg + cascade_input  # cascade: feed previous group's output forward
            q = self.q_projs[g](xg)
            k = self.k_projs[g](xg)
            v = self.v_projs[g](xg)
            attn = torch.matmul(q, k.transpose(-1, -2)) / (self.group_dim ** 0.5)
            attn = attn.softmax(dim=-1)
            og = torch.matmul(attn, v)
            outputs.append(og)
            cascade_input = og
        out = torch.cat(outputs, dim=-1)
        return self.out_proj(out)


class SparseSpatialCrossAttention(nn.Module):
    """Sparse BEV <- camera-view cross attention (the view-transform / BEV
    lifting step).

    Every BEV grid cell is a query. Instead of densely sampling reference
    points across *every* one of the ``num_cameras`` views (the
    BEVFormer-style baseline), each query first scores its relevance to
    each camera view (via a cheap dot product against a per-camera token
    summary), keeps only the ``topk_cameras`` most relevant views, and
    restricts its fine-grained cross-attention to tokens from those views
    only.
    """

    def __init__(self, dim: int, num_cameras: int = 6, topk_cameras: int = 2, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_cameras = num_cameras
        self.topk_cameras = topk_cameras
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, bev_queries: torch.Tensor, camera_tokens: torch.Tensor, tokens_per_camera: int):
        """
        bev_queries:   (B, Nq, C)  -- one query per BEV grid cell
        camera_tokens: (B, num_cameras * tokens_per_camera, C) -- flattened
                       backbone feature tokens from all camera views
        Returns:
            fused:        (B, Nq, C)
            cam_gate_avg: (B, num_cameras) -- softmax relevance of each
                          camera view, averaged over all BEV queries
                          (used for the live per-camera attention-gate bar
                          chart in the simulation).
        """
        B, Nq, C = bev_queries.shape
        Ncam, Ntok = self.num_cameras, tokens_per_camera

        q = self.q_proj(bev_queries)
        k = self.k_proj(camera_tokens)
        v = self.v_proj(camera_tokens)

        # Per-camera relevance score: query vs. mean-pooled token summary of that camera.
        k_cam = k.view(B, Ncam, Ntok, C)
        cam_summary = k_cam.mean(dim=2)  # (B, Ncam, C)
        gate_logits = torch.matmul(q, cam_summary.transpose(-1, -2)) / (C ** 0.5)  # (B, Nq, Ncam)
        gate_weights_full = gate_logits.softmax(dim=-1)  # (B, Nq, Ncam)

        topk = min(self.topk_cameras, Ncam)
        topk_idx = gate_logits.topk(topk, dim=-1).indices  # (B, Nq, topk)

        cam_mask = torch.zeros(B, Nq, Ncam, device=bev_queries.device, dtype=torch.bool)
        cam_mask.scatter_(-1, topk_idx, True)
        token_mask = cam_mask.unsqueeze(-1).expand(-1, -1, -1, Ntok).reshape(B, Nq, Ncam * Ntok)

        qh = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        kh = k.view(B, Ncam * Ntok, self.num_heads, self.head_dim).transpose(1, 2)
        vh = v.view(B, Ncam * Ntok, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(qh, kh.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, heads, Nq, Ncam*Ntok)
        mask = token_mask.unsqueeze(1)  # (B, 1, Nq, Ncam*Ntok)
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, vh)  # (B, heads, Nq, head_dim)
        out = out.transpose(1, 2).reshape(B, Nq, C)
        out = self.out_proj(out)

        cam_gate_avg = gate_weights_full.mean(dim=1)  # (B, Ncam)
        return out, cam_gate_avg
