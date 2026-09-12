"""
MC-DeTra -- core architecture (reference re-implementation)
Reference implementation inspired by:
  "MC-DeTra: Motion-Consistent Joint Object Detection and Socially-Aware
   Trajectory Forecasting in Bird's-Eye-View Images" (arXiv:2609.11717,
   submitted Sept 10 2026), built on the DeTra backbone
   (Agro et al., ECCV 2024, arXiv:2406.04426).

RECONSTRUCTION NOTE: this daily-review session's ORIGINAL package (the same
day this file was first logged) executed a real forward+backward pass and
recorded exact tensor shapes and a parameter count in the project log --
but ran in a separate, isolated container this later session cannot access.
Only the `MotionConsistencyLosses` module (the paper's actual novel
contribution) survived verbatim in the log; everything else here
(`RefinementBlock`, `FactoredSelfAttention`, `DeformableLiDARCrossAttention`,
`KNNMapCrossAttention`, `MCDeTra`) is a fresh reconstruction of the
surrounding DeTra-style backbone, written from the architecture description
in the log and Waabi's public description of DeTra -- not the authors' own
code, and not a byte-for-byte replay of the earlier package.

Why this design exists (from the paper's framing):
  DeTra unifies 3D object detection and multi-modal trajectory forecasting
  in one shared [Object x Time x Mode] query volume: the t=0 slice reads out
  as detection, later time slices read out as forecast. But DeTra trains
  that shared representation purely from the future-trajectory loss --
  nothing forces it to actually encode true past kinematics or nearby
  traffic, so fast/crowded-actor forecasts can drift and predicted heading
  can disagree with the predicted direction of travel. MC-DeTra adds three
  TRAIN-ONLY auxiliary objectives on the exact same shared query that are
  fully removed at inference (zero added latency): past-motion regression,
  social-occupancy prediction, and heading/velocity consistency.

Pieces implemented here:
  1. InitDetector                    -- seeds the initial [N, T, K] query
                                         volume from pooled LiDAR BEV
                                         features + learned per-object,
                                         per-time, per-mode embeddings.
  2. FactoredSelfAttention            -- self-attention over the query
                                         volume factored into three cheap
                                         passes (over objects, over modes,
                                         over time) instead of one expensive
                                         joint attention over N*T*K tokens.
  3. DeformableLiDARCrossAttention    -- each query predicts a 2D BEV offset
                                         and bilinearly samples the LiDAR
                                         feature map there (single-point
                                         deformable attention).
  4. KNNMapCrossAttention             -- each query attends only to its
                                         top-k nearest map-polyline tokens
                                         by learned similarity, instead of
                                         the full map token set.
  5. RefinementBlock                  -- one block combining all of the
                                         above + a feed-forward network,
                                         stacked B times.
  6. MCDeTra                          -- top-level model: Init Detector ->
                                         B refinement blocks -> detection
                                         head (t=0) + forecast heads (t>0).
  7. MotionConsistencyLosses          -- the paper's actual contribution
                                         (verbatim from the original log).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MCDeTraConfig:
    d_model: int = 64
    n_objects: int = 8            # N: number of object queries
    n_future_steps: int = 4       # T: total time steps, t=0 (detection) .. T-1 (forecast horizon)
    n_modes: int = 3              # K: number of forecast modes (multi-modal futures)
    n_blocks: int = 2             # B: number of RefinementBlocks
    n_heads: int = 4
    lidar_channels: int = 32      # C: LiDAR BEV feature map channel width
    map_feat_dim: int = 32        # D_map: per map-polyline-token feature width
    occupancy_grid: int = 5       # g: side length of the local occupancy patch (social-context aux target)
    mlp_ratio: float = 4.0
    knn_k: int = 4                # number of nearest map tokens attended to per query


# --------------------------------------------------------------------------- #
# 1. Init Detector -- seeds the initial [B, N, T, K, D] query volume
# --------------------------------------------------------------------------- #
class InitDetector(nn.Module):
    """
    Pools the LiDAR BEV feature map into a global context vector, combines
    it with N learned per-object query embeddings, then broadcasts across
    T time steps and K modes with learned positional embeddings -- giving
    every (object, time, mode) triple a distinct initial token before
    refinement.
    """

    def __init__(self, cfg: MCDeTraConfig):
        super().__init__()
        self.cfg = cfg
        self.lidar_pool_proj = nn.Linear(cfg.lidar_channels, cfg.d_model)
        self.object_queries = nn.Parameter(torch.zeros(1, cfg.n_objects, cfg.d_model))
        self.time_embed = nn.Parameter(torch.zeros(1, 1, cfg.n_future_steps, 1, cfg.d_model))
        self.mode_embed = nn.Parameter(torch.zeros(1, 1, 1, cfg.n_modes, cfg.d_model))
        nn.init.trunc_normal_(self.object_queries, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        nn.init.trunc_normal_(self.mode_embed, std=0.02)

    def forward(self, lidar_bev: torch.Tensor) -> torch.Tensor:
        """
        lidar_bev: [B, C, H, W]  LiDAR BEV feature map
        Returns:   [B, N, T, K, D]  initial query volume
        """
        b = lidar_bev.shape[0]
        cfg = self.cfg

        global_ctx = lidar_bev.mean(dim=(-1, -2))                      # [B, C]  global average pool
        global_ctx = self.lidar_pool_proj(global_ctx)                  # [B, D]

        obj_q = self.object_queries.expand(b, -1, -1) + global_ctx.unsqueeze(1)  # [B, N, D]
        obj_q = obj_q.view(b, cfg.n_objects, 1, 1, cfg.d_model)                    # [B, N, 1, 1, D]

        query_volume = obj_q + self.time_embed + self.mode_embed        # broadcast -> [B, N, T, K, D]
        return query_volume


# --------------------------------------------------------------------------- #
# 2. Factored self-attention over [Object, Time, Mode]
# --------------------------------------------------------------------------- #
class FactoredSelfAttention(nn.Module):
    """
    Instead of one expensive joint self-attention over all N*T*K tokens,
    runs three cheap self-attention passes in sequence: over the Object
    axis (with Time/Mode folded into batch), over the Mode axis, and over
    the Time axis -- letting information mix along each axis independently
    while keeping attention cost linear in N*T*K rather than quadratic in
    it.
    """

    def __init__(self, cfg: MCDeTraConfig):
        super().__init__()
        self.cfg = cfg
        self.obj_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.mode_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.time_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.norm_obj = nn.LayerNorm(cfg.d_model)
        self.norm_mode = nn.LayerNorm(cfg.d_model)
        self.norm_time = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T, K, D]
        b, n, t, k, d = x.shape

        # ---- attend over Object axis ----
        h = x.permute(0, 2, 3, 1, 4).reshape(b * t * k, n, d)           # [B*T*K, N, D]
        attn_out, _ = self.obj_attn(h, h, h, need_weights=False)
        h = self.norm_obj(h + attn_out)
        x = h.view(b, t, k, n, d).permute(0, 3, 1, 2, 4)                # [B, N, T, K, D]

        # ---- attend over Mode axis ----
        h = x.permute(0, 1, 2, 3, 4).reshape(b * n * t, k, d)           # [B*N*T, K, D]
        attn_out, _ = self.mode_attn(h, h, h, need_weights=False)
        h = self.norm_mode(h + attn_out)
        x = h.view(b, n, t, k, d)

        # ---- attend over Time axis ----
        h = x.permute(0, 1, 3, 2, 4).reshape(b * n * k, t, d)           # [B*N*K, T, D]
        attn_out, _ = self.time_attn(h, h, h, need_weights=False)
        h = self.norm_time(h + attn_out)
        x = h.view(b, n, k, t, d).permute(0, 1, 3, 2, 4)                # [B, N, T, K, D]

        return x


# --------------------------------------------------------------------------- #
# 3. Deformable LiDAR cross-attention (single-point deformable sampling)
# --------------------------------------------------------------------------- #
class DeformableLiDARCrossAttention(nn.Module):
    """
    Each (object, time, mode) query predicts its own 2D offset into
    normalized BEV coordinates and bilinearly samples the LiDAR feature map
    there -- a simplified, single-sample-point version of deformable
    attention (Deformable DETR-style), letting each query pull in exactly
    the LiDAR evidence relevant to its own hypothesized location instead of
    attending densely over the whole feature map.
    """

    def __init__(self, cfg: MCDeTraConfig):
        super().__init__()
        self.offset_head = nn.Linear(cfg.d_model, 2)          # predicts (dx, dy) in [-1, 1] BEV coords
        self.value_proj = nn.Linear(cfg.lidar_channels, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        """
        x:         [B, N, T, K, D]
        lidar_bev: [B, C, H, W]
        Returns:   [B, N, T, K, D]  (residual update, added by the caller)
        """
        b, n, t, k, d = x.shape
        flat = x.reshape(b, n * t * k, d)                                   # [B, L, D]  L = N*T*K

        offsets = torch.tanh(self.offset_head(flat))                     # [B, L, 2] in [-1, 1]
        grid = offsets.unsqueeze(2)                                      # [B, L, 1, 2]  grid_sample wants [B,Hg,Wg,2]

        sampled = F.grid_sample(lidar_bev, grid, mode="bilinear", align_corners=True)  # [B, C, L, 1]
        sampled = sampled.squeeze(-1).transpose(1, 2)                     # [B, L, C]

        value = self.value_proj(sampled)                                  # [B, L, D]
        out = self.out_proj(value)                                        # [B, L, D]
        return out.reshape(b, n, t, k, d)


# --------------------------------------------------------------------------- #
# 4. k-NN map cross-attention
# --------------------------------------------------------------------------- #
class KNNMapCrossAttention(nn.Module):
    """
    Each query computes similarity to all map-polyline tokens, restricts
    attention to only the top-k most similar (k-NN in learned embedding
    space rather than raw Euclidean map distance, for simplicity), and
    attends over just those -- cheaper than dense cross-attention to every
    map token and closer to how a driver only reasons about nearby lanes.
    """

    def __init__(self, cfg: MCDeTraConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.map_feat_dim, cfg.d_model)
        self.v_proj = nn.Linear(cfg.map_feat_dim, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor, map_tokens: torch.Tensor) -> torch.Tensor:
        """
        x:          [B, N, T, K, D]
        map_tokens: [B, M, map_feat_dim]
        Returns:    [B, N, T, K, D]  (residual update, added by the caller)
        """
        b, n, t, k, d = x.shape
        m = map_tokens.shape[1]
        knn_k = min(self.cfg.knn_k, m)

        flat_q = self.q_proj(x.reshape(b, n * t * k, d))                    # [B, L, D]
        map_k = self.k_proj(map_tokens)                                   # [B, M, D]
        map_v = self.v_proj(map_tokens)                                   # [B, M, D]

        sim = torch.matmul(flat_q, map_k.transpose(-1, -2)) / math.sqrt(d)  # [B, L, M]
        topk_sim, topk_idx = sim.topk(knn_k, dim=-1)                        # [B, L, knn_k]

        attn = F.softmax(topk_sim, dim=-1)                                  # [B, L, knn_k]
        gathered_v = torch.gather(
            map_v.unsqueeze(1).expand(-1, flat_q.shape[1], -1, -1),         # [B, L, M, D]
            dim=2,
            index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, d),             # [B, L, knn_k, D]
        )                                                                    # [B, L, knn_k, D]

        out = (attn.unsqueeze(-1) * gathered_v).sum(dim=2)                  # [B, L, D]
        out = self.out_proj(out)
        return out.reshape(b, n, t, k, d)


# --------------------------------------------------------------------------- #
# 5. One Refinement Block
# --------------------------------------------------------------------------- #
class RefinementBlock(nn.Module):
    def __init__(self, cfg: MCDeTraConfig):
        super().__init__()
        self.factored_self_attn = FactoredSelfAttention(cfg)
        self.lidar_cross_attn = DeformableLiDARCrossAttention(cfg)
        self.map_cross_attn = KNNMapCrossAttention(cfg)
        self.norm_lidar = nn.LayerNorm(cfg.d_model)
        self.norm_map = nn.LayerNorm(cfg.d_model)

        hidden = int(cfg.d_model * cfg.mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden), nn.GELU(), nn.Linear(hidden, cfg.d_model)
        )
        self.norm_ffn = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, lidar_bev: torch.Tensor, map_tokens: torch.Tensor) -> torch.Tensor:
        x = self.factored_self_attn(x)                                     # [B, N, T, K, D]
        x = self.norm_lidar(x + self.lidar_cross_attn(x, lidar_bev))
        x = self.norm_map(x + self.map_cross_attn(x, map_tokens))
        x = self.norm_ffn(x + self.ffn(x))
        return x


# --------------------------------------------------------------------------- #
# 6. Full MC-DeTra model
# --------------------------------------------------------------------------- #
class MCDeTra(nn.Module):
    """
    LiDAR + HD-map encoders (external, not implemented here -- this module
    consumes their output) -> InitDetector seeds an [N, T, K] query volume
    -> B RefinementBlocks -> Detection head (t=0 slice) / Forecast heads
    (t>0 slices).
    """

    def __init__(self, cfg: MCDeTraConfig):
        super().__init__()
        self.cfg = cfg
        self.init_detector = InitDetector(cfg)
        self.blocks = nn.ModuleList([RefinementBlock(cfg) for _ in range(cfg.n_blocks)])

        # Detection head: reads the t=0 slice (pooled over mode) -> box params.
        self.detection_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 5),   # (x, y, length, width, heading)
        )
        # Forecast heads: applied to the full [N, T, K] query volume.
        self.offset_head = nn.Linear(cfg.d_model, 2)      # per-step (dx, dy) relative to the previous step
        self.heading_head = nn.Linear(cfg.d_model, 1)     # per-step absolute heading (radians)

    def forward(self, lidar_bev: torch.Tensor, map_tokens: torch.Tensor):
        """
        lidar_bev:  [B, C, H, W]
        map_tokens: [B, M, map_feat_dim]

        Returns dict of:
            detection_boxes:  [B, N, 5]           (x, y, length, width, heading) at t=0
            forecast_offsets: [B, N, T-1, K, 2]    per-step (dx, dy), t=1..T-1
            headings:         [B, N, T, K]         heading at every step, t=0..T-1
            positions:        [B, N, T, K, 2]      cumulative (x, y), t=0 anchored at the detection center
            t0_query:         [B, N, D]            mode-pooled t=0 query (fed to MotionConsistencyLosses)
        """
        cfg = self.cfg
        query_volume = self.init_detector(lidar_bev)                        # [B, N, T, K, D]

        for block in self.blocks:
            query_volume = block(query_volume, lidar_bev, map_tokens)

        t0_query = query_volume[:, :, 0, :, :].mean(dim=2)                  # [B, N, D]  mode-pooled t=0
        detection_boxes = self.detection_head(t0_query)                    # [B, N, 5]

        headings = self.heading_head(query_volume).squeeze(-1)              # [B, N, T, K]
        offsets_all = self.offset_head(query_volume)                        # [B, N, T, K, 2]
        forecast_offsets = offsets_all[:, :, 1:, :, :]                      # [B, N, T-1, K, 2]  t>0 only

        # Cumulative positions: t=0 anchored at the detection center (x, y);
        # each later step adds its predicted offset on top of the previous one.
        t0_pos = detection_boxes[..., :2].unsqueeze(2).unsqueeze(2)         # [B, N, 1, 1, 2]
        t0_pos = t0_pos.expand(-1, -1, 1, cfg.n_modes, -1)                  # [B, N, 1, K, 2]
        cum_offsets = torch.cumsum(forecast_offsets, dim=2)                  # [B, N, T-1, K, 2]
        future_pos = t0_pos + cum_offsets                                    # [B, N, T-1, K, 2]
        positions = torch.cat([t0_pos, future_pos], dim=2)                   # [B, N, T, K, 2]

        return {
            "detection_boxes": detection_boxes,
            "forecast_offsets": forecast_offsets,
            "headings": headings,
            "positions": positions,
            "t0_query": t0_query,
        }


# --------------------------------------------------------------------------- #
# 7. MotionConsistencyLosses -- the paper's actual contribution
#    (verbatim from the daily-review project log)
# --------------------------------------------------------------------------- #
class MotionConsistencyLosses(nn.Module):
    """MC-DeTra's three train-only auxiliary objectives, attached to the SAME
    shared query volume the base DeTra detection/forecasting heads read
    from. None of these are invoked at inference time -> zero added latency
    in production; they exist purely to shape gradients into the shared
    backbone during training."""

    def __init__(self, cfg: MCDeTraConfig):
        super().__init__()
        self.cfg = cfg
        # (1) Past-motion aux head: regress the actor's OWN observed past
        #     trajectory (not the future) from its t=0 query.
        self.past_motion_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.n_future_steps * 2),  # (dx, dy) per past step
        )
        # (2) Social-context occupancy head: predict a local BEV occupancy
        #     patch around each actor (the "who/what is nearby" signal).
        g = cfg.occupancy_grid
        self.occupancy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, g * g),
        )
        # (3) Heading<->velocity consistency has no learned parameters -- it's
        #     a geometric penalty computed directly from the forecast outputs.

    @staticmethod
    def heading_velocity_consistency_loss(pred_heading, pred_positions, valid_mask, eps=1e-4):
        # pred_heading:   [B, N_obj, T, K]     predicted heading angle (radians)
        # pred_positions: [B, N_obj, T, K, 2]  predicted (x, y) waypoints
        vel = pred_positions[:, :, 1:] - pred_positions[:, :, :-1]   # [B, N, T-1, K, 2]
        speed = vel.norm(dim=-1)                                    # [B, N, T-1, K]
        implied_heading = torch.atan2(vel[..., 1], vel[..., 0])     # direction of travel

        heading_pred = pred_heading[:, :, 1:]  # align with velocity step t-1 -> t
        angular_err = 1.0 - torch.cos(heading_pred - implied_heading)  # wrap-safe

        weight = (speed > eps).float() * valid_mask[:, :, 1:]  # skip near-stationary steps
        return (angular_err * weight).sum() / weight.sum().clamp_min(1)

    def forward(self, t0_query, pred_heading, pred_positions,
                past_motion_gt, occupancy_gt, valid_mask_obj, valid_mask_traj,
                loss_weights=None):
        w = loss_weights or {"past_motion": 1.0, "occupancy": 1.0, "heading_vel": 1.0}

        pred_past = self.past_motion_head(t0_query).view(
            *t0_query.shape[:2], self.cfg.n_future_steps, 2)
        l_past = (F.smooth_l1_loss(pred_past, past_motion_gt, reduction="none")
                  .sum(dim=(-1, -2)) * valid_mask_obj).sum() / valid_mask_obj.sum().clamp_min(1)

        g = self.cfg.occupancy_grid
        logits = self.occupancy_head(t0_query).view(*t0_query.shape[:2], g, g)
        l_occ = (F.binary_cross_entropy_with_logits(logits, occupancy_gt, reduction="none")
                 .mean(dim=(-1, -2)) * valid_mask_obj).sum() / valid_mask_obj.sum().clamp_min(1)

        l_hv = self.heading_velocity_consistency_loss(pred_heading, pred_positions, valid_mask_traj)

        total = w["past_motion"] * l_past + w["occupancy"] * l_occ + w["heading_vel"] * l_hv
        return {"loss_past_motion": l_past, "loss_occupancy": l_occ,
                "loss_heading_velocity": l_hv, "loss_mc_total": total}
