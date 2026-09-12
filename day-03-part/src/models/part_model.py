"""
PART -- Physics-Aware Radar Transformer (core architecture)
Reference implementation inspired by:
  "If It Moves, Radar Knows: A Physics-Aware Radar Transformer for
   Class-Agnostic Moving-Object Detection" (arXiv:2609.02289, Sept 2026)

This is an original re-implementation of the architecture *described* in the
paper's abstract/summary (module names, mechanisms, and reported metrics),
written for the daily-review series -- it is not the authors' own code.

Why this design exists (from the paper's framing):
  Closed-set 3D detectors miss rare moving objects outside their training
  taxonomy. Automotive radar gives a category-independent motion cue
  (Doppler velocity) that camera/LiDAR-only detectors don't have, and works
  through weather/illumination conditions that degrade vision. But radar
  returns are sparse and noisy, so PART sidesteps full 3D box regression and
  instead predicts, per query: (1) an existence confidence, (2) a
  representative surface point, and (3) a 2D ground-plane velocity.

Pieces implemented here:
  1. DopplerAwareQueryInit (DAQI)     -- turns raw radar returns into a set
                                          of input-dependent object-query
                                          seeds by clustering in a joint
                                          (position, velocity) space, instead
                                          of using N fixed learned queries.
  2. PhysicsGuidedCrossAttention (PGCA) -- cross-attention where the
                                          attention logits are biased by two
                                          physical consistency terms: how well
                                          a point's raw Doppler return matches
                                          the *radial* component of the
                                          query's hypothesized velocity, and
                                          the point's radar cross-section
                                          (RCS), which correlates with how
                                          reliable/large a reflector is.
  3. PARTDecoderLayer                  -- one Transformer block: self-attn
                                          over queries + PGCA cross-attn into
                                          radar points + FFN.
  4. PARTModel                         -- stacks L decoder layers and heads
                                          out existence / surface-point /
                                          velocity per query.
  5. UncertaintyAwareSupervision       -- training-time target assignment:
                                          randomly masks a fraction of GT
                                          objects and assigns soft existence
                                          targets to ambiguous (masked or
                                          low-IoU) query-GT matches, so the
                                          model isn't punished for tentative
                                          detections of ambiguous returns.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Doppler-Aware Query Initialization (DAQI)
# --------------------------------------------------------------------------- #
class DopplerAwareQueryInit(nn.Module):
    """
    Produces N_QUERIES input-dependent query seeds from a raw radar point
    cloud by clustering points in a joint (x, y, z, vx_radial) space, then
    encoding each cluster's summary statistics into a query embedding.

    This replaces DETR-style fixed learned queries (which assume a constant
    object count/location prior) with proposals that move with wherever the
    radar actually sees coherent, moving reflectors -- important in sparse
    scenes where most of the query budget would otherwise attend to empty
    space.

    Clustering is done with a lightweight, differentiable soft-assignment
    (Gaussian-kernel affinity + iterative mean-shift-style updates) rather
    than a hard non-differentiable clustering algorithm (e.g. DBSCAN), so
    gradients can flow back into the point encoder during training.
    """

    def __init__(self, point_feat_dim: int, model_dim: int, n_queries: int, n_iters: int = 3):
        super().__init__()
        self.n_queries = n_queries
        self.n_iters = n_iters
        self.model_dim = model_dim

        # Encode each raw radar point (x, y, z, doppler_v, rcs) into a
        # feature vector used both for clustering geometry and as the value
        # stream consumed later by PGCA.
        self.point_encoder = nn.Sequential(
            nn.Linear(5, point_feat_dim),   # 5 raw channels -> point_feat_dim
            nn.LayerNorm(point_feat_dim),
            nn.GELU(),
            nn.Linear(point_feat_dim, point_feat_dim),
        )

        # Clustering happens in a 4D "joint space" of (x, y, z, radial
        # doppler velocity) -- a small learned projection lets the network
        # re-weight how much position vs. velocity matters for grouping.
        self.joint_space_proj = nn.Linear(4, 4)

        # Cluster summary (mean joint-space coords + mean point feature)
        # -> a query embedding in model_dim.
        self.query_embed = nn.Sequential(
            nn.Linear(4 + point_feat_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

        # Learned initial cluster-center offsets (breaks symmetry across the
        # n_queries seeds before the first soft-assignment iteration).
        self.init_offsets = nn.Parameter(torch.randn(n_queries, 4) * 0.1)

    def forward(self, radar_points: torch.Tensor, point_mask: torch.Tensor):
        """
        radar_points: [B, P, 5]   raw (x, y, z, doppler_v, rcs) per point
        point_mask:   [B, P]      True where a point is valid (vs. padding)

        Returns:
            query_seeds:  [B, Q, model_dim]   initial query embeddings
            point_feats:  [B, P, point_feat_dim]  per-point features (for PGCA)
        """
        b, p, _ = radar_points.shape
        q = self.n_queries

        point_feats = self.point_encoder(radar_points)          # [B, P, Fp]

        joint = self.joint_space_proj(radar_points[..., :4])     # [B, P, 4]  (x,y,z,doppler_v)

        # Initialize Q cluster centers by seeding from evenly-spaced points
        # plus a learned offset (cheap, permutation-friendly init).
        seed_idx = torch.linspace(0, p - 1, steps=q, device=radar_points.device).long()  # [Q]
        centers = joint[:, seed_idx, :] + self.init_offsets.unsqueeze(0)                 # [B, Q, 4]

        neg_inf_mask = (~point_mask).float() * -1e4                                       # [B, P]

        # ---- iterative soft (differentiable) mean-shift-style clustering ----
        for _ in range(self.n_iters):
            # Pairwise squared distance in joint (position+velocity) space.
            dist2 = torch.cdist(centers, joint, p=2) ** 2               # [B, Q, P]
            affinity = -dist2 + neg_inf_mask.unsqueeze(1)                 # [B, Q, P]
            weights = F.softmax(affinity, dim=-1)                        # [B, Q, P], sums to 1 over P
            centers = torch.bmm(weights, joint)                          # [B, Q, 4] weighted mean

        # Final soft assignment also pools point_feats into each cluster,
        # giving every query seed both "where/how fast" (centers) and
        # "what it looks like" (pooled point feature) information.
        dist2 = torch.cdist(centers, joint, p=2) ** 2
        affinity = -dist2 + neg_inf_mask.unsqueeze(1)
        weights = F.softmax(affinity, dim=-1)                            # [B, Q, P]
        pooled_feats = torch.bmm(weights, point_feats)                   # [B, Q, Fp]

        query_seeds = self.query_embed(torch.cat([centers, pooled_feats], dim=-1))  # [B, Q, D]
        return query_seeds, point_feats                                   # [B,Q,D], [B,P,Fp]


# --------------------------------------------------------------------------- #
# 2. Physics-Guided Cross-Attention (PGCA)
# --------------------------------------------------------------------------- #
class PhysicsGuidedCrossAttention(nn.Module):
    """
    Cross-attention from object queries into raw radar points, where the
    standard dot-product attention logits are additively biased by two
    physically-grounded consistency terms:

      1. Radial-Doppler consistency: each query predicts a candidate 2D
         ground-plane velocity (vx, vy) per attention step; for a given
         radar point at azimuth angle theta, radar physics only measures the
         velocity component *along the line of sight* (the radial velocity).
         A point is a good match for a query if the point's own measured
         Doppler return agrees with the query's hypothesized velocity
         projected onto that point's radial direction:

             v_radial_hat = vx * cos(theta) + vy * sin(theta)
             consistency  = -| v_radial_hat - doppler_v_point |

      2. RCS reliability weighting: a point's radar cross-section (RCS)
         correlates with how strong/reliable a reflector it is. Points with
         higher (learned-normalized) RCS get an additive log-weight bonus,
         so the query attends more to confident returns and less to
         speckle/clutter.

    Both bias terms are learnable-scaled and added directly to the raw
    attention logits before softmax, exactly like a relative-position bias
    in other Transformer variants -- so the network can learn *how much* to
    trust radar physics vs. plain learned content-based attention.
    """

    def __init__(self, model_dim: int, point_feat_dim: int, n_heads: int):
        super().__init__()
        assert model_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(point_feat_dim, model_dim)
        self.v_proj = nn.Linear(point_feat_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        # Predicts a candidate (vx, vy) hypothesis per query, used only to
        # compute the radial-Doppler consistency bias (not the final output
        # velocity -- that comes from the dedicated velocity head).
        self.velocity_hypothesis = nn.Linear(model_dim, 2)

        # Learnable scalar weights for how strongly each physics term
        # influences the attention logits (initialized small and positive).
        self.doppler_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.rcs_bias_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        queries: torch.Tensor,          # [B, Q, D]
        point_feats: torch.Tensor,      # [B, P, Fp]
        point_xy: torch.Tensor,         # [B, P] radar point azimuth angle theta (radians)
        point_doppler: torch.Tensor,    # [B, P] raw measured radial Doppler velocity
        point_rcs: torch.Tensor,        # [B, P] radar cross-section (already log-scaled/normalized)
        point_mask: torch.Tensor,       # [B, P] True where valid
    ) -> torch.Tensor:
        b, q_n, d = queries.shape
        p = point_feats.shape[1]
        h, hd = self.n_heads, self.head_dim

        q = self.q_proj(queries).view(b, q_n, h, hd).transpose(1, 2)         # [B, H, Q, hd]
        k = self.k_proj(point_feats).view(b, p, h, hd).transpose(1, 2)       # [B, H, P, hd]
        v = self.v_proj(point_feats).view(b, p, h, hd).transpose(1, 2)       # [B, H, P, hd]

        content_logits = torch.matmul(q, k.transpose(-1, -2)) / (hd ** 0.5)  # [B, H, Q, P]

        # ---- physics bias term 1: radial-Doppler consistency ----
        v_hat = self.velocity_hypothesis(queries)                            # [B, Q, 2]  (vx, vy)
        cos_t, sin_t = torch.cos(point_xy), torch.sin(point_xy)              # [B, P] each
        # Projected radial velocity per (query, point) pair:
        v_radial_hat = (
            v_hat[..., 0:1] * cos_t.unsqueeze(1) + v_hat[..., 1:2] * sin_t.unsqueeze(1)
        )                                                                     # [B, Q, P]
        doppler_consistency = -torch.abs(v_radial_hat - point_doppler.unsqueeze(1))  # [B, Q, P]
        doppler_bias = self.doppler_bias_scale * doppler_consistency          # [B, Q, P]

        # ---- physics bias term 2: RCS reliability ----
        rcs_bias = self.rcs_bias_scale * point_rcs.unsqueeze(1)               # [B, 1, P] -> [B, Q, P] via broadcast

        physics_bias = (doppler_bias + rcs_bias).unsqueeze(1)                 # [B, 1, Q, P] -> broadcast over heads

        logits = content_logits + physics_bias                                # [B, H, Q, P]
        logits = logits.masked_fill(~point_mask[:, None, None, :], float("-inf"))

        attn = F.softmax(logits, dim=-1)                                       # [B, H, Q, P]
        out = torch.matmul(attn, v)                                            # [B, H, Q, hd]
        out = out.transpose(1, 2).reshape(b, q_n, d)                           # [B, Q, D]
        return self.out_proj(out)                                              # [B, Q, D]


# --------------------------------------------------------------------------- #
# 3. One PART decoder layer: self-attn over queries + PGCA + FFN
# --------------------------------------------------------------------------- #
class PARTDecoderLayer(nn.Module):
    def __init__(self, model_dim: int, point_feat_dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)

        self.pgca = PhysicsGuidedCrossAttention(model_dim, point_feat_dim, n_heads)
        self.norm2 = nn.LayerNorm(model_dim)

        hidden = int(model_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, hidden), nn.GELU(), nn.Linear(hidden, model_dim)
        )
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, queries, point_feats, point_xy, point_doppler, point_rcs, point_mask):
        # Query-to-query self-attention: lets queries suppress duplicate
        # detections of the same object (similar in spirit to DETR).
        q_attn, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q_attn)                                  # [B, Q, D]

        cross_out = self.pgca(queries, point_feats, point_xy, point_doppler, point_rcs, point_mask)
        queries = self.norm2(queries + cross_out)                                # [B, Q, D]

        queries = self.norm3(queries + self.ffn(queries))                        # [B, Q, D]
        return queries


# --------------------------------------------------------------------------- #
# 4. Full PART model
# --------------------------------------------------------------------------- #
@dataclass
class PARTConfig:
    model_dim: int = 128
    point_feat_dim: int = 64
    n_heads: int = 4
    n_layers: int = 6
    n_queries: int = 64


class PARTModel(nn.Module):
    """
    End-to-end: raw radar points -> DAQI query seeds -> L PART decoder
    layers (self-attn + PGCA + FFN) -> per-query heads:
        existence confidence  in [0, 1]
        surface point         (x, y, z) relative offset from cluster center
        ground velocity       (vx, vy) in the ground plane
    """

    def __init__(self, cfg: PARTConfig):
        super().__init__()
        self.cfg = cfg
        self.daqi = DopplerAwareQueryInit(cfg.point_feat_dim, cfg.model_dim, cfg.n_queries)

        self.layers = nn.ModuleList([
            PARTDecoderLayer(cfg.model_dim, cfg.point_feat_dim, cfg.n_heads)
            for _ in range(cfg.n_layers)
        ])

        self.existence_head = nn.Linear(cfg.model_dim, 1)
        self.surface_point_head = nn.Linear(cfg.model_dim, 3)   # (dx, dy, dz)
        self.velocity_head = nn.Linear(cfg.model_dim, 2)        # (vx, vy)

    def forward(self, radar_points: torch.Tensor, point_mask: torch.Tensor):
        """
        radar_points: [B, P, 5]  (x, y, z, doppler_v, rcs)
        point_mask:   [B, P]     True where valid

        Returns dict of:
            existence_logits: [B, Q, 1]
            surface_points:   [B, Q, 3]
            velocities:       [B, Q, 2]
        """
        point_xy = torch.atan2(radar_points[..., 1], radar_points[..., 0])   # [B, P] azimuth theta
        point_doppler = radar_points[..., 3]                                  # [B, P]
        point_rcs = radar_points[..., 4]                                      # [B, P]

        queries, point_feats = self.daqi(radar_points, point_mask)            # [B, Q, D], [B, P, Fp]

        for layer in self.layers:
            queries = layer(queries, point_feats, point_xy, point_doppler, point_rcs, point_mask)

        existence_logits = self.existence_head(queries)        # [B, Q, 1]
        surface_points = self.surface_point_head(queries)      # [B, Q, 3]
        velocities = self.velocity_head(queries)                # [B, Q, 2]

        return {
            "existence_logits": existence_logits,
            "surface_points": surface_points,
            "velocities": velocities,
        }


# --------------------------------------------------------------------------- #
# 5. Uncertainty-Aware Supervision (training-time target assignment)
# --------------------------------------------------------------------------- #
class UncertaintyAwareSupervision(nn.Module):
    """
    At each training step, randomly masks a fraction of ground-truth objects
    from the matching process (simulating missed/ambiguous radar returns),
    and assigns *soft* existence targets -- rather than a hard 0/1 -- to
    queries matched to masked or low-confidence GT objects. This keeps the
    existence loss from over-penalizing the network for genuinely ambiguous
    detections, which is common with sparse, noisy radar returns.
    """

    def __init__(self, mask_prob: float = 0.15, soft_target: float = 0.5):
        super().__init__()
        self.mask_prob = mask_prob
        self.soft_target = soft_target

    def forward(self, matched_gt_mask: torch.Tensor) -> torch.Tensor:
        """
        matched_gt_mask: [B, Q] bool, True where a query was matched
                         (e.g. via Hungarian matching) to a real GT object.

        Returns:
            existence_targets: [B, Q] float in [0, 1] -- 1.0 for a clean
            match, `soft_target` for a match that was randomly designated
            "ambiguous" this step, 0.0 for background/no-match queries.
        """
        targets = matched_gt_mask.float()                                     # [B, Q] in {0, 1}
        random_mask = torch.rand_like(targets) < self.mask_prob               # [B, Q]
        ambiguous = matched_gt_mask & random_mask                             # only touch real matches
        targets = torch.where(ambiguous, torch.full_like(targets, self.soft_target), targets)
        return targets                                                         # [B, Q]


# --------------------------------------------------------------------------- #
# Training loss combining all three heads
# --------------------------------------------------------------------------- #
def part_loss(
    outputs: dict,
    existence_targets: torch.Tensor,     # [B, Q] in [0,1] (from UncertaintyAwareSupervision)
    surface_point_targets: torch.Tensor,  # [B, Q, 3]
    velocity_targets: torch.Tensor,       # [B, Q, 2]
    matched_gt_mask: torch.Tensor,        # [B, Q] bool -- only supervise geometry/velocity on matches
    lambda_point: float = 5.0,
    lambda_vel: float = 2.0,
) -> Tuple[torch.Tensor, dict]:
    existence_loss = F.binary_cross_entropy_with_logits(
        outputs["existence_logits"].squeeze(-1), existence_targets
    )

    match_f = matched_gt_mask.unsqueeze(-1).float()
    n_matched = matched_gt_mask.sum().clamp(min=1)

    point_loss = (F.l1_loss(outputs["surface_points"], surface_point_targets, reduction="none") * match_f).sum() / (n_matched * 3)
    vel_loss = (F.l1_loss(outputs["velocities"], velocity_targets, reduction="none") * match_f).sum() / (n_matched * 2)

    total = existence_loss + lambda_point * point_loss + lambda_vel * vel_loss
    return total, {
        "existence_loss": existence_loss.item(),
        "point_loss": point_loss.item(),
        "velocity_loss": vel_loss.item(),
        "total_loss": total.item(),
    }
