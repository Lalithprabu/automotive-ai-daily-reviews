"""
DRiFModel -- wires the shared BEVEncoder to the three parallel heads
(static map / dynamic risk / planning) and implements the combined,
weighted multi-task loss described in the paper.

Loss mix (paper's ranking-only supervision needs reweighting toward the
sparse ranking term, plus a TV-smoothness regularizer on the dense risk
map -- see module docstring in risk_head.py / README "known issue" note):

    L_total = w_rank * L_rank + w_map * L_map + w_plan * L_plan + w_tv * L_tv

Bug fix baked in from the start: without w_tv, a densely-decoded risk map
trained purely from sparse pairwise pairs renders as checkerboard noise
once you visualize the *full* grid, because nothing in the sampled-pair
loss constrains neighboring un-sampled cells to agree. L_tv is a total-
variation smoothness penalty on the dense R_hat grid that fixes this.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bev_encoder import BEVEncoder
from src.models.risk_head import (
    DynamicRiskHead,
    PlanningHead,
    StaticMapHead,
    pairwise_ranking_loss,
    sample_risk_at_points,
)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Multi-class soft Dice loss.
    Input:  logits   # [B, C, H, W]
            target    # [B, H, W] int64 class indices
    Output: scalar
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]

    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dims)
    union = probs.sum(dims) + target_onehot.sum(dims)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice_per_class.mean()


def total_variation_loss(risk_map: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV smoothness regularizer on a dense [B, H, W] risk grid:
    mean absolute difference between horizontally/vertically adjacent cells.
    Encourages spatial consistency between points that pairwise supervision
    never directly compares.
    """
    d_h = (risk_map[:, 1:, :] - risk_map[:, :-1, :]).abs().mean()
    d_w = (risk_map[:, :, 1:] - risk_map[:, :, :-1]).abs().mean()
    return d_h + d_w


class DRiFModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 32,
        bottleneck_channels: int = 128,
        attn_heads: int = 4,
        static_map_classes: int = 3,
        planning_horizon: int = 8,
        planning_dim: int = 2,
    ):
        super().__init__()
        self.encoder = BEVEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            bottleneck_channels=bottleneck_channels,
            attn_heads=attn_heads,
        )
        self.static_map_head = StaticMapHead(bottleneck_channels, static_map_classes)
        self.dynamic_risk_head = DynamicRiskHead(bottleneck_channels)
        self.planning_head = PlanningHead(bottleneck_channels, planning_horizon, planning_dim)

    def forward(self, bev_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_t = self.encoder(bev_grid)                     # [B, C_bneck, G/4, G/4] shared feature
        map_logits = self.static_map_head(f_t)              # [B, num_classes, G, G]
        risk_map = self.dynamic_risk_head(f_t)                # [B, G, G]
        waypoints = self.planning_head(f_t)                    # [B, horizon, dim]
        return {"map_logits": map_logits, "risk_map": risk_map, "waypoints": waypoints, "f_t": f_t}

    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        w_rank: float = 3.0,
        w_map: float = 0.4,
        w_plan: float = 0.3,
        w_tv: float = 0.08,
        rank_margin: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        map_logits = outputs["map_logits"]
        risk_map = outputs["risk_map"]
        waypoints = outputs["waypoints"]

        # --- Static map: CE + Dice --------------------------------------
        map_ce = F.cross_entropy(map_logits, batch["map_gt"])
        map_dice = dice_loss(map_logits, batch["map_gt"])
        l_map = map_ce + map_dice

        # --- Dynamic risk: pairwise ranking + TV smoothness --------------
        r_hat = sample_risk_at_points(risk_map, batch["points_xy"])  # [B, N]
        l_rank = pairwise_ranking_loss(
            r_hat, batch["idx_i"], batch["idx_j"], batch["y_ij"], margin=rank_margin
        )
        l_tv = total_variation_loss(risk_map)

        # --- Planning: Smooth-L1 ------------------------------------------
        l_plan = F.smooth_l1_loss(waypoints, batch["planning_gt"])

        l_total = w_rank * l_rank + w_map * l_map + w_plan * l_plan + w_tv * l_tv

        return {
            "total": l_total,
            "rank": l_rank,
            "map": l_map,
            "plan": l_plan,
            "tv": l_tv,
        }
