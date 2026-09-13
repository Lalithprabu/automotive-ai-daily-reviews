"""
TrajFusionNet+ top-level model — late-fuses the three branches:

    SAM (Sequence Attention Module) -> proj_dim=40   [trajectory, verified]
    VAM (Visual Attention Module)   -> proj_dim=40   [visual overlay, verified]
    GAM (Graph Attention Module)    -> proj_dim=40   [scene graph, THE new branch]

into a single 120-dim vector, then a small dense classifier predicts
pedestrian crossing intention (binary: cross / no-cross).

Fusion trunk width (120 -> 60 -> 2) follows the base TrajFusionNet's
verified two-branch fusion trunk (80 -> 40 -> 2 for two 40-dim branches),
extended by the same ratio for a third 40-dim branch — a good-faith
reconstruction (see gam.py's sourcing note for why exact TrajFusionNet+
fusion dims aren't independently confirmed this cycle).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .gam import GAMConfig, PedestrianCentricGAM
from .sam import SAMConfig, SequenceAttentionModule
from .vam import VAMConfig, VisualAttentionModule


@dataclass
class TrajFusionNetPlusConfig:
    sam: SAMConfig = field(default_factory=SAMConfig)
    vam: VAMConfig = field(default_factory=VAMConfig)
    gam: GAMConfig = field(default_factory=GAMConfig)
    fusion_hidden_dim: int = 60
    dropout: float = 0.1


class TrajFusionNetPlus(nn.Module):
    def __init__(self, cfg: TrajFusionNetPlusConfig | None = None):
        super().__init__()
        self.cfg = cfg or TrajFusionNetPlusConfig()
        self.sam = SequenceAttentionModule(self.cfg.sam)
        self.vam = VisualAttentionModule(self.cfg.vam)
        self.gam = PedestrianCentricGAM(self.cfg.gam)

        fused_dim = self.cfg.sam.proj_dim + self.cfg.vam.proj_dim + self.cfg.gam.proj_dim  # 40+40+40=120
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, self.cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.fusion_hidden_dim, 2),  # binary logits: [no-cross, cross]
        )

    def forward(
        self,
        past_traj: torch.Tensor,            # [B, past_len, 5]
        observed_frame: torch.Tensor,       # [B, 3, H, W]
        predicted_frame: torch.Tensor,      # [B, 3, H, W]
        node_positions: torch.Tensor,       # [B, N, 2]
        node_classes: torch.Tensor,         # [B, N]
        node_areas: torch.Tensor,           # [B, N]
        node_valid_mask: torch.Tensor,      # [B, N]
    ):
        sam_feat, pred_future_traj = self.sam(past_traj)                 # [B,40], [B,future_len,5]
        vam_feat = self.vam(observed_frame, predicted_frame)             # [B,40]

        node_features, edge_bias, adj_mask = PedestrianCentricGAM.build_graph(
            node_positions, node_classes, node_areas, node_valid_mask
        )
        gam_feat = self.gam(node_features, edge_bias, adj_mask)          # [B,40]

        fused = torch.cat([sam_feat, vam_feat, gam_feat], dim=-1)        # [B,120]
        logits = self.fusion(fused)                                      # [B,2]

        return {
            "crossing_logits": logits,
            "predicted_trajectory": pred_future_traj,
            "branch_features": {"sam": sam_feat, "vam": vam_feat, "gam": gam_feat},
        }
