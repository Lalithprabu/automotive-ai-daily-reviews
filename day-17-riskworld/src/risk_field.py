"""Spatial risk field module.

RECONSTRUCTION NOTE: the abstract names "spatial risk fields" as an input
component combined with BEV+actor context. This module's exact head shape
(2-layer conv -> sigmoid, producing a per-cell scalar in [0, 1]) is this
repo's own default, not a paper-verified detail.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatialRiskField(nn.Module):
    """Predicts a dense per-cell risk value from fused BEV+actor features.

    Input:  (B, C, H, W) fused feature map.
    Output: (B, 1, H, W) risk map, values in [0, 1] (1 = maximally risky).
    """

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_feat: (B, C, H, W)
        Returns:
            risk_field: (B, 1, H, W), values in [0, 1].
        """
        logits = self.net(fused_feat)      # (B, 1, H, W)
        risk_field = torch.sigmoid(logits)  # (B, 1, H, W), in [0, 1]
        return risk_field
