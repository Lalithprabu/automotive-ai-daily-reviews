"""
BEVDetector: compact BEV backbone + anchor-free detection head.

Shared architecture used by BOTH the student (ego) and the teacher
(collaborator, an EMA-tracked copy of the student) in the Mean-Teacher
style self-training loop -- see models/lde.py.
"""

import torch
import torch.nn as nn


class BEVDetector(nn.Module):
    """
    Shapes through forward(x), x: (B, C_in, H, W):
        backbone conv1 -> (B, 32, H, W)
        backbone conv2 -> (B, 64, H, W)
        backbone conv3 (features) -> (B, feat_channels, H, W)
        cls_head -> (B, 1, H, W)   objectness logit per BEV cell
        reg_head -> (B, 4, H, W)   (dx, dy, dw, dl) box regression
    """

    def __init__(self, in_channels: int = 8, feat_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        # Anchor-free detection head: 1x1 convs directly on the backbone
        # feature map, one prediction per BEV cell (no anchors/proposals).
        self.cls_head = nn.Conv2d(feat_channels, 1, kernel_size=1)
        self.reg_head = nn.Conv2d(feat_channels, 4, kernel_size=1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, H, W) -> (B, feat_channels, H, W)"""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.backbone(x)            # (B, feat_channels, H, W)
        cls_logits = self.cls_head(feat)    # (B, 1, H, W)
        reg = self.reg_head(feat)           # (B, 4, H, W)
        return {"features": feat, "cls_logits": cls_logits, "reg": reg}
