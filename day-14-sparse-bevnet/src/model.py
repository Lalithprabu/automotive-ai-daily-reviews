"""
SparseBEVNet: camera-only multi-view BEV 3D detector.

Pipeline (matches the paper's described flow):
    6 camera feeds
      -> shared conv backbone, BRA-routed                (src/backbone.py)
      -> flattened per-camera tokens
      -> SparseSpatialCrossAttention lifts into a shared
         BEV grid (each BEV cell attends only its top-k
         most relevant camera views)                      (src/attention.py)
      -> CascadedGroupAttention fuses the BEV tokens       (src/attention.py)
      -> per-cell detection head: objectness logit +
         6-dim box regression (dx, dy, dw, dl, sin, cos)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.attention import CascadedGroupAttention, SparseSpatialCrossAttention
from src.backbone import ConvBackbone


class SparseBEVNet(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.num_cameras = config["num_cameras"]
        self.bev_size = config["bev_size"]
        C = config["backbone_channels"]
        self.C = C
        num_heads = config.get("num_heads", 4)

        self.backbone = ConvBackbone(
            in_channels=3,
            base_channels=C,
            use_bra=True,
            bra_region_grid=config.get("bra_region_grid", 2),
            bra_topk=config.get("bra_topk", 2),
            bra_heads=num_heads,
        )

        self.cross_attn = SparseSpatialCrossAttention(
            dim=C,
            num_cameras=self.num_cameras,
            topk_cameras=config.get("topk_cameras", 2),
            num_heads=num_heads,
        )

        # Learnable per-cell BEV positional query embedding.
        self.bev_pos_embed = nn.Parameter(torch.randn(1, self.bev_size * self.bev_size, C) * 0.02)

        self.cga = CascadedGroupAttention(dim=C, num_groups=config.get("cga_groups", 4))
        self.head_norm = nn.LayerNorm(C)

        self.objectness_head = nn.Linear(C, 1)
        self.box_head = nn.Linear(C, 6)  # dx, dy, log(dw), log(dl), sin(theta), cos(theta)

    def forward(self, images: torch.Tensor) -> dict:
        """
        images: (B, num_cameras, 3, H, W)
        Returns dict with:
            obj_logits: (B, bev_size, bev_size)
            box_reg:    (B, bev_size, bev_size, 6)
            cam_gate:   (B, num_cameras) -- averaged camera relevance weights
        """
        B, Ncam, Ch, H, W = images.shape
        assert Ncam == self.num_cameras

        imgs_flat = images.reshape(B * Ncam, Ch, H, W)
        feat = self.backbone(imgs_flat)  # (B*Ncam, C, fh, fw)
        fh, fw = feat.shape[-2:]
        tokens_per_camera = fh * fw

        feat = feat.view(B, Ncam, self.C, fh, fw)
        feat = feat.permute(0, 1, 3, 4, 2).reshape(B, Ncam * tokens_per_camera, self.C)

        bev_q = self.bev_pos_embed.expand(B, -1, -1)
        fused, cam_gate = self.cross_attn(bev_q, feat, tokens_per_camera)
        fused = fused + bev_q

        fused = self.cga(fused) + fused
        fused = self.head_norm(fused)

        obj_logits = self.objectness_head(fused).squeeze(-1)  # (B, bev*bev)
        box_reg = self.box_head(fused)  # (B, bev*bev, 6)

        obj_logits = obj_logits.view(B, self.bev_size, self.bev_size)
        box_reg = box_reg.view(B, self.bev_size, self.bev_size, 6)

        return {"obj_logits": obj_logits, "box_reg": box_reg, "cam_gate": cam_gate}
