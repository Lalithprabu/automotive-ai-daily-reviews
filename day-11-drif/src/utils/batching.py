"""Collate a list of synthetic scene dicts (see synthetic_scene.py) into
batched torch tensors ready for DRiFModel / compute_losses.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def collate_scenes(scenes: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-scene numpy dicts into batched torch tensors.

    Output keys / shapes:
        bev_grid        [B, C_in, G, G]      float32
        map_gt          [B, G, G]             int64
        risk_signature  [B, G, G]              float32
        classical_map   [B, G, G]               float32
        points_xy       [B, N, 2]                float32
        idx_i, idx_j    [B, P]                     int64
        y_ij            [B, P]                       float32
        planning_gt     [B, horizon, 2]               float32
    """
    def stack(key: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(np.stack([s[key] for s in scenes], axis=0), dtype=dtype)

    return {
        "bev_grid": stack("bev_grid", torch.float32),
        "map_gt": stack("map_gt", torch.int64),
        "risk_signature": stack("risk_signature", torch.float32),
        "classical_map": stack("classical_map", torch.float32),
        "points_xy": stack("points_xy", torch.float32),
        "idx_i": stack("idx_i", torch.int64),
        "idx_j": stack("idx_j", torch.int64),
        "y_ij": stack("y_ij", torch.float32),
        "planning_gt": stack("planning_gt", torch.float32),
    }
