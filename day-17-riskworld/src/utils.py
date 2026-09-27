"""Small shared helpers: config loading, seeding, and grid coordinate math.

RECONSTRUCTION NOTE: the normalized-coordinate / grid_sample warping convention
used throughout this repo is a standard PyTorch pattern (used the same way in
optical-flow warping code generally), not something taken from the paper --
the paper's abstract only says flow-guided transport is used, not the exact
warping implementation.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load the YAML config. All numeric hyperparameters must be plain decimals
    (see the comment at the top of config.yaml) so this stays a straightforward
    yaml.safe_load with no custom float-parsing workarounds needed."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def grid_to_norm(coords: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Convert continuous grid-cell coordinates (range [0, grid_size - 1]) to the
    normalized [-1, 1] coordinates that `torch.nn.functional.grid_sample`
    expects.

    Args:
        coords: (..., 2) tensor of (x, y) in cell units.
        grid_size: number of cells per side (assumes a square grid).
    Returns:
        (..., 2) tensor of normalized coordinates in [-1, 1].
    """
    return 2.0 * coords / (grid_size - 1) - 1.0


def make_base_grid(grid_size: int, device: torch.device) -> torch.Tensor:
    """Build the identity sampling grid used as the base for flow warping.

    Returns:
        (1, H, W, 2) tensor of normalized (x, y) coordinates, ready to be
        broadcast-added to a predicted flow field and passed to grid_sample.
    """
    ys, xs = torch.meshgrid(
        torch.linspace(-1.0, 1.0, grid_size, device=device),
        torch.linspace(-1.0, 1.0, grid_size, device=device),
        indexing="ij",
    )
    base = torch.stack([xs, ys], dim=-1)  # (H, W, 2) -- grid_sample wants (x, y) order
    return base.unsqueeze(0)  # (1, H, W, 2)
