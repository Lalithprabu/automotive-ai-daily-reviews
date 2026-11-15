"""
src/utils.py
==========================================================================
Shared utilities: config loading, seeding, checkpoint I/O, and dataset
construction (data -> normalized train/val/test BatteryICDataset splits).
==========================================================================
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import yaml

# Make sure the project root is importable regardless of CWD when this is
# run as `python src/train.py` from the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.synthetic_battery import generate_cells, split_cells, BatteryICDataset  # noqa: E402
from models.chargeic_lstm import ChargeICLSTM  # noqa: E402


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(_ROOT, "config.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(cfg: dict):
    """
    Generate the synthetic 53-cell dataset, split by CELL (unseen-cell
    generalization), and build normalized train/val/test BatteryICDataset
    objects. Val and test reuse the TRAIN split's mean/std (never their own)
    to avoid leaking held-out statistics into normalization.
    """
    seed = cfg.get("seed", 42)
    cells = generate_cells(cfg, seed=seed)
    train_cells, val_cells, test_cells = split_cells(cells, cfg, seed=seed)

    train_ds = BatteryICDataset(train_cells)  # computes mean/std from train only
    val_ds = BatteryICDataset(val_cells, mean=train_ds.mean, std=train_ds.std)
    test_ds = BatteryICDataset(test_cells, mean=train_ds.mean, std=train_ds.std)

    return train_ds, val_ds, test_ds, train_cells, val_cells, test_cells


def build_model(cfg: dict) -> ChargeICLSTM:
    m = cfg["model"]
    d = cfg["data"]
    return ChargeICLSTM(
        input_size=m["input_size"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        ic_grid_size=d["ic_grid_M"],
        ic_head_hidden=m["ic_head_hidden"],
        soh_head_hidden=m["soh_head_hidden"],
    )


def save_checkpoint(path: str, model: torch.nn.Module, cfg: dict, mean: np.ndarray, std: np.ndarray, extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "norm_mean": mean,
        "norm_std": std,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)
