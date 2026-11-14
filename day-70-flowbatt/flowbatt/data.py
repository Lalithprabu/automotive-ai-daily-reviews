"""Cached synthetic splits + trajectory normalisation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import world as W

CACHE = Path(__file__).resolve().parent.parent / "checkpoints" / "fleet_cache.npz"
CAP_MEAN, CAP_STD = 0.5, 0.3


def load_splits(cfg: dict) -> dict:
    """{'train'|'val'|'test': fleet dict}.  Cells are split by CELL (no cycles shared across splits)."""
    d = cfg["data"]
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        out = {}
        for s in ("train", "val", "test"):
            out[s] = {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(s + "/")}
        return out
    out = {"train": W.make_fleet(d["n_train"], d["seed"]),
           "val": W.make_fleet(d["n_val"], d["seed"] + 1),
           "test": W.make_fleet(d["n_test"], d["seed"] + 2)}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **{f"{s}/{k}": v for s, f in out.items() for k, v in f.items()})
    return out


class Normaliser:
    """fade = 1 - SOH, standardised per grid point with TRAIN statistics only."""

    def __init__(self, soh_train: np.ndarray):
        fade = 1.0 - soh_train
        self.mu = fade.mean(0).astype(np.float32)
        self.sd = np.maximum(fade.std(0), 0.01).astype(np.float32)

    def fwd(self, soh):
        if isinstance(soh, torch.Tensor):
            return ((1 - soh) - torch.as_tensor(self.mu, device=soh.device)) / torch.as_tensor(self.sd, device=soh.device)
        return ((1 - soh) - self.mu) / self.sd

    def inv(self, x):
        if isinstance(x, torch.Tensor):
            return 1 - (x * torch.as_tensor(self.sd, device=x.device) + torch.as_tensor(self.mu, device=x.device))
        return 1 - (x * self.sd + self.mu)


def tensors(fleet: dict, norm: Normaliser):
    cap = torch.from_numpy((fleet["cap"] - CAP_MEAN) / CAP_STD).float()
    # NB: the encoder's dQ channel is computed on the standardised matrix; the 1/CAP_STD factor is absorbed.
    pre = torch.from_numpy(fleet["prefix"]).float()
    x1 = torch.from_numpy(norm.fwd(fleet["soh"])).float()
    return cap, pre, x1
