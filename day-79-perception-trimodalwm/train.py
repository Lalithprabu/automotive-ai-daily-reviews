"""Pretrain a base video world model on clear-weather RGB, then post-train it per sensor and as one tri-sensor video.

    python train.py --run base                  # RGB only, clear weather only        (the "pretrained video model")
    python train.py --run rgb|thermal|lidar     # post-train from base on one sensor, all six weathers
    python train.py --run tri                   # post-train from base on [RGB | thermal | LiDAR] as one video
    python train.py --run tri_scratch           # optional: same as tri without the base initialisation
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from twm.data import T_CTX, make_dataset, to_tiles
from twm.model import VideoWorldModel, n_params, tile_loss

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
RUNS = {"base": [0], "rgb": [0], "thermal": [1], "lidar": [2], "tri": [0, 1, 2], "tri_scratch": [0, 1, 2]}


def load_split(split: str) -> dict:
    c = CFG["data"]
    if split == "train":
        return make_dataset(c["train_per_weather"], c["train_seed"])
    return make_dataset(c["test_per_weather"], c["test_seed"], keep_boxes=True)


def batch_tiles(d: dict, idx: np.ndarray, sensors: list[int]) -> torch.Tensor:
    """-> canvas (B, T, 3, 48, 64 n): the chosen sensors' tiles concatenated along width."""
    t = to_tiles(d["rgb"][idx], d["th"][idx], d["depth"][idx], d["valid"][idx])      # (B, T, 3 sensors, 3, H, W)
    return torch.from_numpy(np.concatenate([t[:, :, s] for s in sensors], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tc = CFG["train"]
    d = load_split("train")
    pool = np.nonzero(d["weather"] == 0)[0] if a.run == "base" else np.arange(len(d["weather"]))
    sensors = RUNS[a.run]
    m = VideoWorldModel(d=CFG["model"]["d"])
    if a.run not in ("base", "tri_scratch"):
        sd = torch.load(ROOT / "results/ckpt/base_s0.pt", map_location="cpu")["model"]
        m.load_state_dict(sd)
    steps = tc["base_steps"] if a.run == "base" else tc["steps"]
    opt = torch.optim.AdamW(m.parameters(), lr=tc["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    w = torch.ones(1, 13, 1, 1, 1)
    w[:, :T_CTX - 1] = tc["ctx_weight"]                                                # one-step context forecasts
    log = []
    t0 = time.time()
    for step in range(steps):
        idx = rng.choice(pool, tc["batch"], replace=False)
        x = batch_tiles(d, idx, sensors)
        y = m(x, T_CTX)
        loss, parts = tile_loss(y * w + (1 - w) * y.detach(), x[:, 1:], sensors)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            rec = dict(step=step, loss=float(loss.detach()), **parts, sec=round(time.time() - t0, 1))
            log.append(rec)
            print(json.dumps(rec), flush=True)
    out = ROOT / f"results/ckpt/{a.run}_s{a.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=m.state_dict(), sensors=sensors, params=n_params(m), minutes=(time.time() - t0) / 60), out)
    json.dump(log, open(ROOT / f"results/logs/{a.run}_s{a.seed}.json", "w"))
    print(f"saved {out}  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
