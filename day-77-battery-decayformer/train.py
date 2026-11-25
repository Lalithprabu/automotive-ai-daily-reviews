"""Train every variant on every aging-condition-exclusive fold with ONE fixed configuration (no per-fold tuning).

    python train.py --all                       # 6 variants x 6 folds x seeds in config.yaml
    python train.py --variant decayformer --fold 0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from dfm.data import condition_folds, make_dataset
from dfm.model import VARIANTS, build, n_params, trajectory_loss

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def load_data() -> dict:
    cache = ROOT / "results" / "data.npz"
    if cache.exists():
        return dict(np.load(cache))
    d = make_dataset(CFG["data"]["cells_per_cond"], CFG["data"]["seed"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **d)
    return d


def split(d: dict, fold: int) -> tuple[np.ndarray, np.ndarray]:
    held = condition_folds(CFG["data"]["n_folds"], CFG["data"]["seed"])[fold]
    test = np.isin(d["cond"], held)
    return np.nonzero(~test)[0], np.nonzero(test)[0]


def ckpt_path(variant: str, fold: int, seed: int) -> Path:
    return ROOT / "results" / "ckpt" / f"{variant}_f{fold}_s{seed}.pt"


def train_one(variant: str, fold: int, seed: int, d: dict, log: dict) -> None:
    tc = CFG["train"]
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed * 31 + fold)
    tr, _ = split(d, fold)
    feats = VARIANTS[variant]["feats"]
    X = torch.from_numpy(d["x"][:, :, feats])
    TR = torch.from_numpy(d["traj"]); L = torch.from_numpy(d["life"]); grid = torch.from_numpy(d["grid"])
    model = build(variant)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tc["lr"], total_steps=tc["steps"], pct_start=0.1)
    K = X.shape[1]
    t0, hist = time.time(), []
    model.train()
    for step in range(tc["steps"]):
        b = rng.choice(tr, tc["batch"], replace=False)
        k = rng.integers(tc["k_min"], K + 1, tc["batch"])
        k[rng.random(tc["batch"]) < tc["p_full"]] = K
        valid = torch.arange(K)[None] < torch.from_numpy(k)[:, None]
        out = model(X[b], valid)
        loss, parts = trajectory_loss(model, out, TR[b], grid, L[b])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == tc["steps"] - 1:
            hist.append(dict(step=step, **parts))
    p = ckpt_path(variant, fold, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    log[f"{variant}_f{fold}_s{seed}"] = dict(params=n_params(model), sec=round(time.time() - t0, 1), hist=hist)
    print(f"{variant:12s} fold {fold} seed {seed}  mape {hist[-1]['mape'] * 100:.2f}%  {time.time() - t0:.0f}s",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--variant", default="decayformer")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)                      # run two processes in parallel on 2 cores
    d = load_data()
    jobs = []
    if a.all or a.variants:
        vs = a.variants.split(",") if a.variants else list(VARIANTS)
        jobs = [(v, f, s) for v in vs for f in range(CFG["data"]["n_folds"]) for s in CFG["train"]["seeds"]]
    else:
        jobs = [(a.variant, a.fold, a.seed)]
    log = {}
    for v, f, s in jobs:
        if ckpt_path(v, f, s).exists() and (a.all or a.variants):
            continue
        train_one(v, f, s, d, log)
        lp = ROOT / "results" / "logs" / f"{v}_f{f}_s{s}.json"
        lp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(log.pop(f"{v}_f{f}_s{s}"), open(lp, "w"), indent=1)


if __name__ == "__main__":
    main()
