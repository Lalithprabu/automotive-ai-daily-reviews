"""Train ID-MAE, its no-reconstruction ablation and a supervised GRU baseline on the synthetic simulator data.

    python train.py --all                                   # every variant x label fraction x seed in config.yaml
    python train.py --variant idmae --label_frac 0.1 --seed 0

Variants
    idmae     masked-diffusion objective: label CE + lambda * masked gaze / vehicle reconstruction, ALL windows
              (unlabelled windows contribute reconstruction only; with partial labels, half of each batch is drawn
              from the labelled windows so both variants see labels at a similar rate)
    norecon   same network and masking, lambda = 0, so only labelled windows carry any signal
    gru       supervised 2-layer GRU over the raw 7-channel signals, labelled windows only ("old way")
All variants get the same number of optimizer steps and batch size.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from idm.data import make_dataset
from idm.model import GRUClassifier, IDMAE, n_params

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def load_data(split: str) -> dict:
    """Generate once and cache to results/data_<split>.npz."""
    p = ROOT / f"results/data_{split}.npz"
    if p.exists():
        return dict(np.load(p))
    c = CFG["data"]
    lo, hi = c[f"{split}_drivers"]
    t0 = time.time()
    d = make_dataset(range(lo, hi), c["n_loc"], c["win_s"], c["warm_s"], seed=0 if split == "train" else 1)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **d)
    print(f"generated {split}: {len(d['y'])} windows in {time.time() - t0:.0f}s")
    return d


def labelled_mask(n: int, frac: float, seed: int) -> np.ndarray:
    """Which training windows keep their label (fixed per label fraction, shared across variants)."""
    r = np.random.default_rng(1234 + int(frac * 1000))
    m = np.zeros(n, bool)
    m[r.permutation(n)[:int(round(frac * n))]] = True
    return m


def build(variant: str):
    if variant == "gru":
        return GRUClassifier()
    m = CFG["model"]
    return IDMAE(d=m["d"], layers=m["layers"], heads=m["heads"], ff=m["ff"], dropout=m["dropout"])


def ckpt_path(variant, frac, seed):
    return ROOT / f"results/ckpt/{variant}_f{int(frac * 100):03d}_s{seed}.pt"


def train_one(variant: str, frac: float, seed: int, data: dict | None = None) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    d = data or load_data("train")
    G, V, R = (torch.from_numpy(d[k]) for k in ("gaze", "veh", "road"))
    Y = torch.from_numpy(d["y"])
    lab = torch.from_numpy(labelled_mask(len(Y), frac, seed))
    pool = torch.arange(len(Y)) if variant == "idmae" else torch.nonzero(lab)[:, 0]
    model = build(variant)
    tc = CFG["train"]
    opt = torch.optim.AdamW(model.parameters(), tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tc["lr"], total_steps=tc["steps"], pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    log, t0 = [], time.time()
    model.train()
    for step in range(tc["steps"]):
        idx = pool[torch.randint(len(pool), (tc["batch"],), generator=g)]
        if variant == "idmae" and frac < 1.0:                  # half the batch from the labelled pool
            lp = torch.nonzero(lab)[:, 0]
            idx[: tc["batch"] // 2] = lp[torch.randint(len(lp), (tc["batch"] // 2,), generator=g)]
        if variant == "gru":
            loss = F.cross_entropy(model(G[idx], V[idx], R[idx]), Y[idx]); parts = dict(cls=float(loss))
        else:
            lam = tc["lam"] if variant == "idmae" else 0.0
            loss, parts = model.loss(G[idx], V[idx], R[idx], Y[idx], lab[idx], lam=lam, t_min=tc["t_min"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == tc["steps"] - 1:
            log.append(dict(step=step, **parts))
            print(f"{variant} f={frac} s={seed} step {step}: {parts}  ({time.time() - t0:.0f}s)", flush=True)
    p = ckpt_path(variant, frac, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    return dict(variant=variant, frac=frac, seed=seed, params=n_params(model), secs=round(time.time() - t0),
                n_pool=len(pool), log=log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="idmae", choices=["idmae", "norecon", "gru"])
    ap.add_argument("--label_frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    data = load_data("train"); load_data("test")
    jobs = ([(v, f, s) for f in CFG["train"]["label_fracs"] for s in CFG["train"]["seeds"]
             for v in ("idmae", "norecon", "gru")] if a.all else [(a.variant, a.label_frac, a.seed)])
    lp = ROOT / "results/train_log.json"
    logs = json.load(open(lp)) if lp.exists() else {}
    for v, f, s in jobs:
        key = f"{v}_f{int(f * 100):03d}_s{s}"
        if a.all and key in logs and ckpt_path(v, f, s).exists():
            continue
        logs[key] = train_one(v, f, s, data)
        json.dump(logs, open(lp, "w"), indent=1)


if __name__ == "__main__":
    main()
