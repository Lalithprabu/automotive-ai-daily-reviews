"""Train SpecFlow and its ablations on the synthetic intersection scenes (one fixed config for every variant).

    python train.py --variant specflow --seed 0
    python train.py --all            # 5 variants x seeds in config
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

from dsf.data import agent_frame, make_dataset
from dsf.model import VARIANTS, build, n_params

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
KEYS = ["hist", "fut", "fut_clean", "v0", "rel", "mask", "kind", "origin", "psi"]


def load_data() -> dict:
    """Generate (or load the cached) train / val / test splits.  Different seeds per split -> disjoint scenes."""
    cache = ROOT / "results/data.npz"
    if cache.exists():
        z = np.load(cache)
        return {sp: {k[len(sp) + 1:]: z[k] for k in z.files if k.startswith(sp + "_")} for sp in ("train", "val", "test")}
    c = CFG["data"]
    out, flat = {}, {}
    for i, (sp, n) in enumerate((("train", c["n_train"]), ("val", c["n_val"]), ("test", c["n_test"]))):
        raw = make_dataset(n, c["seed"] * 1000 + i)
        f = agent_frame(raw["pos"], raw["clean"], raw["mask"], raw["kind"])
        f.update(pos=raw["pos"], clean=raw["clean"], man=raw["man"])
        out[sp] = f
        flat.update({f"{sp}_{k}": v for k, v in f.items()})
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **flat)
    return out


def to_batch(d: dict, idx) -> dict:
    return {k: torch.as_tensor(d[k][idx]) for k in KEYS}


def train(variant: str, seed: int, steps: int | None = None, quiet: bool = False) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    D = load_data()
    tr, va = D["train"], D["val"]
    tc = CFG["train"]
    steps = steps or tc["steps"]
    m = build(variant)
    m.h_train = tc["h_train"]
    m.fit_norm(torch.as_tensor(tr["fut"]), torch.as_tensor(tr["v0"]), torch.as_tensor(tr["mask"]))
    opt = torch.optim.AdamW(m.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 300) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    rng = np.random.default_rng(seed)
    n = len(tr["mask"])
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        b = to_batch(tr, rng.integers(0, n, tc["batch"]))
        loss, parts = m.loss(b)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % tc["log_every"] == 0 or step == steps:
            m.eval()
            with torch.no_grad():
                vb = to_batch(va, np.arange(256))
                vl, _ = m.loss(vb)
            m.train()
            rec = dict(step=step, loss=float(loss), val=float(vl), sec=round(time.time() - t0, 1), **parts)
            log.append(rec)
            if not quiet:
                print(json.dumps(rec), flush=True)
    ck = ROOT / f"results/ckpt/{variant}_s{seed}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=m.state_dict(), variant=variant, seed=seed, params=n_params(m)), ck)
    json.dump(log, open(ROOT / f"results/logs/{variant}_s{seed}.json", "w"))
    return ck


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="specflow", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        for s in (0, 1):
            for v in VARIANTS:
                train(v, s, a.steps)
    else:
        train(a.variant, a.seed, a.steps)
