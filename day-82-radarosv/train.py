"""Train one configuration.

    python train.py --run base    # imaging radar only + EXTERNAL ego velocity (the paper's cascaded-radar baseline)
    python train.py --run osv     # imaging + single-chip radar, RADAR-ESTIMATED ego velocity (RadarOSV-style)
    python train.py --run osv_ext # ablation: imaging + single-chip, external ego velocity
    python train.py --run osv_fixed # ablation: RadarOSV-style with fixed-only PCG sampling (no deformable offsets)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from osv.model import RadarOSVNet, losses, n_params
from osv.radar import make_dataset
from osv.scene import FOV_MASK

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
RUNS = {"base": dict(use_sc=False, ego="external"), "osv": dict(use_sc=True, ego="estimated"),
        "osv_ext": dict(use_sc=True, ego="external"), "osv_fixed": dict(use_sc=True, ego="estimated", deformable=False)}


def load_split(split: str) -> dict:
    """Arrays are stored as one .npy per key and memory-mapped, so parallel runs share one copy in the page cache."""
    keys = ("img", "sc", "sem", "vgr", "vrel", "vego")
    q = ROOT / "data" / split
    if all((q / f"{k}.npy").exists() for k in keys):
        return {k: np.load(q / f"{k}.npy", mmap_mode="r") for k in keys}
    p = ROOT / "data" / f"{split}.npz"
    if not p.exists():
        n, seed = (CFG["data"]["n_train"], 1) if split == "train" else (CFG["data"]["n_test"], 2)
        t = time.time()
        d = make_dataset(n, seed)
        p.parent.mkdir(exist_ok=True)
        np.savez(p, **d)
        print(f"generated {split}: {n} scenes in {time.time() - t:.0f} s")
    z = np.load(p)
    q.mkdir(parents=True, exist_ok=True)
    for k in keys:
        np.save(q / f"{k}.npy", z[k])
    p.unlink()
    return {k: np.load(q / f"{k}.npy", mmap_mode="r") for k in keys}


def to_batch(d: dict, idx) -> dict:
    idx = np.sort(np.asarray(idx))
    return dict(img=torch.from_numpy(d["img"][idx].astype(np.float32)),
                sc=torch.from_numpy(d["sc"][idx].astype(np.float32)),
                sem=torch.from_numpy(d["sem"][idx].astype(np.int64)),
                vrel=torch.from_numpy(d["vrel"][idx].astype(np.float32)),
                vgr=torch.from_numpy(d["vgr"][idx].astype(np.float32)),
                vego=torch.from_numpy(d["vego"][idx].astype(np.float32)))


def build(run: str) -> RadarOSVNet:
    return RadarOSVNet(**RUNS[run])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=CFG["train"]["steps"])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = load_split("train")
    m = build(a.run)
    tc = CFG["train"]
    opt = torch.optim.AdamW(m.parameters(), lr=tc["lr"], weight_decay=1e-4)
    warm = int(0.05 * a.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    fov = torch.from_numpy(FOV_MASK)
    cls_w = torch.tensor(tc["class_weights"], dtype=torch.float32)
    log, t0 = [], time.time()
    print(f"run {a.run} seed {a.seed}: {n_params(m):,} params")
    for s in range(a.steps):
        b = to_batch(d, rng.integers(0, len(d["sem"]), tc["batch"]))
        out = m(b["img"], b["sc"], b["vego"])
        loss, parts = losses(out, b["sem"], b["vrel"], b["vego"], fov, cls_w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if s % 100 == 0 or s == a.steps - 1:
            parts.update(step=s, loss=loss.item(), t=round(time.time() - t0, 1))
            log.append(parts)
            print(json.dumps(parts), flush=True)
    ck = ROOT / "results" / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=m.state_dict(), run=a.run, seed=a.seed, steps=a.steps), ck / f"{a.run}_s{a.seed}.pt")
    (ROOT / "results" / "logs").mkdir(exist_ok=True)
    json.dump(log, open(ROOT / "results" / "logs" / f"{a.run}_s{a.seed}.json", "w"))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
