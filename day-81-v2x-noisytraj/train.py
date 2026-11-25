"""Train the predictor on clean or noise-augmented histories, or train the learned input denoiser.

    python train.py --run clean   --seed 0     # curated / noise-free training (the usual practice)
    python train.py --run aug     --seed 0     # retrain on noise-augmented data, level ~ U(0, 4) per scene
    python train.py --run denoise --seed 0     # learned input denoiser (noisy -> clean history), for the clean model
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from nr.model import HistoryDenoiser, InteractionTransformer, denoise_loss, n_params, wta_loss
from nr.noise import add_noise
from nr.scene import T_H, make_dataset

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def load_split(split: str) -> dict:
    p = ROOT / "data" / f"{split}.npz"
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        n, seed = CFG["data"][split]
        t0 = time.time()
        d = make_dataset(n, seed)
        np.savez_compressed(p, **d)
        print(f"generated {split}: {n} scenes in {time.time() - t0:.1f}s")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def to_batch(d: dict, idx) -> dict:
    st = torch.from_numpy(d["states"][idx])
    return dict(hist=st[:, :, :T_H], fut=st[:, :, T_H:, :2], v2x=torch.from_numpy(d["v2x"][idx]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["clean", "aug", "denoise"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=CFG["train"]["steps"])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(100 + a.seed)
    gen = torch.Generator().manual_seed(1000 + a.seed)
    d = load_split("train")
    n = len(d["states"])
    max_lvl = CFG["noise"]["train_max_level"]
    model = HistoryDenoiser() if a.run == "denoise" else InteractionTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["train"]["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["train"]["lr"], total_steps=a.steps, pct_start=0.1)
    name = f"{a.run}_s{a.seed}"
    log = open(ROOT / "results" / "logs" / f"{name}.log", "w")
    print(f"{name}: {n_params(model):,} params, {a.steps} steps", file=log, flush=True)
    t0 = time.time()
    run = {}
    bs = CFG["train"]["batch"]
    for step in range(1, a.steps + 1):
        b = to_batch(d, rng.integers(0, n, bs))
        if a.run == "clean":
            hist, fresh = b["hist"], torch.ones(b["hist"].shape[:3])
        else:
            lvl = torch.rand(bs, generator=gen) * max_lvl                         # one noise level per scene
            hist, fresh = add_noise(b["hist"], b["v2x"], lvl, gen)
        if a.run == "denoise":
            den, _ = model(hist, fresh, b["v2x"])
            loss = denoise_loss(den, b["hist"])
            info = dict(loss=loss.item())
        else:
            loss, info = wta_loss(model(hist, fresh, b["v2x"]), b["fut"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in info.items():
            run[k] = run.get(k, 0.0) + v
        if step % 200 == 0:                                                       # mean over the last 200 steps
            print(f"step {step} {json.dumps({k: round(v / 200, 4) for k, v in run.items()})} {time.time() - t0:.0f}s",
                  file=log, flush=True)
            run = {}
    (ROOT / "results" / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ROOT / "results" / "ckpt" / f"{name}.pt")
    print(f"done {time.time() - t0:.0f}s", file=log, flush=True)


if __name__ == "__main__":
    main()
