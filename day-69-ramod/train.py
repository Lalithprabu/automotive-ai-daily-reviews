"""Train RaMOD or a baseline on the synthetic radar world.

    python train.py --model ramod --seed 0
    models: ramod            full model (T=3, ego + static suppression, two velocity hypotheses + p_wrap)
            ramod_noegosup   same network, ego-motion branch NOT supervised      (paper ablation)
            ramod_nowraploss same network, no wrapping loss                      (paper ablation)
            ramod_nosup      no ego branch / no static suppression (structure removed)
            rad_cnn          single-frame RAD-cube detector, raw-Doppler velocity only (typical RAD baseline)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ramod.data import RadarSeq, geom, load_or_make
from ramod.losses import ramod_loss
from ramod.model import RaMOD, n_params

VARIANTS = {   # (model kwargs, loss kwargs)
    "ramod":            (dict(T=3, suppress=True,  wrap_head=True),  dict()),
    "ramod_noegosup":   (dict(T=3, suppress=True,  wrap_head=True),  dict(ego_sup=False)),
    "ramod_nowraploss": (dict(T=3, suppress=True,  wrap_head=True),  dict(wrap_sup=False)),
    "ramod_nosup":      (dict(T=3, suppress=False, wrap_head=True),  dict()),
    "rad_cnn":          (dict(T=1, suppress=False, wrap_head=False), dict()),
}


def build(name: str, cfg: dict) -> RaMOD:
    return RaMOD(geom(), c=cfg["model"]["channels"], **VARIANTS[name][0])


def splits(cfg: dict):
    d = cfg["data"]
    tr = load_or_make(d["train_cache"], d["train_episodes"], d["train_seed"])
    te = load_or_make(d["test_cache"], d["test_episodes"], d["test_seed"])
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ramod", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(cfg["train"]["threads"])
    tr, _ = splits(cfg)
    model = build(a.model, cfg)
    ds = RadarSeq(tr, T=model.T)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch"], shuffle=True, drop_last=True, num_workers=0)
    steps = a.steps or cfg["train"]["steps"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["train"]["lr"], total_steps=steps, pct_start=0.1)
    print(f"{a.model}: {n_params(model)} params, {len(ds)} train samples")
    it, t0, step = iter(dl), time.time(), 0
    model.train()
    while step < steps:
        try:
            cubes, heat, reg, mask, ego, _ = next(it)
        except StopIteration:
            it = iter(dl); continue
        out = model(cubes)
        loss, parts = ramod_loss(out, heat, reg, mask, ego, model.vmax, **VARIANTS[a.model][1])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d} loss {loss.item():.3f} " + " ".join(f"{k}={v:.3f}" for k, v in parts.items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{a.model}_s{a.seed}.pt")
    json.dump(dict(model=a.model, seed=a.seed, steps=steps, params=n_params(model), train_s=time.time() - t0),
              open(f"checkpoints/{a.model}_s{a.seed}_train.json", "w"))


if __name__ == "__main__":
    main()
