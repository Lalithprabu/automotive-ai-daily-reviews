"""Train LOWP / LOWP-S / dense IPM / ego-MLP on the synthetic data. Usage:
    python train.py --model lowp --steps 1500 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from lowp.data import Frames, rig_tensors
from lowp.model import LOWP, DenseIPMPlanner, EgoStatusMLP, n_params
from lowp.world import make_rig


def build(name: str, cfg: dict):
    d, T, hw = cfg["model"]["d"], cfg["model"]["T"], tuple(cfg["data"]["img_hw"])
    if name == "lowp":
        return LOWP(d, T, hw, "both", cfg["model"]["alpha_straight"], cfg["model"]["alpha_turn"])
    if name == "lowp_s":
        return LOWP(d, T, hw, "global")
    if name == "dense":
        return DenseIPMPlanner(d, T, hw)
    if name == "ego_mlp":
        return EgoStatusMLP(d, T)
    raise ValueError(name)


def load(split: str):
    z = np.load(f"data/{split}.npz")
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lowp", choices=["lowp", "lowp_s", "dense", "ego_mlp"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    steps = a.steps or cfg["train"]["steps"]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(cfg["train"].get("threads", 2))

    train = Frames(load("train"))
    dl = torch.utils.data.DataLoader(train, batch_size=cfg["train"]["batch"], shuffle=True, drop_last=True,
                                     generator=torch.Generator().manual_seed(a.seed))
    K, R, t = rig_tensors(make_rig(*cfg["data"]["img_hw"]))
    model = build(a.model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["train"]["lr"], total_steps=steps, pct_start=0.1)
    print(f"{a.model}: {n_params(model):,} params, {steps} steps")
    step, t0, log = 0, time.time(), []
    while step < steps:
        for img, st, cmd, fut in dl:
            B = img.shape[0]
            Kb, Rb, tb = (x[None].expand(B, *x.shape) for x in (K, R, t))
            out = model(img, Kb, Rb, tb, st, cmd)
            loss = model.loss(out, fut)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0 or step == 1:
                print(f"step {step:5d}  loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
                log.append((step, loss.item()))
            if step >= steps:
                break
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{a.model}_s{a.seed}.pt")
    json.dump(dict(log=log, seconds=time.time() - t0, params=n_params(model)),
              open(f"checkpoints/{a.model}_s{a.seed}_train.json", "w"))


if __name__ == "__main__":
    main()
