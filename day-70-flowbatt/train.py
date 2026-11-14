"""python train.py --model {flowbatt,diffbatt,transformer,mlp} [--seed 0] [--steps N]"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from flowbatt import world as W
from flowbatt.data import Normaliser, load_splits, tensors
from flowbatt.model import MODELS, n_params


def build(name: str, cfg: dict):
    return MODELS[name](cfg["model"], cfg["data"]["traj_len"], cfg["data"]["prefix_len"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(int(__import__("os").environ.get("NTHREADS", "2")))
    sp = load_splits(cfg)
    norm = Normaliser(sp["train"]["soh"])
    cap, pre, x1 = tensors(sp["train"], norm)
    vcap, vpre, vx1 = tensors(sp["val"], norm)
    m = build(a.model, cfg)
    tc = cfg["train"]
    steps = a.steps or tc["steps"]
    opt = torch.optim.AdamW(m.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / tc["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    g = torch.Generator().manual_seed(a.seed)
    log, t0 = [], time.time()
    for s in range(steps):
        m.train()
        idx = torch.randint(0, len(cap), (tc["batch"],), generator=g)
        loss = m.loss(x1[idx], cap[idx], pre[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if s % 250 == 0 or s == steps - 1:
            m.eval()
            with torch.no_grad():
                torch.manual_seed(123)
                vl = float(np.mean([m.loss(vx1, vcap, vpre).item() for _ in range(3)]))
            log.append(dict(step=s, train=loss.item(), val=vl, sec=round(time.time() - t0, 1)))
            print(f"[{a.model} s{a.seed}] step {s:5d}  train {loss.item():.4f}  val {vl:.4f}  {time.time() - t0:6.1f}s",
                  flush=True)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(m.state_dict(), f"checkpoints/{a.model}_s{a.seed}.pt")     # final step, no test-based selection
    json.dump(dict(model=a.model, seed=a.seed, params=n_params(m), steps=steps, log=log,
                   train_sec=round(time.time() - t0, 1)), open(f"checkpoints/{a.model}_s{a.seed}_log.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
