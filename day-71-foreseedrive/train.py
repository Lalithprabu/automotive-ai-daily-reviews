"""Three-stage training (paper: teacher -> predictor distillation, then adapters on a FROZEN generator).

    python train.py --stage base                      # base anchor generator (WTA imitation)
    python train.py --stage predictor                 # token predictor h_ψ distilled from logged-future teacher
    python train.py --stage cond --variant full       # ForeseeDrive adapters (geometry + logit, preserve, guard)
    python train.py --stage cond --variant scorer     # logit path only  ("a scorer can only pick")
    python train.py --stage cond --variant naive      # geometry + logit, NO preservation / guard
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

from foresee import data as D
from foresee import losses as L
from foresee import world as W
from foresee.model import BaseGenerator, ForeseeDrive, TokenPredictor, inject, n_params

import os
CK = Path(os.environ.get("FD_CK", "checkpoints"))          # FD_CK / FD_CFG let you run a second config
CFG = os.environ.get("FD_CFG", "config.yaml")


def kmeans(x: np.ndarray, k: int, it: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(it):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([x[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
    return c


def get_data(cfg):
    dc = cfg["data"]
    tr = D.to_torch(D.load("train", dc["n_train"], dc["seed_train"]))
    te = D.to_torch(D.load("test", dc["n_test"], dc["seed_test"]))
    return tr, te


def build_base(cfg, tr=None):
    p = CK / "anchors.pt"
    if p.exists():
        anchors = torch.load(p)
    else:
        ex = tr["expert"].numpy().reshape(len(tr["expert"]), -1)
        anchors = torch.from_numpy(kmeans(ex, cfg["model"]["K"]).reshape(-1, W.H, 2)).float()
        CK.mkdir(exist_ok=True); torch.save(anchors, p)
    return BaseGenerator(anchors, cfg["model"]["d"], cfg["model"]["base_layers"])


def load_base(cfg):
    base = build_base(cfg)
    base.load_state_dict(torch.load(CK / "base.pt")); base.eval()
    return base


def load_pred(cfg):
    pred = TokenPredictor(cfg["model"]["K"], cfg["model"]["d"])
    pred.load_state_dict(torch.load(CK / "predictor.pt")); pred.eval()
    return pred


def build_cond(cfg, base, variant):
    return ForeseeDrive(base, geometry=variant != "scorer", logit=True, d=cfg["model"]["d"])


def loop(model_params, n, epochs, bs, lr, step_fn, log, tag):
    opt = torch.optim.AdamW(model_params, lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot, t0 = 0.0, time.time()
        for i in range(0, n, bs):
            loss = step_fn(perm[i:i + bs])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, 5.0)
            opt.step(); sched.step()
            tot += loss.item() * len(perm[i:i + bs])
        msg = f"[{tag}] epoch {ep + 1}/{epochs} loss {tot / n:.4f} ({time.time() - t0:.0f}s)"
        print(msg, flush=True); log.append(msg)


@torch.no_grad()
def batched(fn, n, bs=512):
    outs = [fn(torch.arange(i, min(n, i + bs))) for i in range(0, n, bs)]
    if isinstance(outs[0], tuple):
        return tuple(torch.cat([o[j] for o in outs]) for j in range(len(outs[0])))
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["base", "predictor", "cond"])
    ap.add_argument("--variant", default="full", choices=["full", "scorer", "naive"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG)); tc = cfg["train"]
    torch.manual_seed(a.seed)
    tr, te = get_data(cfg)
    n = len(tr["v_ego"]); log = []
    CK.mkdir(exist_ok=True)

    if a.stage == "base":
        base = build_base(cfg, tr)
        print("base params", n_params(base))

        def step(idx):
            b = D.batch(tr, idx)
            traj, logit, _, _ = base(b)
            return L.wta_loss(traj, logit, b["expert"])[0]
        base.train()
        loop(list(base.parameters()), n, tc["base_epochs"], tc["batch"], tc["lr"], step, log, "base")
        torch.save(base.state_dict(), CK / "base.pt")

    elif a.stage == "predictor":
        base = load_base(cfg)
        tb = batched(lambda i: base(D.batch(tr, i))[0], n)                    # frozen base candidates
        pred = TokenPredictor(cfg["model"]["K"], cfg["model"]["d"])
        print("predictor params", n_params(pred))

        def step(idx):
            b = D.batch(tr, idx)
            teacher = L.teacher_tokens(tb[idx], b)
            return L.distill_loss(pred(b, tb[idx]), teacher)
        pred.train()
        loop(list(pred.parameters()), n, tc["pred_epochs"], tc["batch"], tc["lr"], step, log, "predictor")
        torch.save(pred.state_dict(), CK / "predictor.pt")

    else:
        base, pred = load_base(cfg), load_pred(cfg)
        tb = batched(lambda i: base(D.batch(tr, i))[0], n)
        v_pred = batched(lambda i: inject(*pred(D.batch(tr, i), tb[i]), from_logits=True), n)
        fd = build_cond(cfg, base, a.variant)
        params = [p for p in fd.parameters() if p.requires_grad]
        print(f"ForeseeDrive[{a.variant}] trainable params", sum(p.numel() for p in params),
              "| frozen base", n_params(base))
        lp = 0.0 if a.variant == "naive" else tc["lambda_p"]
        lg = 0.0 if a.variant == "naive" else tc["lambda_g"]
        g = torch.Generator().manual_seed(a.seed + 7)

        def step(idx):
            b = D.batch(tr, idx)
            tE, tC, tR = L.teacher_tokens(tb[idx], b)
            if torch.rand(1, generator=g).item() < tc["p_teacher"]:
                v = inject(tE, tC, tR, from_logits=False)
            else:
                v = v_pred[idx]
            o = fd(b, v)
            loss, _ = L.wta_loss(o["traj"], o["logit"], b["expert"])
            if a.variant != "scorer":
                loss = loss + lp * L.preserve_loss(o["traj"], o["base_traj"], b, tR[:, -base.K:])
                loss = loss + lg * L.guard_loss(o["traj"], o["base_traj"], b, tc["eps_b"], tc["eps_g"])
            return loss
        fd.train()
        loop(params, n, tc["cond_epochs"], tc["batch"], tc["lr"], step, log, f"cond-{a.variant}")
        torch.save({k: v for k, v in fd.state_dict().items() if not k.startswith("base.")},
                   CK / f"cond_{a.variant}_s{a.seed}.pt")
    Path(CK / f"log_{a.stage}_{a.variant if a.stage == 'cond' else ''}{a.seed}.txt").write_text("\n".join(log))


if __name__ == "__main__":
    main()
