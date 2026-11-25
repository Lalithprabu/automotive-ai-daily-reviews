"""Train one planner variant on the synthetic scenes.

    python train.py --run gaze            # GGDriver: gaze for refinement AND scoring
    python train.py --run glance          # old way: every candidate refined + scored on the same 9 shared tokens
    python train.py --run gaze_refine     # gaze for refinement only (scorer uses the glance tokens)
    python train.py --run gaze_score      # gaze for scoring only
    python train.py --run random          # a random patch mask with the same patch count as the corridor
    python train.py --run all             # attention over all 240 patches

Loss per scene, over all 45 anchors:
    BCE(glance_logit, anchor score)                     glance learns to rank the vocabulary from shared tokens
  + 2 * L1(delta / range, best delta in the 5x5 grid)   refinement target = best nearby (d, v) under the rule score
  + BCE(sub-score logits, NC / DAC / EP of the REFINED proposal, computed on the fly by the rule-based simulator)
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

from ggd.model import GlanceGazePlanner, n_params
from ggd.scene import REF_D, REF_V, make_dataset, score_traj, traj_points

ROOT = Path(__file__).resolve().parent
VARIANTS = {"glance": ("glance", "glance"), "gaze": ("gaze", "gaze"), "gaze_refine": ("gaze", "glance"),
            "gaze_score": ("glance", "gaze"), "random": ("random", "random"), "all": ("all", "all")}


def load_split(split: str, cfg: dict | None = None) -> dict:
    cfg = cfg or yaml.safe_load(open(ROOT / "config.yaml"))
    p = ROOT / "data" / f"{split}.npz"
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        n, seed = cfg["data"][f"n_{split}"], cfg["data"][f"seed_{split}"]
        t0 = time.time()
        d = make_dataset(n, seed)
        np.savez_compressed(p, **d)
        print(f"generated {split}: {n} scenes in {time.time() - t0:.0f}s")
    return dict(np.load(p))


def to_batch(d: dict, idx: np.ndarray, aug: np.random.Generator | None = None) -> dict:
    img = torch.from_numpy(d["img"][idx]).float().permute(0, 3, 1, 2) / 255.0
    if aug is not None:                                                     # photometric jitter only (geometry fixed:
        g = torch.from_numpy(aug.uniform(0.85, 1.15, (len(idx), 1, 1, 1))).float()   # calibration must stay valid)
        img = (img * g + torch.randn_like(img) * 0.01).clamp(0, 1)
    return dict(img=img, v0=torch.from_numpy(d["v0"][idx]), obj=torch.from_numpy(d["obj"][idx]),
                valid=torch.from_numpy(d["typ"][idx] >= 0), edges=torch.from_numpy(d["edges"][idx]),
                zmax=torch.from_numpy(d["zmax"][idx]), anchor_score=torch.from_numpy(d["anchor_score"][idx]),
                best_delta=torch.from_numpy(d["best_delta"][idx]))


def losses(m: GlanceGazePlanner, b: dict) -> tuple[torch.Tensor, dict]:
    o = m(b["img"], b["v0"])
    l_glance = F.binary_cross_entropy_with_logits(o["glance_logit"], b["anchor_score"])
    scale = torch.tensor([REF_D, REF_V])
    l_ref = ((o["delta"] - b["best_delta"]).abs() / scale).mean()
    with torch.no_grad():
        gt = score_traj(o["pts"], o["dv"].detach(), b["v0"], b["obj"], b["valid"], b["edges"], b["zmax"])
    tgt = torch.stack([gt["nc"], gt["dac"], gt["ep"]], -1)
    l_score = F.binary_cross_entropy_with_logits(o["sub"], tgt, weight=torch.tensor([2.0, 1.0, 1.0]))
    loss = l_glance + 2.0 * l_ref + l_score
    return loss, dict(glance=float(l_glance.detach()), refine=float(l_ref.detach()), score=float(l_score.detach()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    tc = cfg["train"]
    steps = a.steps or tc["steps"]
    torch.manual_seed(a.seed)
    torch.set_num_threads(tc.get("threads", 1))
    d = load_split("train", cfg)
    rng = np.random.default_rng(a.seed)
    m = GlanceGazePlanner(*VARIANTS[a.run], d=cfg["model"]["d"], n_reg=cfg["model"]["n_reg"])
    opt = torch.optim.AdamW(m.parameters(), lr=tc["lr"], weight_decay=tc["wd"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tc["lr"], total_steps=steps, pct_start=0.05)
    tag = a.run if a.seed == 0 else f"{a.run}_s{a.seed}"
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"{tag}.jsonl", "w")
    print(f"{tag}: refine={m.refine_mode} score={m.score_mode} params {n_params(m):,}, {steps} steps")
    t0 = time.time()
    run = {}
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(d["v0"]), tc["batch"])
        loss, parts = losses(m, to_batch(d, idx, rng))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            run[k] = 0.98 * run.get(k, v) + 0.02 * v
        if step % 100 == 0 or step == steps:
            rec = dict(step=step, sec=round(time.time() - t0, 1), **{k: round(v, 4) for k, v in run.items()})
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
    ck = ROOT / "results" / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=m.state_dict(), variant=a.run, modes=VARIANTS[a.run], cfg=cfg, steps=steps), ck / f"{tag}.pt")


if __name__ == "__main__":
    main()
