"""Evaluate trained checkpoints on the synthetic test split.

Metrics (nuScenes-style, on our toy): L2 (m) at 1/2/3 s and their average, 3-s collision rate (%),
L2 split by command (straight / turns), params, CPU latency (batch 1, median of 50 runs, 1 thread).
Also reports per-expert results for LOWP (deformable-only, global-only, mixed).
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import yaml

from lowp.data import Frames, rig_tensors
from lowp.model import n_params
from lowp.world import collides, make_rig
from train import build, load


@torch.no_grad()
def predict(model, data, K, R, t, key="tau", bs=100):
    ds = Frames(data)
    outs = []
    for i in range(0, len(ds), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        img = torch.stack([x[0] for x in items]); st = torch.stack([x[1] for x in items])
        cmd = torch.tensor([x[2] for x in items]); B = img.shape[0]
        Kb, Rb, tb = (x[None].expand(B, *x.shape) for x in (K, R, t))
        o = model(img, Kb, Rb, tb, st, cmd)
        outs.append(o[key] if isinstance(key, str) else torch.stack([o[k] for k in key], 0))
    return torch.cat(outs, -3).numpy()


def metrics(pred, data):
    gt = data["fut"]
    err = np.linalg.norm(pred - gt, axis=-1)                         # (S, T)
    l2 = {f"L2@{s}s": float(err[:, 2 * s - 1].mean()) for s in (1, 2, 3)}   # 0.5 s spacing: idx 1, 3, 5
    l2["L2_avg"] = float(np.mean([l2["L2@1s"], l2["L2@2s"], l2["L2@3s"]]))
    obs = data["obs_fut"]
    col = [collides(pred[i], obs[i][:, np.isfinite(obs[i][0, :, 0])]) for i in range(len(pred))]
    l2["collision_%"] = 100 * float(np.mean(col))
    cmd = data["cmd"]
    l2["L2_avg_straight"] = float(err[cmd == 0][:, [1, 3, 5]].mean())
    l2["L2_avg_turns"] = float(err[cmd != 0][:, [1, 3, 5]].mean())
    return l2


@torch.no_grad()
def latency(model, K, R, t, hw, n=50):
    torch.set_num_threads(1)
    img = torch.rand(1, 4, 3, *hw); st = torch.zeros(1, 2); cmd = torch.zeros(1, dtype=torch.long)
    for _ in range(5):
        model(img, K[None], R[None], t[None], st, cmd)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); model(img, K[None], R[None], t[None], st, cmd); ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    test = load("test")
    K, R, t = rig_tensors(make_rig(*cfg["data"]["img_hw"]))
    res = {}
    for name in ["ego_mlp", "dense", "lowp_s", "lowp"]:
        m = build(name, cfg).eval()
        m.load_state_dict(torch.load(f"checkpoints/{name}_s0.pt"))
        r = metrics(predict(m, test, K, R, t), test)
        r["params"] = n_params(m)
        r["latency_ms_cpu_b1"] = latency(m, K, R, t, cfg["data"]["img_hw"])
        res[name] = r
        if name == "lowp":
            p = predict(m, test, K, R, t, key=("tau_def", "tau_glob"))
            res["lowp_def_only"] = metrics(p[0], test)
            res["lowp_glob_head_only"] = metrics(p[1], test)
            # mix-weight sweep (paper fixes 0.5 straight / 0.8 turns)
            res["alpha_sweep_L2avg"] = {f"{a:.1f}": metrics(a * p[0] + (1 - a) * p[1], test)["L2_avg"]
                                        for a in np.linspace(0, 1, 6)}
    # constant-velocity, straight-ahead reference
    cv = np.zeros_like(test["fut"])
    cv[..., 0] = test["status"][:, :1] * 10.0 * (np.arange(1, 7) * 0.5)[None]
    res["const_velocity"] = metrics(cv, test)
    json.dump(res, open("checkpoints/test_metrics.json", "w"), indent=2)
    keys = ["L2@1s", "L2@2s", "L2@3s", "L2_avg", "collision_%", "L2_avg_straight", "L2_avg_turns"]
    print(f"{'model':22s}" + "".join(f"{k:>16s}" for k in keys) + f"{'params':>10s}{'ms':>8s}")
    for n, r in res.items():
        if n == "alpha_sweep_L2avg":
            continue
        print(f"{n:22s}" + "".join(f"{r[k]:16.3f}" for k in keys)
              + f"{r.get('params', 0):10d}{r.get('latency_ms_cpu_b1', 0):8.1f}")
    print("alpha sweep (L2_avg):", res["alpha_sweep_L2avg"])


if __name__ == "__main__":
    main()
