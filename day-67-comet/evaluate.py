"""Delay sweep: uncorrected vs current-message-only vs CoMET (Kalman) vs oracle offset.

Usage:  python evaluate.py --seeds 20 --out assets/results.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from comet import CoMETConfig, CoMETTracker
from comet.metrics import fuse, match_greedy, prf
from comet.world import WorldConfig, generate

THR = 1.0  # m, center-distance match threshold (strict on purpose: pose errors are ~1-2 m)


def run_sequence(seed: int, delay_ms: float, thr: float = THR, tracker_kw: dict | None = None):
    wc = WorldConfig(delay_ms=delay_ms)
    frames = generate(seed, wc)
    kw = tracker_kw or {}
    trackers = {
        "comet": CoMETTracker(CoMETConfig(use_filter=True, **kw)),
        "single": CoMETTracker(CoMETConfig(use_filter=False, **kw)),
    }
    cnt = {k: np.zeros(3, int) for k in ["raw", "single", "comet", "oracle"]}
    cnt_road = {k: np.zeros(3, int) for k in cnt}  # recall bookkeeping on roadside-only objects
    err = {"single": [], "comet": [], "raw": []}
    no_pair = 0
    for f in frames:
        ego = torch.tensor(f.ego_xy, dtype=torch.float32)[None]
        road = torch.tensor(f.road_xy, dtype=torch.float32)[None]
        em = torch.ones(1, ego.shape[1], dtype=torch.bool)
        rm = torch.ones(1, road.shape[1], dtype=torch.bool)
        off, corrected = {}, {}
        for k, tr in trackers.items():
            out = tr.step(ego, road, em, rm, dt=wc.dt)
            off[k] = out["offset"][0].numpy()
            corrected[k] = out["corrected_road_xy"][0].numpy()
            if k == "comet" and int(out["n_pairs"][0]) == 0:
                no_pair += 1
        corrected["raw"] = f.road_xy
        corrected["oracle"] = f.road_xy - f.true_err
        for k in ["single", "comet"]:
            err[k].append(np.linalg.norm(off[k] - f.true_err))
        err["raw"].append(np.linalg.norm(f.true_err))
        gt_all = f.gt_xy[f.gt_vis_ego | f.gt_vis_rsu]
        road_only = (f.gt_vis_rsu & ~f.gt_vis_ego)[f.gt_vis_ego | f.gt_vis_rsu]
        for k in cnt:
            pred = fuse(f.ego_xy, corrected[k])
            hp, hg = match_greedy(pred, gt_all, thr)
            cnt[k] += [hg.sum(), (~hp).sum(), (~hg).sum()]
            cnt_road[k] += [hg[road_only].sum(), 0, (~hg[road_only]).sum()]
    return cnt, cnt_road, err, no_pair / len(frames)


def sweep(seeds: int, delays=(0, 100, 200, 400), tracker_kw=None):
    res = {}
    for d in delays:
        agg = {k: np.zeros(3, int) for k in ["raw", "single", "comet", "oracle"]}
        agg_r = {k: np.zeros(3, int) for k in agg}
        errs = {"raw": [], "single": [], "comet": []}
        npairs = []
        for s in range(seeds):
            c, cr, e, npf = run_sequence(5000 + s, d, tracker_kw=tracker_kw)
            for k in agg:
                agg[k] += c[k]
                agg_r[k] += cr[k]
            for k in errs:
                errs[k] += e[k]
            npairs.append(npf)
        row = {}
        for k in agg:
            p, r, f = prf(*agg[k])
            rp, rr, _ = prf(*agg_r[k])
            row[k] = {"precision": p, "recall": r, "f1": f, "roadside_only_recall": rr}
        row["offset_rmse"] = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}
        row["frac_msgs_no_pair"] = float(np.mean(npairs))
        res[str(d)] = row
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="assets/results.json")
    a = ap.parse_args()
    res = sweep(a.seeds)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"{'delay':>6} | " + " | ".join(f"{k:>8}" for k in ["raw", "single", "comet", "oracle"]) + " | offset RMSE raw/single/comet | no-pair")
    for d, r in res.items():
        print(f"{d:>6} | " + " | ".join(f"{r[k]['f1']:8.3f}" for k in ["raw", "single", "comet", "oracle"])
              + f" | {r['offset_rmse']['raw']:.2f}/{r['offset_rmse']['single']:.2f}/{r['offset_rmse']['comet']:.2f}"
              + f" | {r['frac_msgs_no_pair']:.2f}")
    mean = {k: np.mean([r[k]['f1'] for r in res.values()]) for k in ["raw", "single", "comet", "oracle"]}
    print("mean F1@1.0m:", {k: round(v, 3) for k, v in mean.items()})
