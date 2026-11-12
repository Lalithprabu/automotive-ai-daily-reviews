"""Evaluate every trained variant on the synthetic test split.

Metrics
  * cube mAP @ IoU 0.30  (axis-aligned boxes in range-bin x azimuth-bin space, VOC-style all-point AP, mean over 3 classes)
  * radial-velocity MAE (m/s) on true positives, overall and split into wrapped (|k|>0) vs unwrapped objects
  * Doppler-wrapping accuracy: predicted wrap index k == true k, on true positives
  * forward-velocity MAE (m/s), ego-speed MAE (m/s, RaMOD only), params, CPU latency (batch 1, 1 thread)

    python evaluate.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ramod.data import RadarSeq
from ramod.losses import decode
from ramod.model import n_params
from ramod.world import CLASSES, S_MAX, S_RES, wrap_v
from train import VARIANTS, build, splits


def iou(a, b):
    r0, a0 = max(a[0], b[0]), max(a[1], b[1])
    r1, a1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, r1 - r0) * max(0.0, a1 - a0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-9)


def ap_voc(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    o = np.argsort(-conf)
    tp = tp[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


@torch.no_grad()
def run(model, ds, thr_iou=0.3, score_thresh=0.05):
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    per_cls = {c: dict(tp=[], conf=[], n=0) for c in range(3)}
    vel, ego_err = [], []
    near_hit, far_hit = [], []          # GT recall split: inside vs outside the static-Doppler band
    for cubes, heat, reg, mask, ego, idx in dl:
        out = model(cubes)
        dets = decode(out, model.vmax, thresh=score_thresh, wrap_head=model.wrap_head)
        if "v_e" in out:
            ego_err += (out["v_e"][:, -1] - ego[:, -1]).abs().tolist()
        for b, i in enumerate(idx.tolist()):
            gts = ds.objects(i)
            for c in range(3):
                per_cls[c]["n"] += sum(g["cls"] == c for g in gts)
            used, used30 = set(), set()      # used30: matched by a detection with score >= 0.3 (for recall)
            v_e_true = float(ego[b, -1])
            for d in sorted(dets[b], key=lambda d: -d["score"]):
                best, bj = 0.0, -1
                for j, g in enumerate(gts):
                    if g["cls"] != d["cls"] or j in used:
                        continue
                    v = iou(d["box"], g["box"])
                    if v > best:
                        best, bj = v, j
                hit = best >= thr_iou
                per_cls[d["cls"]]["tp"].append(float(hit)); per_cls[d["cls"]]["conf"].append(d["score"])
                if hit:
                    used.add(bj)
                    if d["score"] >= 0.3:
                        used30.add(bj)
                    g = gts[bj]
                    vel.append((abs(d["v_rad"] - g["v_rad"]), d["wrapped"] == (g["k"] != 0), g["k"] != 0,
                                abs(d["v_fwd"] - g["v_fwd"]), d["p_wrap"]))
            for j, g in enumerate(gts):
                # does this mover's measured Doppler sit inside the band the static mask removes?
                s_az = g["ab"] * S_RES - S_MAX
                v_stat = float(wrap_v(np.array(-v_e_true * np.sqrt(max(0.0, 1 - s_az ** 2)))))
                near = abs(float(wrap_v(np.array(g["v_meas"] - v_stat)))) < 0.5
                (near_hit if near else far_hit).append(j in used30)
    aps = {CLASSES[c]: ap_voc(np.array(v["tp"]), np.array(v["conf"]), v["n"]) for c, v in per_cls.items()}
    V = np.array(vel, dtype=float) if vel else np.zeros((0, 5))
    wr = V[:, 2] > 0
    return dict(mAP=float(np.nanmean(list(aps.values()))), AP=aps,
                v_mae=float(V[:, 0].mean()), v_mae_wrapped=float(V[wr, 0].mean()) if wr.any() else None,
                v_mae_unwrapped=float(V[~wr, 0].mean()), wrap_acc=float(V[:, 1].mean()),
                wrap_acc_wrapped=float(V[wr, 1].mean()) if wr.any() else None,
                fwd_mae=float(V[:, 3].mean()), n_tp=int(len(V)), n_tp_wrapped=int(wr.sum()),
                ego_mae=float(np.mean(ego_err)) if ego_err else None,
                recall_near_static=float(np.mean(near_hit)) if near_hit else None, n_near_static=len(near_hit),
                recall_other=float(np.mean(far_hit)) if far_hit else None)


def latency(model, T):
    torch.set_num_threads(1)
    x = torch.randn(1, T, 40, 40, 32)   # (B, T, R, A, D)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        ts = []
        for _ in range(30):
            t0 = time.perf_counter(); model(x); ts.append(1000 * (time.perf_counter() - t0))
    torch.set_num_threads(2)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=list(VARIANTS))
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    torch.set_num_threads(2)
    _, te = splits(cfg)
    res = {}
    for name in a.models:
        m = build(name, cfg).eval()
        m.load_state_dict(torch.load(f"checkpoints/{name}_s{a.seed}.pt"))
        r = run(m, RadarSeq(te, T=m.T, first=2), cfg["eval"]["iou"])      # identical 950 test frames for all
        r.update(params=n_params(m), cpu_ms=latency(m, m.T))
        res[name] = r
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "AP"}),
              {k: round(v, 3) for k, v in r["AP"].items()}, flush=True)
    json.dump(res, open(f"checkpoints/metrics_s{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
