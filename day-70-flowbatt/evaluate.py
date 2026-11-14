"""Evaluate all models on the held-out synthetic test cells.

    python evaluate.py --seed 0 [--models flowbatt diffbatt transformer mlp rf]

Metrics (mirroring the paper's protocol as far as I could read it):
  traj_rmse   mean per-cell RMSE of SOH (percentage points) over the TRUE trajectory up to its 80 % EOL
              (generative models: element-wise median of N_POINT samples)
  rul_mae     |EOL_pred - EOL_true| in cycles (EOL = first crossing of 80 %), and rul_mape in %
  cov90       fraction of reference SOH points (up to EOL) inside the 5-95 % band of 100 samples (nominal 0.90)
  width90     mean band width (SOH %-points)
  rul_cov90   fraction of cells whose true EOL lies inside the 5-95 % range of per-sample EOLs
  nfe sweep   FlowBatt traj_rmse vs number of Euler steps
  latency     ms for one cell x 100 samples, CPU, 1 thread
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestRegressor

from flowbatt import world as W
from flowbatt.data import Normaliser, load_splits, tensors
from train import build

N_POINT = 20          # samples for the median point forecast (all test cells)
N_INT = 100           # samples for intervals (paper: 100)
N_INT_CELLS = 96      # subset used for the 100-sample interval metrics (CPU budget)


def rf_features(fleet):
    cap = fleet["cap"]
    dq = cap[:, 99] - cap[:, 9]                                          # Severson-style Delta-Q_100-10(V)
    scal = np.column_stack([np.log(np.var(dq, 1) + 1e-12), np.log(np.abs(dq.min(1)) + 1e-9), dq.mean(1),
                            np.argmin(dq, 1)])
    return np.column_stack([scal, cap[:, 9, ::10], 1 - fleet["prefix"]])          # (B, 4 + 10 + 10)


def point_metrics(pred_soh, fleet):
    """pred_soh (B, L) -> dict."""
    soh, clean = fleet["soh"], fleet["soh_clean"]
    eol_t = W.eol_cycle(clean)
    mask = W.GRID[None] <= eol_t[:, None]
    rmse = np.sqrt(((pred_soh - soh) ** 2 * mask).sum(1) / mask.sum(1)) * 100
    eol_p = W.eol_cycle(pred_soh)
    return dict(traj_rmse=float(rmse.mean()), traj_rmse_sd_cells=float(rmse.std()),
                rul_mae=float(np.abs(eol_p - eol_t).mean()),
                rul_mape=float((np.abs(eol_p - eol_t) / eol_t).mean() * 100)), rmse


def interval_metrics(samp_soh, fleet):
    """samp_soh (B, n, L)."""
    soh, clean = fleet["soh"], fleet["soh_clean"]
    eol_t = W.eol_cycle(clean)
    mask = W.GRID[None] <= eol_t[:, None]
    lo, hi = np.percentile(samp_soh, 5, 1), np.percentile(samp_soh, 95, 1)
    inside = (soh >= lo) & (soh <= hi)
    eols = W.eol_cycle(samp_soh)                                        # (B, n)
    elo, ehi = np.percentile(eols, 5, 1), np.percentile(eols, 95, 1)
    hid = np.abs(fleet["hidden"]) > 1.0
    rul_in = (eol_t >= elo) & (eol_t <= ehi)
    return dict(cov90=float((inside * mask).sum() / mask.sum()),
                width90=float(((hi - lo) * mask).sum() / mask.sum() * 100),
                rul_cov90=float(rul_in.mean()),
                rul_cov90_hidden_defect=float(rul_in[hid].mean()) if hid.any() else float("nan"),
                rul_cov90_typical=float(rul_in[~hid].mean()))


def sample_soh(m, norm, cap, pre, n, chunk=16, **kw):
    outs = []
    for i in range(0, len(cap), chunk):
        x = m.sample(cap[i:i + chunk], pre[i:i + chunk], n, **kw)
        outs.append(norm.inv(x.numpy()))
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["flowbatt", "diffbatt", "transformer", "mlp", "rf"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    torch.set_num_threads(int(__import__("os").environ.get("NTHREADS", "2")))
    sp = load_splits(cfg)
    norm = Normaliser(sp["train"]["soh"])
    te = sp["test"]
    cap, pre, _ = tensors(te, norm)
    sub = {k: v[:N_INT_CELLS] for k, v in te.items()}
    res = {}
    for name in a.models:
        t0 = time.time()
        if name == "rf":
            rf = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=1, random_state=a.seed)
            rf.fit(rf_features(sp["train"]), norm.fwd(sp["train"]["soh"]))
            pred = norm.inv(rf.predict(rf_features(te)))
            r, _ = point_metrics(pred, te)
            r["params"] = "300 trees"
            res[name] = r
            print(name, r, flush=True)
            continue
        m = build(name, cfg).eval()
        m.load_state_dict(torch.load(f"checkpoints/{name}_s{a.seed}.pt"))
        gen = name in ("flowbatt", "diffbatt")
        kw = dict(steps=cfg["sample"]["flow_steps"]) if name == "flowbatt" else {}
        torch.manual_seed(1000 + a.seed)
        if gen:
            s = sample_soh(m, norm, cap, pre, N_POINT, **kw)                    # (B, 20, L)
            r, _ = point_metrics(np.median(s, 1), te)
            r["traj_rmse_mean_of_samples"] = point_metrics(s.mean(1), te)[0]["traj_rmse"]
            r["traj_rmse_single_sample"] = point_metrics(s[:, 0], te)[0]["traj_rmse"]
            s100 = sample_soh(m, norm, cap[:N_INT_CELLS], pre[:N_INT_CELLS], N_INT, **kw)
            r.update(interval_metrics(s100, sub))
            # latency: 1 cell x 100 samples, 1 thread
            torch.set_num_threads(1)
            ts = []
            for _ in range(3):
                t1 = time.perf_counter(); m.sample(cap[:1], pre[:1], 100, **kw); ts.append(time.perf_counter() - t1)
            r["latency_ms_100samples_1thread"] = float(np.median(ts) * 1000)
            torch.set_num_threads(int(__import__("os").environ.get("NTHREADS", "2")))
            if name == "flowbatt":
                r["nfe_sweep"] = {}
                for st in (1, 2, 4, 8, 16, 32, 64):
                    torch.manual_seed(77)
                    s8 = sample_soh(m, norm, cap[:N_INT_CELLS], pre[:N_INT_CELLS], N_POINT, steps=st)
                    r["nfe_sweep"][st] = point_metrics(np.median(s8, 1), sub)[0]["traj_rmse"]
                torch.manual_seed(77)
                sh = sample_soh(m, norm, cap[:N_INT_CELLS], pre[:N_INT_CELLS], N_POINT, steps=8, method="heun")
                r["heun8_traj_rmse"] = point_metrics(np.median(sh, 1), sub)[0]["traj_rmse"]
        else:
            with torch.no_grad():
                pred = norm.inv(m.predict(cap, pre).numpy())
            r, _ = point_metrics(pred, te)
        r["params"] = sum(p.numel() for p in m.parameters())
        r["eval_sec"] = round(time.time() - t0, 1)
        res[name] = r
        print(name, json.dumps(r), flush=True)
    out = a.out or f"checkpoints/metrics_s{a.seed}_{'_'.join(a.models)}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
