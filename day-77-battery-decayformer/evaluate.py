"""Evaluate all variants on their held-out aging-condition folds -> results/summary.json + results/RESULTS.md.

Every cell is tested exactly once (by the model of the fold that held its condition out).

Questions (from the abstract of arXiv 2611.19116; Q2-Q4 are my ablations, the abstract reports none):
    Q1  does the compact hybrid, with one fixed config, forecast full SOH trajectories well on unseen conditions,
        and does it need a scaled model? (DecayFormer vs a ~5x larger Transformer and a GRU)
    Q2  master-curve head vs an unconstrained head (same encoder)
    Q3  SOC-interval statistics vs scalar features only
    Q4  state-query attention vs recurrence alone
    Ext prefix length (25 / 50 / 100 cycles), knee-dominated cells, monotonicity, CPU latency
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from dfm.baselines import sqrt_fit
from dfm.data import N_MAX, condition_folds, conditions, make_cell, soh_curve
from dfm.model import VARIANTS, build, n_params
from train import CFG, ckpt_path, load_data, split

ROOT = Path(__file__).resolve().parent
NAMES = list(VARIANTS)


def load_model(variant: str, fold: int, seed: int = 0):
    m = build(variant)
    m.load_state_dict(torch.load(ckpt_path(variant, fold, seed), map_location="cpu"))
    m.eval()
    return m


@torch.no_grad()
def predict(m, variant: str, x: np.ndarray, k: int, grid: np.ndarray):
    X = torch.from_numpy(x[:, :, VARIANTS[variant]["feats"]])
    valid = torch.arange(X.shape[1])[None].expand(X.shape[0], -1) < k
    out = m(X, valid)
    return m.head.soh(out, torch.from_numpy(grid)).numpy(), m.head.lifetime(out).numpy(), out


def traj_metrics(pred: np.ndarray, traj: np.ndarray, grid: np.ndarray, life_p: np.ndarray, life_t: np.ndarray):
    ok = ~np.isnan(traj)
    ape = np.where(ok, np.abs(pred - traj) / np.where(ok, traj, 1.0), np.nan)
    late = ok & (np.nan_to_num(traj, nan=1.0) < 0.9)
    ape_late = np.where(late, ape, np.nan)
    # monotonicity: any increase > 1e-4 between consecutive grid points before the true EOL
    inc = np.diff(pred, axis=1) > 1e-4
    viol = (inc & ok[:, 1:]).any(1)
    return dict(traj=np.nanmean(ape, 1) * 100, late=np.nanmean(ape_late, 1) * 100,
                life=np.abs(life_p - life_t) / life_t * 100, viol=viol.astype(float))


def knee_fraction(d: dict) -> np.ndarray:
    """Share of the 20 % SOH loss at EOL that comes from the knee term (recomputed from the hidden cell params)."""
    out = []
    for i, (ci, life) in enumerate(zip(d["cond"], d["life"])):
        k = i % CFG["data"]["cells_per_cond"]
        cell = make_cell(conditions()[ci], CFG["data"]["seed"] * 100_003 + ci * 101 + k)
        out.append(cell["k_knee"] * max(0.0, life - cell["n_k"]) ** 2 / 0.2)
    return np.asarray(out)


def boot_diff(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0):
    """Mean(a - b) with a 95 % CI from a bootstrap over CONDITIONS (cells of a condition move together)."""
    r = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.nonzero(groups == g)[0] for g in ug}
    diff = a - b
    bs = []
    for _ in range(n):
        gs = r.choice(ug, len(ug))
        bs.append(diff[np.concatenate([idx[g] for g in gs])].mean())
    return dict(mean=float(diff.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def main():
    torch.set_num_threads(2)
    d = load_data()
    grid = d["grid"]
    M = len(d["life"])
    nf = CFG["data"]["n_folds"]
    seed = CFG["train"]["seeds"][0]
    ks = CFG["eval"]["k_ext"]
    res = {k: {v: dict(pred=np.zeros((M, len(grid))), life=np.zeros(M)) for v in NAMES + ["sqrt_fit"]} for k in ks}
    fold_of = np.zeros(M, int)
    for f in range(nf):
        _, te = split(d, f)
        fold_of[te] = f
        for v in NAMES:
            m = load_model(v, f, seed)
            for k in ks:
                p, L, _ = predict(m, v, d["x"][te], k, grid)
                res[k][v]["pred"][te], res[k][v]["life"][te] = p, L
        for k in ks:
            p, L = sqrt_fit(d["x"][te], k, grid)
            res[k]["sqrt_fit"]["pred"][te], res[k]["sqrt_fit"]["life"][te] = p, L
    kf = knee_fraction(d)
    knee_dom = kf > 0.25
    groups = d["cond"]
    S = dict(n_cells=int(M), n_conditions=len(conditions()), n_folds=nf, knee_dominated=int(knee_dom.sum()),
             params={v: n_params(build(v)) for v in NAMES}, metrics={}, per_fold={}, diffs={}, knee={}, prefix={})
    per_cell = {}
    for k in ks:
        for v in NAMES + ["sqrt_fit"]:
            mt = traj_metrics(res[k][v]["pred"], d["traj"], grid, res[k][v]["life"], d["life"])
            per_cell[(k, v)] = mt
            S["prefix"].setdefault(v, {})[str(k)] = dict(traj=float(mt["traj"].mean()), life=float(mt["life"].mean()))
    km = CFG["eval"]["k_main"]
    for v in NAMES + ["sqrt_fit"]:
        mt = per_cell[(km, v)]
        S["metrics"][v] = {key: float(val.mean()) for key, val in mt.items()}
        S["metrics"][v]["late"] = float(np.nanmean(mt["late"]))
        S["per_fold"][v] = [float(mt["traj"][fold_of == f].mean()) for f in range(nf)]
        S["knee"][v] = dict(knee_traj=float(mt["traj"][knee_dom].mean()), knee_life=float(mt["life"][knee_dom].mean()),
                            mild_traj=float(mt["traj"][~knee_dom].mean()), mild_life=float(mt["life"][~knee_dom].mean()))
    ref = per_cell[(km, "decayformer")]
    for v in NAMES[1:] + ["sqrt_fit"]:
        o = per_cell[(km, v)]
        S["diffs"][v] = dict(traj=boot_diff(ref["traj"], o["traj"], groups), life=boot_diff(ref["life"], o["life"], groups))
    # CPU latency, batch 1, 100 cycles
    lat = {}
    x1 = d["x"][:1]
    for v in ["decayformer", "transformer", "gru"]:
        m = load_model(v, 0, seed)
        predict(m, v, x1, 100, grid)
        t0 = time.perf_counter()
        for _ in range(30):
            predict(m, v, x1, 100, grid)
        lat[v] = (time.perf_counter() - t0) / 30 * 1000
    S["latency_ms"] = lat
    out = ROOT / "results"
    json.dump(S, open(out / "summary.json", "w"), indent=1)
    np.savez_compressed(out / "predictions_k100.npz", **{v: res[km][v]["pred"] for v in NAMES + ["sqrt_fit"]},
                        **{v + "_life": res[km][v]["life"] for v in NAMES + ["sqrt_fit"]}, fold=fold_of, knee_frac=kf)
    write_md(S)
    print(json.dumps({v: {k: round(x, 2) for k, x in S["metrics"][v].items()} for v in S["metrics"]}, indent=0))


def write_md(S: dict) -> None:
    L = ["# Full results (synthetic toy, 288 cells, 48 aging conditions, 6 condition-exclusive folds)\n",
         "Each cell is forecast by the model of the fold that held its condition out. One run per fold, one fixed "
         "config (the paper's protocol). Forecast from the first 100 cycles unless stated.\n",
         "| Variant | params | traj MAPE % | late-life MAPE % (SOH<0.9) | lifetime MAPE % | non-monotone curves |",
         "|---|---|---|---|---|---|"]
    for v, m in S["metrics"].items():
        p = f"{S['params'][v]:,}" if v in S["params"] else "4 (fit)"
        L.append(f"| {v} | {p} | {m['traj']:.2f} | {m['late']:.2f} | {m['life']:.1f} | {m['viol'] * 100:.0f}% |")
    L += ["\n## DecayFormer minus other (negative = DecayFormer better), 95% CI by condition bootstrap\n",
          "| vs | traj MAPE (pts) | lifetime MAPE (pts) |", "|---|---|---|"]
    for v, dd in S["diffs"].items():
        t, l = dd["traj"], dd["life"]
        L.append(f"| {v} | {t['mean']:+.2f} ({t['lo']:+.2f} .. {t['hi']:+.2f}) | {l['mean']:+.1f} ({l['lo']:+.1f} .. {l['hi']:+.1f}) |")
    L += [f"\n## Knee-dominated cells (knee > 25% of the fade at EOL, n = {S['knee_dominated']}) vs the rest\n",
          "| Variant | knee traj | knee life | other traj | other life |", "|---|---|---|---|---|"]
    for v, m in S["knee"].items():
        L.append(f"| {v} | {m['knee_traj']:.2f} | {m['knee_life']:.1f} | {m['mild_traj']:.2f} | {m['mild_life']:.1f} |")
    L += ["\n## Observed prefix length (traj MAPE % / lifetime MAPE %)\n", "| Variant | 25 cycles | 50 cycles | 100 cycles |",
          "|---|---|---|---|"]
    for v, m in S["prefix"].items():
        L.append(f"| {v} | " + " | ".join(f"{m[k]['traj']:.2f} / {m[k]['life']:.1f}" for k in ["25", "50", "100"]) + " |")
    L += ["\n## Per-fold traj MAPE % (one fixed config, no per-fold tuning)\n",
          "| Variant | " + " | ".join(f"f{i}" for i in range(S["n_folds"])) + " |", "|---|" + "---|" * S["n_folds"]]
    for v, pf in S["per_fold"].items():
        L.append(f"| {v} | " + " | ".join(f"{x:.2f}" for x in pf) + " |")
    L.append("\nCPU latency per cell (batch 1, 100 cycles): " +
             ", ".join(f"{k} {v:.1f} ms" for k, v in S["latency_ms"].items()))
    open(ROOT / "results" / "RESULTS.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
