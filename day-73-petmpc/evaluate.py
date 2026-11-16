"""Closed-loop + open-loop evaluation of the PET-MPC toy re-implementation.

    python evaluate.py --study obstacle   # paired-seed study, one moving generic obstacle (held-out seeds 1000+)
    python evaluate.py --study crowd      # crowd navigation + semantic ablation (pedestrian-aware vs generic)
    python evaluate.py --study prediction # open-loop ADE/FDE and empirical coverage of the Markov margins
    python evaluate.py --study report     # merge results/*.json into results/RESULTS.md

Seeds 0-119 were used while developing; every reported number uses seeds >= 1000.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from petmpc.episode import run_episode, prediction_study

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
SEED0 = 1000


def _job(args):
    import torch
    torch.set_num_threads(1)
    scenario, seed, method, aware, n_ped = args
    r = run_episode(scenario, seed, method, CFG, aware=aware, n_ped=n_ped)
    return r


def summarise(rs):
    a = lambda k: np.array([r[k] for r in rs], float)
    reached = a("reached").astype(bool)
    return dict(n=len(rs), contact_rate=float(a("contact").mean()), contact_time_frac=float(a("contact_frac").mean()),
                ps_overlap_frac=float(a("ps_frac").mean()), mean_time=float(a("time").mean()),
                mean_time_reached=float(a("time")[reached].mean()) if reached.any() else float("nan"),
                success=float(reached.mean()), min_clear_p05=float(np.percentile(a("min_clear"), 5)))


def paired(ra, rb):
    """Paired comparison on identical seeds: how many runs flip contact / no-contact, mean time difference."""
    ca = np.array([r["contact"] for r in ra]); cb = np.array([r["contact"] for r in rb])
    ta = np.array([r["time"] for r in ra]); tb = np.array([r["time"] for r in rb])
    both = np.array([r["reached"] for r in ra]).astype(bool) & np.array([r["reached"] for r in rb]).astype(bool)
    return dict(a_only=int((ca & ~cb.astype(bool)).sum()), b_only=int((cb & ~ca.astype(bool)).sum()),
                both=int((ca & cb).sum()), dtime_mean=float((tb - ta)[both].mean()),
                dtime_ci95=float(1.96 * (tb - ta)[both].std() / np.sqrt(both.sum())))


def run(jobs):
    with Pool(2) as p:
        return p.map(_job, jobs, chunksize=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    if a.study == "obstacle":
        n = a.seeds or CFG["eval"]["obstacle_seeds"]
        seeds = range(SEED0, SEED0 + n)
        methods = ["cv_nominal", "koopman_nominal", "koopman_fixed", "pet_mpc"]
        R = {m: run([("obstacle", s, m, True, 0) for s in seeds]) for m in methods}
        out = dict(study="obstacle", seeds=[SEED0, SEED0 + n - 1], summary={m: summarise(R[m]) for m in methods},
                   paired_pet_vs_nominal=paired(R["koopman_nominal"], R["pet_mpc"]),
                   paired_pet_vs_fixed=paired(R["koopman_fixed"], R["pet_mpc"]),
                   per_seed={m: [r["contact"] for r in R[m]] for m in methods})
    elif a.study == "crowd":
        n = a.seeds or CFG["eval"]["crowd_seeds"]
        npd = CFG["eval"]["n_ped"]
        seeds = range(SEED0, SEED0 + n)
        variants = {"cv_nominal": ("cv_nominal", True), "koopman_nominal": ("koopman_nominal", True),
                    "koopman_fixed": ("koopman_fixed", True), "pet_mpc": ("pet_mpc", True),
                    "pet_mpc_generic": ("pet_mpc", False)}
        R = {k: run([("crowd", s, m, aw, npd) for s in seeds]) for k, (m, aw) in variants.items()}
        out = dict(study="crowd", n_ped=npd, seeds=[SEED0, SEED0 + n - 1],
                   summary={k: summarise(v) for k, v in R.items()},
                   paired_pet_vs_nominal=paired(R["koopman_nominal"], R["pet_mpc"]),
                   paired_aware_vs_generic=paired(R["pet_mpc_generic"], R["pet_mpc"]))
    elif a.study == "prediction":
        out = dict(study="prediction")
        for sc in ("obstacle", "crowd"):
            out[sc] = prediction_study(CFG, sc, range(SEED0, SEED0 + 20), T=300, n_ped=CFG["eval"]["n_ped"])
    elif a.study == "report":
        report()
        return
    else:
        raise ValueError(a.study)
    out["wall_s"] = time.time() - t0
    json.dump(out, open(RES / f"{a.study}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=1)[:4000])


def report():
    L = ["# Results (synthetic toy world, held-out seeds >= 1000)\n"]
    names = {"cv_nominal": "Const-velocity, no margin", "koopman_nominal": "Koopman, no margin (nominal)",
             "koopman_fixed": "Koopman + fixed 0.30 m margin", "pet_mpc": "**PET-MPC** (Koopman + Markov residual margins)",
             "pet_mpc_generic": "PET-MPC, pedestrians treated as generic obstacles"}
    for st in ("obstacle", "crowd"):
        f = RES / f"{st}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        L.append(f"\n## {st} study ({d['summary']['pet_mpc']['n']} paired seeds)\n")
        L.append("| Controller | Runs with contact | Contact time | Personal-space overlap time | Mean time to goal (s) | Success | 5th pct min clearance (m) |")
        L.append("|---|---|---|---|---|---|---|")
        for k, s in d["summary"].items():
            L.append(f"| {names[k]} | {100*s['contact_rate']:.2f}% | {100*s['contact_time_frac']:.2f}% | "
                     f"{100*s['ps_overlap_frac']:.2f}% | {s['mean_time']:.2f} | {100*s['success']:.0f}% | {s['min_clear_p05']:.2f} |")
        for k, v in d.items():
            if k.startswith("paired"):
                L.append(f"\n{k}: contact only in A = {v['a_only']}, only in B = {v['b_only']}, both = {v['both']}; "
                         f"time B - A = {v['dtime_mean']:+.2f} ± {v['dtime_ci95']:.2f} s (95% CI, runs where both reached)")
    f = RES / "prediction.json"
    if f.exists():
        d = json.load(open(f))
        L.append("\n## Open-loop prediction (20 seeds x 300 steps, 2 s horizon)\n")
        L.append("| Scenario | Predictor | ADE (m) | FDE @2 s (m) | Error @1 s | Markov margin @1 s | Empirical coverage (target >= 75%) |")
        L.append("|---|---|---|---|---|---|---|")
        for sc in ("obstacle", "crowd"):
            for k, v in d[sc].items():
                L.append(f"| {sc} | {k} | {v['ADE']:.3f} | {v['FDE']:.3f} | {v['err_by_step'][9]:.3f} | "
                         f"{v['margin_by_step'][9]:.2f} | {100*v['coverage']:.1f}% |")
    (RES / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
