"""Evaluation: open-loop multi-step prediction + closed-loop tracking, matched / mismatched / pooled data.

    python evaluate.py --study prediction     -> results/prediction.json
    python evaluate.py --study closedloop     -> results/closedloop.json
    python evaluate.py --study report         -> results/RESULTS.md
For each held-out seed s, fresh recordings are made at mu_high and mu_low (seed s for training data, s+5000 for
test data), so data, noise and excitation differ per seed.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from dpcf import vehicle as V
from dpcf.control import BoxQP, DPCController, ModelMPC
from dpcf.episode import collect, run
from dpcf.hankel import DPCPredictor, HankelData, multistep_rmse

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def build(seed: int, cfg=CFG):
    """Record training data at both frictions and build high / low / pooled predictors."""
    w, d, p = cfg["world"], cfg["data"], cfg["dpc"]
    rec = {mu: collect(V.Track(mu_high=mu, mu_low=mu), d["T"], seed=seed, amp=d["excitation"])
           for mu in (w["mu_high"], w["mu_low"])}
    kw = dict(lam=p["lam"], w_u=p["w_u"], w_y=p["w_y"])
    hd = HankelData([rec[w["mu_high"]]], p["T_ini"], p["N"])
    ld = HankelData([rec[w["mu_low"]]], p["T_ini"], p["N"])
    pd = HankelData([rec[w["mu_high"]], rec[w["mu_low"]]], p["T_ini"], p["N"])
    return dict(high=DPCPredictor(hd, **kw), low=DPCPredictor(ld, **kw), pool=DPCPredictor(pd, **kw)), hd.n_cols


def tracks(cfg=CFG):
    w = cfg["world"]
    return {"uniform_high": V.Track(mu_high=w["mu_high"], mu_low=w["mu_high"]),
            "uniform_low": V.Track(mu_high=w["mu_low"], mu_low=w["mu_low"]),
            "mixed": V.Track(mu_high=w["mu_high"], mu_low=w["mu_low"], wet=[tuple(x) for x in w["wet_sectors"]])}


def study_prediction(seeds):
    w = CFG["world"]
    out = {"per_seed": [], "seeds": seeds}
    for s in seeds:
        P, c_high = build(s)
        row = {}
        for mu, lab in ((w["mu_high"], "test_high"), (w["mu_low"], "test_low")):
            u, y = collect(V.Track(mu_high=mu, mu_low=mu), CFG["data"]["T"], seed=s + 5000, amp=CFG["data"]["excitation"])
            r = {}
            for k, pr in P.items():
                e = multistep_rmse(pr, u, y, stride=2)                     # (N, 2)
                r[k] = dict(ey_cm=float(e[:, 0].pow(2).mean().sqrt() * 100), epsi_deg=float(np.degrees(e[:, 1].pow(2).mean().sqrt())),
                            ey_curve_cm=(e[:, 0] * 100).tolist())
            # which data does the pooled predictor lean on?  share of |g| on the mu_high columns
            Ti, N = P["pool"].Ti, P["pool"].N
            idx = list(range(0, u.shape[0] - Ti - N, 10))
            wu = torch.stack([u[i:i + Ti + N] for i in idx]); wy = torch.stack([y[i:i + Ti + N] for i in idx])
            g = P["pool"].g(wu[:, :Ti], wy[:, :Ti], wu[:, Ti:])
            r["pool_share_high_cols"] = float((g[:, :c_high].abs().sum(1) / g.abs().sum(1)).mean())
            row[lab] = r
        out["per_seed"].append(row)
        print(s, {lab: {k: round(v["ey_cm"], 2) for k, v in row[lab].items() if isinstance(v, dict)} for lab in row},
              flush=True)
    return out


def study_closedloop(seeds):
    q = CFG["qp"]
    N = CFG["dpc"]["N"]
    qp = BoxQP(N, q=(q["q_ey"], q["q_epsi"]), r=q["r"], r_d=q["r_d"])
    thr = 0.5 * (CFG["world"]["mu_high"] + CFG["world"]["mu_low"])
    T = tracks()
    names = ["dpc_high", "dpc_low", "dpc_pool", "dpc_oracle", "model_mpc_true_mu"]
    res = {tn: {n: [] for n in names} for tn in T}
    for s in seeds:
        P, _ = build(s)
        for tn, tr in T.items():
            for n in names:
                if n == "model_mpc_true_mu":
                    c, kind = ModelMPC(N, qp), "model"
                elif n == "dpc_oracle":
                    c, kind = DPCController(P, lambda mu: "high" if mu > thr else "low", qp), "dpc"
                else:
                    c, kind = DPCController(P, n.split("_")[1], qp), "dpc"
                r = run(tr, c, kind, seed=s, laps=CFG["eval"]["laps"], Ti=CFG["dpc"]["T_ini"], N=N)
                res[tn][n].append(dict(rms=r["rms_ey"], max=r["max_ey"], off=r["off_track"]))
        print(s, {tn: {n: round(res[tn][n][-1]["rms"] * 100, 2) for n in names} for tn in T}, flush=True)
    return {"seeds": seeds, "per_seed": res}


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report():
    pr = json.load(open(ROOT / "results/prediction.json"))
    cl = json.load(open(ROOT / "results/closedloop.json"))
    L = ["# Results (synthetic toy world; held-out seeds %d-%d)\n" % (pr["seeds"][0], pr["seeds"][-1])]
    L.append("## Open-loop multi-step prediction (horizon 0.75 s), RMS lateral error e_y [cm], mean over seeds\n")
    L.append("| Test friction | matched data | mismatched data | pooled data | mismatched / matched | pooled / matched | pooled share of |g| on mu_high columns |")
    L.append("|---|---|---|---|---|---|---|")
    summ = {}
    for lab, mk, mm in (("test_high", "high", "low"), ("test_low", "low", "high")):
        a = np.array([[r[lab][mk]["ey_cm"], r[lab][mm]["ey_cm"], r[lab]["pool"]["ey_cm"]] for r in pr["per_seed"]])
        sh = np.mean([r[lab]["pool_share_high_cols"] for r in pr["per_seed"]])
        m = a.mean(0)
        rat_mm, rat_p = (a[:, 1] / a[:, 0]), (a[:, 2] / a[:, 0])
        summ[lab] = dict(matched=m[0], mismatched=m[1], pooled=m[2], r_mm=rat_mm.mean(), r_p=rat_p.mean(), share=sh)
        L.append(f"| {'mu = 1.0 (dry)' if lab == 'test_high' else 'mu = 0.5 (low)'} | {m[0]:.2f} | {m[1]:.2f} | {m[2]:.2f} | "
                 f"{rat_mm.mean():.2f}x | {rat_p.mean():.2f}x | {sh:.2f} |")
    L.append("\nPaper (abstract): pooled within 1.5x of matched; mismatched 2.2x-3.4x.\n")
    L.append("## Closed-loop tracking, 2 laps, RMS |e_y| [cm] (mean, 95% bootstrap CI) / mean max |e_y| [cm] / off-track runs\n")
    names = ["dpc_high", "dpc_low", "dpc_pool", "dpc_oracle", "model_mpc_true_mu"]
    L.append("| Track | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    csumm = {}
    for tn, d in cl["per_seed"].items():
        cells = []
        csumm[tn] = {}
        for n in names:
            rms = np.array([x["rms"] for x in d[n]]) * 100
            mx = np.array([x["max"] for x in d[n]]) * 100
            off = sum(x["off"] for x in d[n])
            lo, hi = boot_ci(rms)
            csumm[tn][n] = dict(rms=rms.mean(), lo=lo, hi=hi, max=mx.mean(), off=off)
            cells.append(f"{rms.mean():.2f} [{lo:.2f}, {hi:.2f}] / {mx.mean():.1f} / {off}")
        L.append(f"| {tn} | " + " | ".join(cells) + " |")
    # paired comparison on the mixed track
    d = cl["per_seed"]["mixed"]
    pool = np.array([x["rms"] for x in d["dpc_pool"]]); best_single = np.minimum(
        np.array([x["rms"] for x in d["dpc_high"]]), np.array([x["rms"] for x in d["dpc_low"]]))
    hi_ = np.array([x["rms"] for x in d["dpc_high"]])
    L.append(f"\nMixed track, paired: pooled beats high-only data in {(pool < hi_).sum()}/{len(pool)} seeds and beats the "
             f"better of the two single data sets (chosen in hindsight per seed) in {(pool < best_single).sum()}/{len(pool)}.")
    L.append("\nPaper (abstract): sim tracking DPC pooled 13.7 cm vs NMPC 13.3 cm; CPM Lab low-mu 1.6 vs 1.5 cm, high-mu 0.9 vs 1.1 cm.")
    (ROOT / "results/RESULTS.md").write_text("\n".join(L) + "\n")
    json.dump(dict(prediction=summ, closedloop=csumm), open(ROOT / "results/summary.json", "w"), indent=1, default=float)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="prediction", choices=["prediction", "closedloop", "report"])
    ap.add_argument("--seeds", type=int, nargs=2, default=CFG["eval"]["seeds"])
    a = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    seeds = list(range(*a.seeds))
    t0 = time.time()
    if a.study == "prediction":
        json.dump(study_prediction(seeds), open(ROOT / "results/prediction.json", "w"))
    elif a.study == "closedloop":
        json.dump(study_closedloop(seeds), open(ROOT / "results/closedloop.json", "w"))
    else:
        report()
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
