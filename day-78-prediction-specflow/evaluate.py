"""Evaluate every trained variant on the 4,000 held-out test scenes -> results/summary.json + results/RESULTS.md.

Metrics (per agent, against the NOISE-FREE future, best-of-20 like the paper):
    minADE / minFDE [m], miss rate (minFDE > 2 m)
    kinematic realism of the 20 samples, vehicles only:
        RMS jerk [m/s^3] (truth: computed on the clean track) and share of samples with any |a| > 6 m/s^2
    diversity: mean distance of the 20 sample endpoints from their centroid
Questions:
    Q1 acceleration-spectrum state vs position state (pos_state) and vs time-domain acceleration (acc_time)
    Q2 joint phase-space loss (no_phase)
    Q3 endpoint conditioning (no_endpoint), top-1 vs confidence-stratified endpoints (inference-only extension)
    Ext few-step sampling: 1 / 2 / 5 / 10 Euler steps
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dsf.data import DT, PED, VEH, MANEUVERS
from dsf.model import VARIANTS, build
from train import CFG, load_data, to_batch

ROOT = Path(__file__).resolve().parent
SEEDS = (0, 1)


def load_model(variant: str, seed: int):
    ck = torch.load(ROOT / f"results/ckpt/{variant}_s{seed}.pt", map_location="cpu")
    m = build(variant)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["params"]


def run(m, d: dict, H=20, steps=10, ep_mode="top1", chunk=250, seed=123) -> dict:
    g = torch.Generator().manual_seed(seed)
    P, A_ = [], []
    for s in range(0, len(d["mask"]), chunk):
        b = to_batch(d, np.arange(s, min(s + chunk, len(d["mask"]))))
        o = m.sample(b, H=H, steps=steps, ep_mode=ep_mode, generator=g)
        P.append(o["p"].numpy()); A_.append(o["a"].numpy())
    return dict(p=np.concatenate(P), a=np.concatenate(A_))            # (S, H, A, T, 2)


def cv_pred(d: dict) -> np.ndarray:
    v = d["hist"][:, :, -2:, 2:].mean(2)                                 # mean of the last 2 velocities
    k = np.arange(1, 13)[None, None, :, None] * DT
    return (v[:, :, None] * k)[:, None]                                  # (S, 1, A, T, 2)


def metrics(p: np.ndarray, a: np.ndarray | None, d: dict) -> dict:
    """p (S, H, A, T, 2) -> per-agent arrays (flattened over valid agents)."""
    gt = d["fut_clean"][:, None]
    err = np.linalg.norm(p - gt, axis=-1)                                # (S, H, A, T)
    ade = err.mean(-1).min(1)                                            # (S, A)
    fde = err[..., -1].min(1)
    endp = p[..., -1, :]
    div = np.linalg.norm(endp - endp.mean(1, keepdims=True), axis=-1).mean(1)
    out = dict(ade=ade, fde=fde, div=div)
    if a is not None:
        jerk = np.diff(a, axis=-2) / DT
        out["jerk"] = np.sqrt((jerk ** 2).sum(-1).mean(-1)).mean(1)      # (S, A) mean over samples of RMS jerk
        out["infeas"] = (np.linalg.norm(a, axis=-1).max(-1) > 6.0).mean(1)
    return out


def truth_kin(d: dict) -> dict:
    c = d["clean"][:, :, 6:]                                             # last 2 obs + future
    a = np.diff(c, 2, axis=2) / DT ** 2
    j = np.diff(a, axis=2) / DT
    return dict(jerk=np.sqrt((j ** 2).sum(-1).mean(-1)), infeas=(np.linalg.norm(a, axis=-1).max(-1) > 6.0).astype(float))


def boot_ci(x: np.ndarray, n=2000, seed=0):
    r = np.random.default_rng(seed)
    bs = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.mean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    torch.set_num_threads(2)
    D = load_data()
    te = D["test"]
    mask = te["mask"]
    veh, ped = mask & (te["kind"] == VEH), mask & (te["kind"] == PED)
    R, params, per_scene = {}, {}, {}
    raw = {}
    for v in VARIANTS:
        mets = []
        for s in SEEDS:
            m, params[v] = load_model(v, s)
            o = run(m, te)
            mets.append(metrics(o["p"], o["a"], te))
            if v == "specflow" and s == 0:
                np.savez_compressed(ROOT / "results/specflow_s0_test_samples.npz", p=o["p"][:400].astype(np.float16))
        raw[v] = {k: np.mean([mm[k] for mm in mets], 0) for k in mets[0]}
        raw[v]["seed_ade"] = [float(mm["ade"][mask].mean()) for mm in mets]
        print(v, "done", flush=True)
    # inference-only extension: confidence-stratified endpoints for SpecFlow
    mets = []
    for s in SEEDS:
        m, _ = load_model("specflow", s)
        o = run(m, te, ep_mode="sampled")
        mets.append(metrics(o["p"], o["a"], te))
    raw["specflow_stratified"] = {k: np.mean([mm[k] for mm in mets], 0) for k in mets[0]}
    raw["specflow_stratified"]["seed_ade"] = [float(mm["ade"][mask].mean()) for mm in mets]
    raw["cv"] = metrics(cv_pred(te), None, te)
    raw["cv"]["seed_ade"] = [float(raw["cv"]["ade"][mask].mean())]
    tk = truth_kin(te)

    def summ(r):
        o = dict(ade=float(r["ade"][mask].mean()), fde=float(r["fde"][mask].mean()),
                 mr=float((r["fde"][mask] > 2.0).mean()),
                 ade_veh=float(r["ade"][veh].mean()), fde_veh=float(r["fde"][veh].mean()),
                 ade_ped=float(r["ade"][ped].mean()), fde_ped=float(r["fde"][ped].mean()),
                 div_veh=float(r["div"][veh].mean()), seed_ade=r["seed_ade"])
        if "jerk" in r:
            o.update(jerk_veh=float(r["jerk"][veh].mean()), infeas_veh=float(r["infeas"][veh].mean()))
        o["by_maneuver"] = {MANEUVERS[k]: [float(r["ade"][mask & (te["man"] == k)].mean()),
                                           float(r["fde"][mask & (te["man"] == k)].mean())] for k in range(7)}
        return o

    S = {k: summ(r) for k, r in raw.items()}
    S["truth"] = dict(jerk_veh=float(tk["jerk"][veh].mean()), infeas_veh=float(tk["infeas"][veh].mean()))
    # paired differences, specflow minus other, bootstrap over scenes (per-scene mean over agents)
    def scene_mean(x):
        return (np.where(mask, x, 0).sum(1) / mask.sum(1))
    diffs = {}
    for k in raw:
        if k == "specflow":
            continue
        diffs[k] = dict(ade=boot_ci(scene_mean(raw["specflow"]["ade"]) - scene_mean(raw[k]["ade"])),
                        fde=boot_ci(scene_mean(raw["specflow"]["fde"]) - scene_mean(raw[k]["fde"])))
    # few-step sampling
    nfe = {}
    for v in ("specflow", "pos_state", "acc_time"):
        nfe[v] = {}
        for st in (1, 2, 5, 10):
            vals = []
            for s in SEEDS:
                m, _ = load_model(v, s)
                o = run(m, te, steps=st)
                mm = metrics(o["p"], o["a"], te)
                vals.append([mm["ade"][mask].mean(), mm["fde"][mask].mean(), mm["jerk"][veh].mean()])
            nfe[v][st] = np.mean(vals, 0).round(4).tolist()
        print("nfe", v, nfe[v], flush=True)
    # clip scenes: >= 4 agents, a turning vehicle, per-scene specflow minADE closest to the median
    sm = scene_mean(raw["specflow"]["ade"])
    has_turn = ((te["man"] == 1) | (te["man"] == 2)).any(1)
    cand = np.nonzero((mask.sum(1) >= 4) & has_turn)[0]
    order = cand[np.argsort(np.abs(sm[cand] - np.median(sm)))]
    # second clip scene: one where top-1 endpoint conditioning hurts (specflow much worse than stratified)
    gap = scene_mean(raw["specflow"]["fde"]) - scene_mean(raw["specflow_stratified"]["fde"])
    cand2 = np.nonzero((mask.sum(1) >= 3) & has_turn)[0]
    bad = cand2[np.argsort(-gap[cand2])]
    summary = dict(n_scenes=int(len(mask)), n_agents=int(mask.sum()), n_veh=int(veh.sum()), n_ped=int(ped.sum()),
                   params=params, metrics=S, diffs=diffs, nfe=nfe, clip_scenes=[int(order[0]), int(order[1])],
                   failure_scene=int(bad[len(bad) // 20]), median_scene_ade=float(np.median(sm)))
    json.dump(summary, open(ROOT / "results/summary.json", "w"), indent=1)
    write_md(summary)
    print(json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items() if isinstance(vv, float)} for k, v in S.items()}, indent=0))


def write_md(S: dict):
    M = S["metrics"]
    L = ["# Full results (synthetic intersection, 4,000 test scenes, best-of-20, mean of 2 seeds)", "",
         f"{S['n_agents']} agents ({S['n_veh']} vehicles, {S['n_ped']} pedestrians). Errors against the noise-free future, "
         "4.8 s horizon at 2.5 Hz.", "",
         "| Variant | params | minADE | minFDE | MR@2m | veh minADE | veh minFDE | ped minADE | ped minFDE | veh RMS jerk | veh samples |a|>6 | veh endpoint spread |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in M.items():
        if k == "truth":
            continue
        p = f"{S['params'][k] / 1e3:.0f}k" if k in S["params"] else ("= specflow" if k.startswith("specflow") else "-")
        j = f"{r['jerk_veh']:.2f}" if "jerk_veh" in r else "-"
        inf = f"{r['infeas_veh'] * 100:.1f}%" if "infeas_veh" in r else "-"
        L.append(f"| {k} | {p} | {r['ade']:.3f} | {r['fde']:.3f} | {r['mr'] * 100:.1f}% | {r['ade_veh']:.3f} | {r['fde_veh']:.3f} | "
                 f"{r['ade_ped']:.3f} | {r['fde_ped']:.3f} | {j} | {inf} | {r['div_veh']:.2f} |")
    L.append(f"| truth (clean track) | | | | | | | | | {M['truth']['jerk_veh']:.2f} | {M['truth']['infeas_veh'] * 100:.1f}% | |")
    L += ["", "## SpecFlow minus other (negative = SpecFlow better), 95% CI by scene bootstrap", "",
          "| vs | minADE | minFDE |", "|---|---|---|"]
    for k, dd in S["diffs"].items():
        L.append(f"| {k} | {dd['ade'][0]:+.3f} ({dd['ade'][1]:+.3f} .. {dd['ade'][2]:+.3f}) | "
                 f"{dd['fde'][0]:+.3f} ({dd['fde'][1]:+.3f} .. {dd['fde'][2]:+.3f}) |")
    L += ["", "## Seed spread (minADE per seed)", ""]
    for k, r in M.items():
        if "seed_ade" in r:
            L.append(f"* {k}: " + ", ".join(f"{x:.3f}" for x in r["seed_ade"]))
    L += ["", "## minADE / minFDE by maneuver", "", "| Variant | " + " | ".join(MANEUVERS) + " |",
          "|---" * (len(MANEUVERS) + 1) + "|"]
    for k, r in M.items():
        if "by_maneuver" in r:
            L.append(f"| {k} | " + " | ".join(f"{a:.2f} / {b:.2f}" for a, b in r["by_maneuver"].values()) + " |")
    L += ["", "## Few-step sampling (minADE / minFDE / veh RMS jerk)", "", "| Variant | 1 step | 2 steps | 5 steps | 10 steps |",
          "|---|---|---|---|---|"]
    for k, r in S["nfe"].items():
        L.append(f"| {k} | " + " | ".join(f"{a:.3f} / {b:.3f} / {c:.1f}" for a, b, c in r.values()) + " |")
    open(ROOT / "results/RESULTS.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
