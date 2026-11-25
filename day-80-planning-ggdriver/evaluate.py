"""Evaluate every trained variant on the 1,500 held-out scenes and answer the four abstract-level questions.

    Q1  does gazing (per-candidate, path-aligned patches) beat the glance-only planner (shared compressed tokens)?
    Q2  do path-aligned patches beat a MATCHED number of random patches?
    Q3  do path-aligned patches beat attention over ALL patches?
    Q4  is gazing for ranking worth more than gazing for refinement only?

Writes results/summary.json, results/RESULTS.md and results/per_clip.npz (used by simulate.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ggd.model import GlanceGazePlanner, corridor_mask
from ggd.scene import ANCHORS, SCENE_TYPES, score_traj, traj_points
from train import VARIANTS, load_split, to_batch

ROOT = Path(__file__).resolve().parent
ORDER = ["glance", "glance_s1", "gaze_refine", "gaze_score", "gaze", "random", "all"]
LABEL = {"glance": "glance only (shared tokens)", "glance_s1": "glance only, seed 1", "gaze_refine": "gaze for refinement only",
         "gaze_score": "gaze for scoring only", "gaze": "GGDriver-style: gaze refine + score",
         "random": "random patches, matched count", "all": "attention over all 240 patches"}


def load_model(tag: str) -> tuple[GlanceGazePlanner, dict]:
    ck = torch.load(ROOT / "results" / "ckpt" / f"{tag}.pt", weights_only=False)
    m = GlanceGazePlanner(*ck["modes"], d=ck["cfg"]["model"]["d"], n_reg=ck["cfg"]["model"]["n_reg"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck


def gt_of(dv, b):
    pts = traj_points(dv, b["v0"][:, None])
    return score_traj(pts, dv, b["v0"], b["obj"], b["valid"], b["edges"], b["zmax"])


@torch.no_grad()
def run_model(m, d, top_m, bs=100, seed=0):
    torch.manual_seed(seed)                                    # the random-mask variant draws masks at test time
    keys = ["score", "nc", "dac", "ep"]
    out = {k: [] for k in keys + ["dv", "pred", "regret", "top1_ok", "glance_pick", "frac", "rank_corr"]}
    for s in range(0, len(d["v0"]), bs):
        idx = np.arange(s, min(s + bs, len(d["v0"])))
        b = to_batch(d, idx)
        p = m.plan(b["img"], b["v0"], top_m)
        g = gt_of(p["dv"][:, None], b)
        for k in keys:
            out[k].append(g[k][:, 0])
        out["dv"].append(p["dv"])
        out["pred"].append(p["score"])
        gm = gt_of(p["out"]["dv"], b)["score"]                                      # GT of all M refined proposals
        out["regret"].append(gm.max(-1).values - g["score"][:, 0])
        out["top1_ok"].append((gm.gather(1, p["best"][:, None])[:, 0] >= gm.max(-1).values - 1e-6).float())
        # glance alone: top-1 anchor by glance logit, unrefined
        gi = p["full"]["glance_logit"].argmax(-1)
        out["glance_pick"].append(torch.from_numpy(d["anchor_score"][idx])[torch.arange(len(idx)), gi])
        ms = p["out"]["mask_score"]
        out["frac"].append(ms.float().mean((1, 2)) if ms is not None else torch.ones(len(idx)))
        # Spearman-like: rank correlation of predicted vs true score over the full 45 refined proposals
        fp, fg = p["full"]["score"], gt_of(p["full"]["dv"], b)["score"]
        rp, rg = fp.argsort(-1).argsort(-1).float(), fg.argsort(-1).argsort(-1).float()
        rp, rg = rp - rp.mean(-1, keepdim=True), rg - rg.mean(-1, keepdim=True)
        out["rank_corr"].append((rp * rg).sum(-1) / (rp.norm(dim=-1) * rg.norm(dim=-1)).clamp(min=1e-6))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def boot(a, b=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = a - (b if b is not None else 0)
    i = rng.integers(0, len(x), (n, len(x)))
    bs = x[i].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    torch.set_num_threads(2)
    cfg = __import__("yaml").safe_load(open(ROOT / "config.yaml"))
    d = load_split("test", cfg)
    st = d["stype"]
    b = to_batch(d, np.arange(len(d["v0"])))
    keep_dv = torch.stack([torch.zeros_like(b["v0"]), b["v0"]], -1)[:, None]
    keep = gt_of(keep_dv, b)
    R = {"keep_lane": dict(score=keep["score"][:, 0].numpy(), nc=keep["nc"][:, 0].numpy(), dac=keep["dac"][:, 0].numpy()),
         "oracle_anchor": dict(score=d["anchor_score"].max(1)), "oracle_refined": dict(score=d["best_score"].max(1))}
    tags = [t for t in ORDER if (ROOT / "results" / "ckpt" / f"{t}.pt").exists()]
    for t in tags:
        m, _ = load_model(t)
        R[t] = run_model(m, d, cfg["eval"]["top_m"])
        print(t, f"score {R[t]['score'].mean():.4f} collision {1 - R[t]['nc'].mean():.4f}", flush=True)

    def row(k):
        r = R[k]
        o = dict(score=float(r["score"].mean()))
        o.update({f"score_{SCENE_TYPES[s]}": float(r["score"][st == s].mean()) for s in range(3)})
        if "nc" in r:
            o["collision_rate"] = float(1 - r["nc"].mean())
            o["collision_small"] = float(1 - r["nc"][st == 0].mean())
            o["offroad_rate"] = float(1 - r["dac"].mean())
        for k2 in ("regret", "top1_ok", "glance_pick", "frac", "rank_corr"):
            if k2 in r:
                o[k2] = float(r[k2].mean())
        return o

    M = {k: row(k) for k in R}
    C = {}
    for t in tags:
        if t != "glance" and "glance" in R:
            C[f"{t}_minus_glance/score"] = boot(R[t]["score"], R["glance"]["score"])
            C[f"{t}_minus_glance/score_small"] = boot(R[t]["score"][st == 0], R["glance"]["score"][st == 0])
            C[f"{t}_minus_glance/collision"] = boot(1 - R[t]["nc"], 1 - R["glance"]["nc"])
    for t in ("random", "all"):
        if t in R and "gaze" in R:
            C[f"gaze_minus_{t}/score"] = boot(R["gaze"]["score"], R[t]["score"])
            C[f"gaze_minus_{t}/score_small"] = boot(R["gaze"]["score"][st == 0], R[t]["score"][st == 0])
    if all(t in R for t in ("gaze_score", "gaze_refine")):
        C["gaze_score_minus_gaze_refine/score"] = boot(R["gaze_score"]["score"], R["gaze_refine"]["score"])

    # ------------------------------------------------------------ clips for the simulation (rules fixed in advance)
    clips = {}
    if "gaze" in R and "glance" in R:
        rng = np.random.default_rng(80)
        small = np.where(st == 0)[0]
        clips["random_small"] = int(rng.choice(small))                    # 1: a random small-hazard scene
        win = np.where((R["gaze"]["nc"] == 1) & (R["glance"]["nc"] == 0) & (st == 0))[0]
        loss = np.where((R["gaze"]["nc"] == 0) & (R["glance"]["nc"] == 1))[0]
        if len(win):                                                      # 2: a glance crash that gaze avoids (median
            clips["gaze_win"] = int(win[np.argsort(R["gaze"]["score"][win])[len(win) // 2]])   # gaze score among them)
        if len(loss):                                                     # 3: gaze crashes, glance does not (median)
            clips["gaze_fail"] = int(loss[np.argsort(R["glance"]["score"][loss])[len(loss) // 2]])
        clips["n_gaze_win"], clips["n_gaze_fail_vs_glance_ok"] = int(len(win)), int(
            ((R["gaze"]["nc"] == 0) & (R["glance"]["nc"] == 1)).sum())
    S = dict(n_clips=int(len(st)), n_by_type={SCENE_TYPES[s]: int((st == s).sum()) for s in range(3)}, metrics=M, ci=C,
             clips=clips, labels=LABEL)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(S, open(ROOT / "results" / "summary.json", "w"), indent=1)
    np.savez_compressed(ROOT / "results" / "per_clip.npz",
                        **{f"{t}_{k}": R[t][k] for t in tags for k in ("score", "nc", "dac", "dv")}, stype=st)

    L = [f"# Full results ({len(st)} held-out scenes: " + ", ".join(f"{v} {k}" for k, v in S["n_by_type"].items()) + ")\n",
         "Score = NC x DAC x (0.75 EP + 0.25 C) of the trajectory each planner picks (top-8 glance hypotheses -> refine "
         "-> re-scored). Collision = share of scenes where the picked trajectory hits an object within 4 s.\n",
         "| Planner | score | small-hazard | stalled car | clear lane | collision | collision (small) | off-road | "
         "patches seen by scorer | rank corr (45) | top-1 among 8 | regret |", "|---|" + "---|" * 11]
    f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    for k in ["keep_lane", "oracle_anchor", "oracle_refined"] + tags:
        r = M[k]
        nm = LABEL.get(k, k.replace("_", " "))
        L.append(f"| {nm} | {f(r['score'])} | {f(r['score_small'])} | {f(r['score_big'])} | {f(r['score_clear'])} | "
                 f"{f(r.get('collision_rate'))} | {f(r.get('collision_small'))} | {f(r.get('offroad_rate'))} | "
                 f"{f(r.get('frac'), 2)} | {f(r.get('rank_corr'))} | {f(r.get('top1_ok'))} | {f(r.get('regret'))} |")
    L += ["", "## Paired differences, 95% CI by scene bootstrap", "", "| Comparison | diff | CI |", "|---|---|---|"]
    for k, (m_, lo, hi) in C.items():
        L.append(f"| {k} | {m_:+.4f} | {lo:+.4f} .. {hi:+.4f} |")
    L += ["", f"Glance-only pick (top-1 anchor by glance logit, no refinement): " +
          ", ".join(f"{t} {M[t]['glance_pick']:.3f}" for t in tags), "", f"Simulation clips: {json.dumps(clips)}"]
    open(ROOT / "results" / "RESULTS.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
