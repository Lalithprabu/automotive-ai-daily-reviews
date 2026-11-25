"""Evaluate every pipeline on the held-out scenes at noise levels 0 ... 4 -> results/summary.json, RESULTS.md, per_clip.npz.

Pipelines (the same noisy histories are fed to all of them at each level):
    cv            constant velocity from the last noisy state (kinematic floor)
    cv_kf         constant velocity from the Kalman-filtered state
    clean         predictor trained on clean (curated) histories                         <- the paper's starting point
    clean_kf      clean predictor + classical Kalman input filter                        <- denoising module (classical)
    clean_dn      clean predictor + learned input denoiser                               <- denoising module (learned)
    aug           predictor retrained on noise-augmented histories                       <- the paper's recommendation
Metrics: minADE6 / minFDE6 over all 8 agents (top-1 ADE too), split by sensor-tracked vs V2X-received agents.
Degradation factor = minADE at level s / minADE of the same pipeline's predictor at level 0 on clean input.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nr.model import HistoryDenoiser, InteractionTransformer, constant_velocity, kalman_cv
from nr.noise import add_noise
from train import CFG, load_split, to_batch

ROOT = Path(__file__).resolve().parent
LABEL = dict(cv="constant velocity (raw)", cv_kf="constant velocity + Kalman", clean="clean-trained",
             clean_kf="clean-trained + Kalman filter", clean_dn="clean-trained + learned denoiser",
             aug="noise-augmented retrain")
SEEDS = (0, 1)


def load(kind: str, seed: int):
    m = HistoryDenoiser() if kind == "denoise" else InteractionTransformer()
    m.load_state_dict(torch.load(ROOT / "results" / "ckpt" / f"{kind}_s{seed}.pt", weights_only=True))
    return m.eval()


@torch.no_grad()
def run_pipeline(name, models, hist, fresh, v2x, seed):
    """-> pred (B, N, K, H, 2), logit (B, N, K) or None."""
    if name == "cv":
        return constant_velocity(hist), None
    if name == "cv_kf":
        return constant_velocity(kalman_cv(hist, fresh)[0]), None
    if name == "clean":
        o = models[f"clean{seed}"](hist, fresh, v2x)
    elif name == "clean_kf":
        h, f = kalman_cv(hist, fresh)
        o = models[f"clean{seed}"](h, f, v2x)
    elif name == "clean_dn":
        h, f = models[f"den{seed}"](hist, fresh, v2x)
        o = models[f"clean{seed}"](h, f, v2x)
    elif name == "aug":
        o = models[f"aug{seed}"](hist, fresh, v2x)
    return o["pred"], o["logit"]


def errors(pred, logit, fut):
    """-> per-agent minADE, minFDE, top-1 ADE  (B, N) each."""
    d = (pred - fut[:, :, None]).norm(dim=-1)                                   # (B, N, K, H)
    ade, fde = d.mean(-1), d[..., -1]
    k = ade.argmin(-1, keepdim=True)
    top = ade[..., 0] if logit is None else ade.gather(-1, logit.argmax(-1, keepdim=True))[..., 0]
    return ade.gather(-1, k)[..., 0], fde.gather(-1, k)[..., 0], top


def boot(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), (n, len(diff)))
    bs = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    torch.set_num_threads(2)
    d = load_split("test")
    S = len(d["states"])
    models = {}
    for s in SEEDS:
        models[f"clean{s}"], models[f"aug{s}"], models[f"den{s}"] = load("clean", s), load("aug", s), load("denoise", s)
    levels = CFG["noise"]["eval_levels"]
    names = list(LABEL)
    # per (name, level): per-scene arrays averaged over agents and seeds
    res = {n: {} for n in names}
    per_scene = {n: {} for n in names}
    hist_err = {k: {} for k in ("raw", "kalman", "learned")}
    for lvl in levels:
        acc = {n: dict(ade=[], fde=[], top=[], ade_s=[], ade_v=[]) for n in names}
        he = {k: [] for k in hist_err}
        for i0 in range(0, S, 500):
            idx = np.arange(i0, min(S, i0 + 500))
            b = to_batch(d, idx)
            g = torch.Generator().manual_seed(int(7000 + 100 * lvl + i0))
            hist, fresh = add_noise(b["hist"], b["v2x"], float(lvl), g)
            v2x = b["v2x"]
            # history (t0 position) error of the raw / filtered input, averaged over agents
            he["raw"].append((hist[:, :, -1, :2] - b["hist"][:, :, -1, :2]).norm(dim=-1).mean(1))
            he["kalman"].append((kalman_cv(hist, fresh)[0][:, :, -1, :2] - b["hist"][:, :, -1, :2]).norm(dim=-1).mean(1))
            with torch.no_grad():
                dn = torch.stack([(models[f"den{s}"](hist, fresh, v2x)[0][:, :, -1, :2] - b["hist"][:, :, -1, :2])
                                  .norm(dim=-1).mean(1) for s in SEEDS]).mean(0)
            he["learned"].append(dn)
            for n in names:
                seeds = (0,) if n.startswith("cv") else SEEDS
                A, Fd, Tp = [], [], []
                for s in seeds:
                    p, lg = run_pipeline(n, models, hist, fresh, v2x, s)
                    a_, f_, t_ = errors(p, lg, b["fut"])
                    A.append(a_), Fd.append(f_), Tp.append(t_)
                A, Fd, Tp = torch.stack(A).mean(0), torch.stack(Fd).mean(0), torch.stack(Tp).mean(0)  # (B, N)
                acc[n]["ade"].append(A.mean(1)), acc[n]["fde"].append(Fd.mean(1)), acc[n]["top"].append(Tp.mean(1))
                vf = v2x.float()
                acc[n]["ade_s"].append((A * (1 - vf)).sum(1) / (1 - vf).sum(1).clamp(min=1))
                acc[n]["ade_v"].append((A * vf).sum(1) / vf.sum(1).clamp(min=1))
                if n == "clean" or n == "aug":                                    # per-seed numbers for spread
                    for s in seeds:
                        p, lg = run_pipeline(n, models, hist, fresh, v2x, s)
                        acc[n].setdefault(f"ade_seed{s}", []).append(errors(p, lg, b["fut"])[0].mean(1))
        for n in names:
            r = {k: torch.cat(v).numpy() for k, v in acc[n].items()}
            per_scene[n][lvl] = r["ade"]
            res[n][str(lvl)] = {k: float(v.mean()) for k, v in r.items()}
        for k in he:
            hist_err[k][str(lvl)] = float(torch.cat(he[k]).mean())
        print(f"level {lvl}: " + "  ".join(f"{n} {res[n][str(lvl)]['ade']:.3f}" for n in names), flush=True)
    # degradation factors
    base = {n: res[n]["0"]["ade"] for n in names}
    base["clean_kf"] = base["clean_dn"] = base["clean"]                           # same predictor, clean input
    for n in names:
        for lvl in levels:
            res[n][str(lvl)]["degradation"] = res[n][str(lvl)]["ade"] / base[n]
            res[n][str(lvl)]["vs_clean_at_0"] = res[n][str(lvl)]["ade"] / res["clean"]["0"]["ade"]
    top = str(levels[-1])
    ci = {
        f"aug_minus_clean@{top}": boot(per_scene["aug"][levels[-1]], per_scene["clean"][levels[-1]]),
        f"aug_minus_clean_dn@{top}": boot(per_scene["aug"][levels[-1]], per_scene["clean_dn"][levels[-1]]),
        f"aug_minus_clean_kf@{top}": boot(per_scene["aug"][levels[-1]], per_scene["clean_kf"][levels[-1]]),
        f"clean_dn_minus_clean_kf@{top}": boot(per_scene["clean_dn"][levels[-1]], per_scene["clean_kf"][levels[-1]]),
        "aug_minus_clean@0": boot(per_scene["aug"][0], per_scene["clean"][0]),
        "aug_minus_clean_dn@1": boot(per_scene["aug"][1], per_scene["clean_dn"][1]),
        "aug_minus_clean_dn@2": boot(per_scene["aug"][2], per_scene["clean_dn"][2]),
    }
    clips = pick_clips(per_scene, levels, S)
    np.savez(ROOT / "results" / "per_clip.npz", **{f"{n}_{l}": per_scene[n][l] for n in names for l in levels})
    out = dict(levels=levels, metrics=res, history_t0_error=hist_err, ci=ci, clips=clips, n_scenes=S, labels=LABEL)
    json.dump(out, open(ROOT / "results" / "summary.json", "w"), indent=1)
    write_md(out)
    print(json.dumps(dict(ci=ci, clips=clips, hist=hist_err), indent=1))


def pick_clips(per_scene, levels, S):
    """Scene 1 random; scene 2 = median of scenes where noise-aug wins by > 1 m at the top level; scene 3 = median of
    scenes where noise-aug is worse than clean-trained by > 0.3 m on CLEAN input (the price of augmentation)."""
    rng = np.random.default_rng(81)
    gain = per_scene["clean"][levels[-1]] - per_scene["aug"][levels[-1]]
    wins = np.where(gain > 1.0)[0]
    cost = per_scene["aug"][levels[0]] - per_scene["clean"][levels[0]]
    loss = np.where(cost > 0.3)[0]
    med = lambda ids, v: int(ids[np.argsort(v[ids])[len(ids) // 2]]) if len(ids) else None
    return dict(random=int(rng.integers(0, S)), win=med(wins, gain), loss=med(loss, cost),
                n_win=int(len(wins)), n_loss=int(len(loss)), n_top_level_worse=int((gain < 0).sum()))


def write_md(o):
    L, M = o["levels"], o["metrics"]
    lines = [f"# Full results ({o['n_scenes']} held-out scenes x 8 agents, 2 seeds per learned pipeline)\n",
             "minADE6 in metres (best of 6 modes, averaged over all agents). Kinematic baselines have one mode.\n",
             "| Pipeline | " + " | ".join(f"level {l}" for l in L) + " | degradation at 4 |", "|---|" + "---|" * (len(L) + 1)]
    for n, lab in o["labels"].items():
        lines.append(f"| {lab} | " + " | ".join(f"{M[n][str(l)]['ade']:.3f}" for l in L)
                     + f" | {M[n][str(L[-1])]['degradation']:.2f}x |")
    lines += ["", "## minFDE6 (m)", "", "| Pipeline | " + " | ".join(f"level {l}" for l in L) + " |", "|---|" + "---|" * len(L)]
    for n, lab in o["labels"].items():
        lines.append(f"| {lab} | " + " | ".join(f"{M[n][str(l)]['fde']:.3f}" for l in L) + " |")
    lines += ["", "## minADE6 by agent source at the top level", "", "| Pipeline | sensor-tracked | V2X-received | top-1 ADE |",
              "|---|---|---|---|"]
    t = str(L[-1])
    for n, lab in o["labels"].items():
        lines.append(f"| {lab} | {M[n][t]['ade_s']:.3f} | {M[n][t]['ade_v']:.3f} | {M[n][t]['top']:.3f} |")
    lines += ["", "## Per-seed minADE6", "", "| Pipeline | " + " | ".join(f"level {l}" for l in L) + " |", "|---|" + "---|" * len(L)]
    for n in ("clean", "aug"):
        for s in (0, 1):
            lines.append(f"| {o['labels'][n]} seed {s} | " + " | ".join(f"{M[n][str(l)][f'ade_seed{s}']:.3f}" for l in L) + " |")
    lines += ["", "## t0 position error of the history fed to the predictor (m, mean over agents)", "",
              "| Input | " + " | ".join(f"level {l}" for l in L) + " |", "|---|" + "---|" * len(L)]
    for k, v in o["history_t0_error"].items():
        lines.append(f"| {k} | " + " | ".join(f"{v[str(l)]:.3f}" for l in L) + " |")
    lines += ["", "## Paired differences in minADE6, 95% CI by scene bootstrap", "", "| Comparison | diff (m) | CI |", "|---|---|---|"]
    for k, (m, lo, hi) in o["ci"].items():
        lines.append(f"| {k} | {m:+.3f} | {lo:+.3f} .. {hi:+.3f} |")
    lines += ["", f"Simulation clips: {json.dumps(o['clips'])}"]
    (ROOT / "results" / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import sys
    if "--clips_only" in sys.argv:                      # recompute clip choice from results/per_clip.npz
        o = json.load(open(ROOT / "results" / "summary.json"))
        z = np.load(ROOT / "results" / "per_clip.npz")
        ps = {n: {l: z[f"{n}_{l}"] for l in o["levels"]} for n in LABEL}
        o["clips"] = pick_clips(ps, o["levels"], o["n_scenes"])
        json.dump(o, open(ROOT / "results" / "summary.json", "w"), indent=1)
        write_md(o)
        print(o["clips"])
    else:
        main()
