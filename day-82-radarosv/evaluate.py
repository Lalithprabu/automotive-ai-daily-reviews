"""Evaluate every trained configuration on the 1,500 held-out scenes -> results/summary.json + results/RESULTS.md.

Metrics (all inside the radars' shared field of view, +-60 deg, 1-25.6 m):
    geometric occupancy mIoU   mean of IoU(occupied) and IoU(free), reported within 12.8 m (as in the paper's K-Radar
                               number) and over the full range
    dynamic class accuracy     on GT cells of the four dynamic classes (annotated object regions): per-class accuracy,
                               then the macro average (as in the paper's BRIDGE-Liv number)
    radial-velocity MAE        |v_gr_hat - v_gr| on GT dynamic cells, ground-relative, m/s; split into cells whose
                               radar-relative speed exceeds the imaging radar's +-2.5 m/s (aliased) and those that do not
    candidate accuracy         share of dynamic cells where the selected aliasing index k is right
    ego MAE                    |v_ego_hat - v_ego| of the radar-estimated ego speed
Paired differences use a scene bootstrap (2,000 resamples) of ratio-of-sums statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from osv.radar import VI
from osv.scene import DYN, FOV_MASK, GR
from train import RUNS, build, load_split, to_batch

ROOT = Path(__file__).resolve().parent
LABEL = {"base": "imaging radar + external ego (baseline)", "base_odo": "baseline, odometry noise 0.5 m/s",
         "osv": "imaging + single-chip, radar ego (RadarOSV-style)", "osv_ext": "imaging + single-chip, external ego",
         "osv_fixed": "RadarOSV-style, fixed-only PCG"}


def load(run, seed):
    ck = torch.load(ROOT / "results" / "ckpt" / f"{run}_s{seed}.pt", weights_only=False)
    m = build(run)
    m.load_state_dict(ck["state"])
    return m.eval()


@torch.no_grad()
def predict_all(m, d, ego_noise=0.0, bs=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {k: [] for k in ("sem", "v_gr", "k", "v_ego")}
    for i in range(0, len(d["sem"]), bs):
        b = to_batch(d, np.arange(i, min(i + bs, len(d["sem"]))))
        ext = b["vego"] + ego_noise * torch.randn(len(b["vego"]), generator=g)
        p = m.predict(b["img"], b["sc"], ext)
        for k in out:
            out[k].append(p[k].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def per_scene(p, d):
    """Per-scene sufficient statistics, so every metric can be bootstrapped over scenes."""
    fov = FOV_MASK[None]
    near = fov & (GR <= 12.8)[None]
    gt, pr = d["sem"], p["sem"]
    go, po = gt > 0, pr > 0
    S = {}
    for name, m in (("near", near), ("full", fov)):
        S[f"occ_i_{name}"] = ((go & po) & m).sum((1, 2))
        S[f"occ_u_{name}"] = ((go | po) & m).sum((1, 2))
        S[f"free_i_{name}"] = ((~go & ~po) & m).sum((1, 2))
        S[f"free_u_{name}"] = ((~go | ~po) & m).sum((1, 2))
    for c in range(6):
        S[f"sem_i_{c}"] = ((gt == c) & (pr == c) & fov).sum((1, 2))
        S[f"sem_u_{c}"] = (((gt == c) | (pr == c)) & fov).sum((1, 2))
    for c in DYN:
        mc = (gt == c) & fov
        S[f"dyn_n_{c}"] = mc.sum((1, 2))
        S[f"dyn_ok_{c}"] = (mc & (pr == c)).sum((1, 2))
    dyn = np.isin(gt, DYN) & fov
    vrel = d["vrel"].astype(np.float32)
    err = np.abs(p["v_gr"] - d["vgr"].astype(np.float32))
    k_gt = np.round((vrel - ((vrel + VI) % (2 * VI) - VI)) / (2 * VI))
    ali = np.abs(vrel) > VI
    for name, m in (("all", dyn), ("ali", dyn & ali), ("una", dyn & ~ali)):
        S[f"verr_{name}"] = (err * m).sum((1, 2))
        S[f"vn_{name}"] = m.sum((1, 2))
    S["kok"] = ((p["k"] == k_gt) & dyn).sum((1, 2))
    st = (gt == 1) & fov
    S["serr"], S["sn"] = (err * st).sum((1, 2)), st.sum((1, 2))
    S["ego_err"] = np.abs(p["v_ego"] - d["vego"])
    return S


def metrics(S, idx=None):
    s = {k: (v if idx is None else v[idx]).sum() for k, v in S.items()}
    r = lambda a, b: float(s[a] / max(s[b], 1))
    M = {}
    for name in ("near", "full"):
        M[f"occ_miou_{name}"] = 0.5 * (r(f"occ_i_{name}", f"occ_u_{name}") + r(f"free_i_{name}", f"free_u_{name}"))
    M["sem_miou"] = float(np.mean([r(f"sem_i_{c}", f"sem_u_{c}") for c in range(6)]))
    M["dyn_acc"] = {int(c): r(f"dyn_ok_{c}", f"dyn_n_{c}") for c in DYN}
    M["dyn_macro_acc"] = float(np.mean(list(M["dyn_acc"].values())))
    for name in ("all", "ali", "una"):
        M[f"vel_mae_{name}"] = r(f"verr_{name}", f"vn_{name}")
    M["cand_acc"] = r("kok", "vn_all")
    M["static_vel_mae"] = r("serr", "sn")
    M["ego_mae"] = float(s["ego_err"] / (len(S["ego_err"]) if idx is None else len(idx)))
    M["share_aliased"] = r("vn_ali", "vn_all")
    return M


def boot(SA, SB, key, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(SA["ego_err"])
    f = (lambda S, i: metrics(S, i)[key]) if key != "dyn_macro_acc" else (lambda S, i: metrics(S, i)[key])
    diffs = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        diffs.append(f(SB, i) - f(SA, i))
    full = metrics(SB)[key] - metrics(SA)[key]
    return dict(diff=full, lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)))


def mean_seeds(ms):
    out = {}
    for k, v in ms[0].items():
        out[k] = ({c: float(np.mean([m[k][c] for m in ms])) for c in v} if isinstance(v, dict)
                  else float(np.mean([m[k] for m in ms])))
    return out


def main():
    torch.set_num_threads(2)
    d = load_split("test")
    ck = ROOT / "results" / "ckpt"
    per_seed, stats = {}, {}
    jobs = [(r, s, 0.0, r) for r in RUNS for s in (0, 1) if (ck / f"{r}_s{s}.pt").exists()]
    jobs += [("base", s, 0.5, "base_odo") for s in (0, 1) if (ck / f"base_s{s}.pt").exists()]
    for run, seed, noise, name in jobs:
        p = predict_all(load(run, seed), d, ego_noise=noise, seed=seed)
        S = per_scene(p, d)
        stats[(name, seed)] = S
        per_seed.setdefault(name, {})[seed] = metrics(S)
        print(name, seed, json.dumps({k: round(v, 4) for k, v in per_seed[name][seed].items()
                                      if not isinstance(v, dict)}), flush=True)
    summary = dict(n_scenes=int(len(d["sem"])), labels=LABEL,
                   metrics={n: mean_seeds(list(v.values())) for n, v in per_seed.items()},
                   per_seed={n: {str(s): m for s, m in v.items()} for n, v in per_seed.items()})
    comps = {}
    for a, b in (("base", "osv"), ("osv", "osv_ext"), ("osv_fixed", "osv"), ("base_odo", "osv")):
        if (a, 0) in stats and (b, 0) in stats:
            for key in ("vel_mae_all", "dyn_macro_acc", "occ_miou_near", "occ_miou_full", "cand_acc"):
                comps[f"{b}_minus_{a}:{key}:seed0"] = boot(stats[(a, 0)], stats[(b, 0)], key, n=500)
    summary["paired_seed0"] = comps
    # per-scene velocity error for both main models (seed 0) - for clip selection / inspection
    np.savez(ROOT / "results" / "per_scene.npz",
             **{f"{n}_verr": stats[(n, 0)]["verr_all"] / np.maximum(stats[(n, 0)]["vn_all"], 1)
                for n in ("base", "osv") if (n, 0) in stats})
    json.dump(summary, open(ROOT / "results" / "summary.json", "w"), indent=1)
    write_md(summary)


def write_md(S):
    M = S["metrics"]
    L = ["# Full results (synthetic, 1,500 held-out scenes; mean over seeds where 2 exist)", "",
         "| Configuration | seeds | occ mIoU <=12.8 m | occ mIoU full | sem mIoU (6 cls) | dyn macro acc | car | truck | "
         "cyclist | ped | vel MAE (m/s) | aliased | unaliased | candidate acc | static vel MAE | ego MAE |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in ("base", "base_odo", "osv", "osv_ext", "osv_fixed"):
        if n not in M:
            continue
        m, ns = M[n], len(S["per_seed"][n])
        a = m["dyn_acc"]
        L.append(f"| {S['labels'][n]} | {ns} | {m['occ_miou_near']:.3f} | {m['occ_miou_full']:.3f} | {m['sem_miou']:.3f} | "
                 f"{m['dyn_macro_acc']:.3f} | {a['2']if '2' in a else a[2]:.3f} | {a.get('3', a.get(3)):.3f} | "
                 f"{a.get('4', a.get(4)):.3f} | {a.get('5', a.get(5)):.3f} | {m['vel_mae_all']:.3f} | "
                 f"{m['vel_mae_ali']:.3f} | {m['vel_mae_una']:.3f} | {m['cand_acc']:.3f} | {m['static_vel_mae']:.3f} | "
                 f"{m['ego_mae']:.3f} |")
    L += ["", f"Share of dynamic cells aliased in the imaging radar: {M['osv']['share_aliased']:.3f}", "",
          "## Per-seed", "", "| Configuration | seed | occ mIoU <=12.8 m | dyn macro acc | vel MAE | ego MAE |", "|---|---|---|---|---|---|"]
    for n, v in S["per_seed"].items():
        for s, m in v.items():
            L.append(f"| {n} | {s} | {m['occ_miou_near']:.3f} | {m['dyn_macro_acc']:.3f} | {m['vel_mae_all']:.3f} | "
                     f"{m['ego_mae']:.3f} |")
    L += ["", "## Paired differences (seed 0), 95% CI by scene bootstrap (500 resamples)", "",
          "| Comparison | diff | CI |", "|---|---|---|"]
    for k, v in S["paired_seed0"].items():
        L.append(f"| {k} | {v['diff']:+.3f} | {v['lo']:+.3f} .. {v['hi']:+.3f} |")
    (ROOT / "results" / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
