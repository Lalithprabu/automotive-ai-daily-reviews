"""Evaluate every trained checkpoint on the 30 held-out drivers -> results/summary.json and results/RESULTS.md.

Questions (paper claims from the abstract in brackets):
  Q1  Does the self-supervised reconstruction objective improve impairment inference?
      [paired AUC on DIVERT 0.706 with reconstruction vs 0.590 without]
  Q2  Does ID-MAE beat a supervised baseline?  [abstract: improves over existing baselines; no numbers given]
  Q3  Can visually demanding distractions be told apart from cognitive ones?  [abstract: yes]
  Q4  Trajectory and gaze prediction quality  [abstract: "strong performance", no numbers given]
  Ext Does knowing the impairment label help forecasting?  Iterative vs one-shot unmasking?  (mine)

Paired AUC: for every (driver, location) the same road was driven under each condition.  For each impaired window
we compare its impairment score s = 1 - p(none) with the score of the location-matched baseline window;
paired AUC = P(s_impaired > s_baseline) (+0.5 for ties).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from idm.baselines import forecast_metrics, kinematic_forecast
from idm.data import CONDS
from idm.model import P
from train import build, ckpt_path, load_data

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
VARIANTS = ("idmae", "norecon", "gru")


def load_model(variant: str, frac: float, seed: int):
    m = build(variant)
    m.load_state_dict(torch.load(ckpt_path(variant, frac, seed), map_location="cpu"))
    return m.eval()


@torch.no_grad()
def predict(m, d, bs=512):
    G, V, R = (torch.from_numpy(d[k]) for k in ("gaze", "veh", "road"))
    probs, embs = [], []
    for i in range(0, len(G), bs):
        p, e = m.classify(G[i:i + bs], V[i:i + bs], R[i:i + bs])
        probs.append(p); embs.append(e if e is not None else torch.zeros(len(p), 1))
    return torch.cat(probs).numpy(), torch.cat(embs).numpy()


def paired_auc(score, y, pair, cls=None):
    """Location-matched paired AUC.  cls: restrict impaired windows to one condition."""
    base = {pr: s for s, yy, pr in zip(score, y, pair) if yy == 0}
    w = [(s > base[pr]) + 0.5 * (s == base[pr]) for s, yy, pr in zip(score, y, pair)
         if yy > 0 and (cls is None or yy == cls) and pr in base]
    return float(np.mean(w))


def pair_bootstrap(score_a, score_b, y, pair, n=1000, seed=0):
    """95% CI of paired-AUC(a) - paired-AUC(b), resampling location groups."""
    r = np.random.default_rng(seed)
    groups = np.unique(pair)
    idx_by = {g: np.nonzero(pair == g)[0] for g in groups}
    diffs = []
    for _ in range(n):
        gs = r.choice(groups, len(groups))
        idx = np.concatenate([idx_by[g] for g in gs])
        # re-label resampled groups so duplicates stay separate pairs
        pr = np.concatenate([np.full(len(idx_by[g]), k) for k, g in enumerate(gs)])
        diffs.append(paired_auc(score_a[idx], y[idx], pr) - paired_auc(score_b[idx], y[idx], pr))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def cls_metrics(p, d):
    y, pair = d["y"], d["pair"]
    s = 1 - p[:, 0]
    out = dict(acc=float((p.argmax(1) == y).mean()), macro_f1=float(f1_score(y, p.argmax(1), average="macro")),
               auc=float(roc_auc_score(y > 0, s)), paired_auc=paired_auc(s, y, pair))
    for c in (1, 2, 3):
        out[f"paired_auc_{CONDS[c]}"] = paired_auc(s, y, pair, c)
    m = (y == 1) | (y == 2)                                            # visual vs cognitive
    out["vis_vs_cog_auc"] = float(roc_auc_score(y[m] == 2, p[m, 2] / (p[m, 1] + p[m, 2] + 1e-9)))
    sus = d["suscept"]
    for name, lo, hi in (("low", 0.0, 0.4), ("mid", 0.4, 0.7), ("high", 0.7, 1.01)):
        mm = (sus >= lo) & (sus < hi)
        out[f"paired_auc_suscept_{name}"] = paired_auc(s[mm], y[mm], pair[mm])
    return out


def probe(emb_tr, y_tr, emb_te, y_te):
    """Linear probe on the label-token embedding: visual vs cognitive (train drivers -> test drivers)."""
    mt, me = (y_tr == 1) | (y_tr == 2), (y_te == 1) | (y_te == 2)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(emb_tr[mt], y_tr[mt] == 2)
    return float(roc_auc_score(y_te[me] == 2, clf.decision_function(emb_te[me])))


@torch.no_grad()
def forecasting(m, d, n_future, steps, label=None, bs=512):
    G, V, R = (torch.from_numpy(d[k]) for k in ("gaze", "veh", "road"))
    pg, pv = [], []
    for i in range(0, len(G), bs):
        lab = None if label is None else torch.from_numpy(label[i:i + bs])
        g, v = m.forecast(G[i:i + bs], V[i:i + bs], R[i:i + bs], n_future, steps, lab)
        pg.append(g); pv.append(v)
    return torch.cat(pg).numpy(), torch.cat(pv).numpy()


def mean_sd(vals):
    return dict(mean=float(np.mean(vals)), sd=float(np.std(vals)), per_seed=[float(v) for v in vals])


def main():
    torch.set_num_threads(2)
    te, tr = load_data("test"), load_data("train")
    tc, ec = CFG["train"], CFG["eval"]
    S = dict(n_test=int(len(te["y"])), n_test_drivers=int(len(np.unique(te["driver"]))), seeds=tc["seeds"],
             cls={}, probe={}, delta={}, forecast={})
    scores = {}
    for frac in tc["label_fracs"]:
        for v in VARIANTS:
            per = []
            sc = []
            for s in tc["seeds"]:
                m = load_model(v, frac, s)
                p, e = predict(m, te)
                per.append(cls_metrics(p, te)); sc.append(1 - p[:, 0])
                if v != "gru":
                    _, etr = predict(m, tr)
                    S["probe"].setdefault(f"{v}_f{frac}", []).append(probe(etr, tr["y"], e, te["y"]))
            key = f"{v}_f{frac}"
            S["cls"][key] = {k: mean_sd([r[k] for r in per]) for k in per[0]}
            scores[key] = np.mean(sc, 0)                                # seed-ensemble score for bootstrap
            print(key, {k: round(S["cls"][key][k]["mean"], 3) for k in ("acc", "auc", "paired_auc", "vis_vs_cog_auc")},
                  flush=True)
        for b in ("norecon", "gru"):
            a_, b_ = scores[f"idmae_f{frac}"], scores[f"{b}_f{frac}"]
            S["delta"][f"idmae_minus_{b}_f{frac}"] = dict(
                ensemble_diff=paired_auc(a_, te["y"], te["pair"]) - paired_auc(b_, te["y"], te["pair"]),
                ci95=pair_bootstrap(a_, b_, te["y"], te["pair"]))
    S["probe"] = {k: mean_sd(v) for k, v in S["probe"].items()}
    # ---------------------------------------------------------------- forecasting (ID-MAE, full labels)
    nf, n_steps = ec["n_future"], ec["n_future"] * P
    kg, kv = kinematic_forecast(te["gaze"], te["veh"], n_steps)
    fm = {"kinematic": forecast_metrics(kg, kv, te["gaze"], te["veh"], n_steps)}
    runs = {"idmae_iterative": dict(steps=ec["sampler_steps"]), "idmae_oneshot": dict(steps=1),
            "idmae_iterative_true_label": dict(steps=ec["sampler_steps"], label=te["y"])}
    for name, kw in runs.items():
        acc = []
        for s in tc["seeds"]:
            m = load_model("idmae", 1.0, s)
            g, v = forecasting(m, te, nf, kw["steps"], kw.get("label"))
            acc.append(forecast_metrics(g, v, te["gaze"], te["veh"], n_steps))
        fm[name] = {k: np.mean([a[k] for a in acc], 0) for k in acc[0]}
    for name, met in fm.items():
        S["forecast"][name] = {k: float(np.mean(x)) for k, x in met.items()}
        S["forecast"][name]["by_cond"] = {CONDS[c]: {k: float(np.mean(x[te["y"] == c])) for k, x in met.items()}
                                          for c in range(4)}
        print(name, {k: round(v_, 3) for k, v_ in S["forecast"][name].items() if k != "by_cond"}, flush=True)
    json.dump(S, open(ROOT / "results/summary.json", "w"), indent=1)
    write_md(S)


def write_md(S):
    L = [f"# Results (synthetic toy, {S['n_test']} held-out windows from {S['n_test_drivers']} unseen drivers)", "",
         f"Mean +- sd over seeds {S['seeds']}. Paired AUC = location-matched impaired vs baseline window.", "",
         "## Impairment inference", "",
         "| Labels | Variant | 4-way acc | macro-F1 | AUC (impaired vs none) | paired AUC | cog | visual | emotional | visual-vs-cognitive AUC |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda d: f"{d['mean']:.3f} +- {d['sd']:.3f}"
    for key, c in S["cls"].items():
        v, frac = key.split("_f")
        L.append(f"| {float(frac) * 100:.0f}% | {v} | {f(c['acc'])} | {f(c['macro_f1'])} | {f(c['auc'])} | "
                 f"{f(c['paired_auc'])} | {c['paired_auc_cognitive']['mean']:.3f} | {c['paired_auc_visual']['mean']:.3f} | "
                 f"{c['paired_auc_emotional']['mean']:.3f} | {f(c['vis_vs_cog_auc'])} |")
    L += ["", "## Q1 / Q2 - paired-AUC differences (seed-ensemble scores, 95% CI by location bootstrap)", ""]
    for k, d in S["delta"].items():
        L.append(f"* {k}: {d['ensemble_diff']:+.3f} (CI {d['ci95'][0]:+.3f} .. {d['ci95'][1]:+.3f})")
    L += ["", "## Paired AUC by driver susceptibility (low < 0.4 <= mid < 0.7 <= high)", "",
          "| Labels | Variant | low | mid | high |", "|---|---|---|---|---|"]
    for key, c in S["cls"].items():
        v, frac = key.split("_f")
        L.append(f"| {float(frac) * 100:.0f}% | {v} | {c['paired_auc_suscept_low']['mean']:.3f} | "
                 f"{c['paired_auc_suscept_mid']['mean']:.3f} | {c['paired_auc_suscept_high']['mean']:.3f} |")
    L += ["", "## Q3 - linear probe on the label-token embedding: visual vs cognitive AUC", ""]
    for k, d in S["probe"].items():
        L.append(f"* {k}: {f(d)}")
    L += ["", "## Q4 - forecasting the last 2 s (ID-MAE trained with 100% labels; metrics averaged over seeds)", "",
          "| Method | position ADE (m) | position FDE (m) | lateral ADE (m) | speed RMSE (m/s) | gaze error (deg) |",
          "|---|---|---|---|---|---|"]
    for k, d in S["forecast"].items():
        L.append(f"| {k} | {d['pos_ade']:.3f} | {d['pos_fde']:.3f} | {d['lat_ade']:.3f} | {d['speed_rmse']:.3f} | "
                 f"{d['gaze_deg']:.2f} |")
    L += ["", "### Gaze error (deg) by condition", "", "| Method | none | cognitive | visual | emotional |",
          "|---|---|---|---|---|"]
    for k, d in S["forecast"].items():
        L.append(f"| {k} | " + " | ".join(f"{d['by_cond'][c]['gaze_deg']:.2f}" for c in CONDS) + " |")
    open(ROOT / "results/RESULTS.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
