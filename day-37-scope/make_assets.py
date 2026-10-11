"""Renders assets/architecture.png (this repo's own diagram, NOT the paper's figure) and assets/results.png from results*.json."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, GRAY, ORANGE, INK = "#2a6fdb", "#7a7a7a", "#e08a1e", "#1b1f2a"


def box(ax, x, y, w, h, t, fc="#eef3fc", ec=BLUE, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, a, b, t=None):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    if t: ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + .12, t, fontsize=7, ha="center", color="#555")


fig, ax = plt.subplots(figsize=(13, 5.6), dpi=110); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")
ax.text(.1, 5.3, "SCOPE-style control-ready uncertainty — this repo's reconstruction", fontsize=12, weight="bold", color=INK)
box(ax, .2, 3.4, 2.2, 1.0, "Scene input\npast 8 steps + crowd feature\n+ mode (left/right)")
box(ax, 3.0, 3.4, 2.6, 1.0, "FROZEN diffusion backbone\nEDM denoiser D(x, σ)\n(own stand-in, 217k params)", fc="#f6f6f6", ec=GRAY)
box(ax, 6.3, 3.4, 2.0, 1.0, "Nominal trajectory\nx* = deterministic\nODE decode (σ: 2→0)", fc="#f6f6f6", ec=GRAY)
arrow(ax, (2.4, 3.9), (3.0, 3.9)); arrow(ax, (5.6, 3.9), (6.3, 3.9))
box(ax, 3.0, 1.55, 2.6, 1.0, "Score curvature at x*\nHVP: (I − ∂D/∂x)·v\nvia forward-mode JVP", fc="#fff4e5", ec=ORANGE)
arrow(ax, (4.3, 3.4), (4.3, 2.55), "teacher")
box(ax, 6.3, 1.55, 2.6, 1.0, "SCOPE precision head (250k)\nP = s₀·(L Lᵀ + U Uᵀ + εI)\nL banded-lower · U low-rank")
arrow(ax, (7.3, 3.4), (7.6, 2.55), "anchor x*")
arrow(ax, (5.6, 2.05), (6.3, 2.05), "distil")
box(ax, 9.6, 1.55, 1.7, 1.0, "Σ = P⁻¹ · α\n(α: 1 scalar,\nheld-out NLL)")
arrow(ax, (8.9, 2.05), (9.6, 2.05))
box(ax, 11.5, 1.55, 1.4, 1.0, "per-step 2-D\nGaussian tubes\n→ occupancy", fc="#e8f6ee", ec="#1f9d55")
arrow(ax, (11.3, 2.05), (11.5, 2.05))
ax.text(.2, 0.95, "Old way: draw K diffusion samples, take their covariance (K × S denoiser calls, noisy, ~45× slower here).", fontsize=9, color=ORANGE)
ax.text(.2, 0.55, "New way: one head forward + one 24×24 solve. No sampling at control time.   Mode ambiguity stays in the backbone's mode probabilities; the tube covers spread WITHIN one mode.", fontsize=9, color=BLUE)
fig.savefig("assets/architecture.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

r = json.load(open("results.json")); b = json.load(open("results_buckets.json")); m = r["test_metrics"]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=110)
names = ["SCOPE (calibrated)", "Fixed covariance", [k for k in m if k.startswith("MC")][0], "Oracle (true noise cov)"]
lab = ["SCOPE", "Fixed", "MC (K=32)", "Oracle"]; cols = [BLUE, GRAY, ORANGE, "#1f9d55"]
axs[0].bar(lab, [-m[n]["nll_per_step"] for n in names], color=cols); axs[0].set_title("-NLL per step (higher = better)\nheld-out synthetic test", fontsize=10)
axs[0].set_ylim(1.1, 1.45)
for i, n in enumerate(names): axs[0].text(i, -m[n]["nll_per_step"] + .005, f"{-m[n]['nll_per_step']:.3f}", ha="center", fontsize=9)
ks = [k for k in b if not k.startswith("_")]; x = range(len(ks))
axs[1].bar([i - .2 for i in x], [b[k]["scope_cov95"] for k in ks], .4, color=BLUE, label="SCOPE"); axs[1].bar([i + .2 for i in x], [b[k]["fixed_cov95"] for k in ks], .4, color=GRAY, label="Fixed")
axs[1].axhline(.95, color="#d64545", ls="--", lw=1); axs[1].set_ylim(.85, 1.0); axs[1].set_xticks(list(x)); axs[1].set_xticklabels(["low\ndensity", "mid", "high\ndensity"], fontsize=9)
axs[1].set_title("95% tube coverage by crowd density\n(target 0.95 = red line)", fontsize=10); axs[1].legend(fontsize=8)
l = r["latency_ms"]; axs[2].bar(["SCOPE head\n+ solve", "MC sampling\n(K=32)"], [l["scope_head_plus_solve_ms"], l["mc_sampling_ms"]], color=[BLUE, ORANGE]); axs[2].set_yscale("log")
axs[2].set_title("Uncertainty latency per scene (ms, CPU, log)", fontsize=10)
for i, v in enumerate([l["scope_head_plus_solve_ms"], l["mc_sampling_ms"]]): axs[2].text(i, v * 1.1, f"{v:.2f} ms", ha="center", fontsize=9)
fig.suptitle("This repo's synthetic-data results (NOT the paper's numbers)", fontsize=9, color="#666", y=0.01)
fig.tight_layout(); fig.savefig("assets/results.png", bbox_inches="tight", facecolor="white")
