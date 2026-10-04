"""Render assets/architecture.png (this repo's own diagram, NOT the paper's figure) and assets/results.png from real run data."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG, FG = "#0b1220", "#e6edf7"
fig, ax = plt.subplots(figsize=(13, 5.2), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2)
def bx(x, y, w, h, t, c, fs=9): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="none", alpha=.9)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=fs, fontweight="bold")
def ar(a, b, c="#9fb0d0"): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
bx(.2, 3.7, 1.8, .8, "Ego BEV feats\n(time t, 40x40)", "#2f6f4f"); bx(.2, .6, 1.8, .8, "Collaborator BEV feats\n(time t-Δ, delayed)", "#8a5a2b")
bx(2.5, .6, 1.6, .8, "Traj-field head\nK=3 base offsets", "#8a5a2b"); ar((2, 1), (2.5, 1))
bx(4.6, 2.1, 2.2, 1.5, "EGO-REFERENCED\nREFINER\nego+collab+offsets →\nunit direction d, step s_k\no_ref = o_base + s_k·d", "#2b5fa8", 8.5)
ar((4.1, 1.2), (4.7, 2.2)); ar((2, 4.1), (4.6, 3.3), "#2ee59d")
bx(7.3, .5, 2, 1.1, "Deformable sample\nof collab feats at\np + o_ref,k (K pts)", "#8a5a2b"); ar((6.8, 2.4), (7.5, 1.6)); ar((4.1, .8), (7.3, .8))
bx(7.3, 2.6, 2, 1.1, "RELIABILITY GATE\nin: traj discrepancy,\nrefine magnitude", "#7a3fa0", 8.5); ar((6.8, 3.0), (7.3, 3.1)); ar((8.3, 1.6), (8.3, 2.6))
bx(10, 1.8, 1.4, 1.1, "Fuse\n[ego , g·aligned]", "#444f66"); ar((9.3, 3.1), (10.1, 2.7)); ar((9.3, 1.2), (10.1, 2.0)); ar((2, 4.3), (10.6, 4.3), "#2ee59d"); ar((10.7, 4.3), (10.7, 2.9), "#2ee59d")
bx(11.7, 1.8, 1.1, 1.1, "Centre\nheatmap\nat time t", "#b33b4a"); ar((11.4, 2.35), (11.7, 2.35))
ax.text(.2, 5.0, "EgoRefine-style fusion — this repo's reconstruction (paper gives module names only)", color=FG, fontsize=11, fontweight="bold")
fig.savefig("assets/architecture.png", dpi=110, facecolor=BG, bbox_inches="tight")
r = json.load(open("checkpoints/results_extra_seeds.json")); P = json.load(open("checkpoints/results.json"))
modes = ["ego_only", "naive", "traf", "egorefine"]; lab = ["ego only", "naive", "old: TraF-style", "EgoRefine-style"]; cols = ["#6b7a99", "#8892a6", "#ff9f43", "#5aa9ff"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)
for a, key, ttl in ((axs[0], "AP@1.0", "AP@1.0 (all objects)"), (axs[1], "AP@1.0_blind", "AP@1.0 — ego-BLIND objects only")):
    m = [np.mean(r[k][key]) for k in modes]; s = [np.std(r[k][key]) for k in modes]
    a.bar(lab, m, yerr=s, color=cols, capsize=4, ecolor="white"); a.set_title(ttl + "\n(mean±sd, 3 eval seeds × 1000 scenes)", color=FG, fontsize=10)
    for i, v in enumerate(m): a.text(i, v + .02, f"{v:.3f}", ha="center", color=FG, fontsize=9)
    a.set_ylim(0, 1.05)
for k, l, c in zip(modes, lab, cols): axs[2].plot(range(7), [P[k]["AP@1.0_by_delay"][str(d)] for d in range(7)], marker="o", color=c, label=l)
axs[2].set_title("AP@1.0 vs. true delay Δ (400 scenes, single seed)", color=FG, fontsize=10); axs[2].set_xlabel("delay Δ (steps)", color=FG); axs[2].legend(facecolor=BG, labelcolor=FG, edgecolor="#33415c", fontsize=8)
for a in axs: a.set_facecolor(BG); a.tick_params(colors="#9fb0d0", labelsize=8); [s.set_color("#33415c") for s in a.spines.values()]
fig.suptitle("SYNTHETIC data — this repo's reconstruction, not the paper's V2V4Real / DAIR-V2X-Seq numbers", color="#ffd166", fontsize=10, y=1.08)
fig.savefig("assets/results.png", dpi=110, facecolor=BG, bbox_inches="tight")
