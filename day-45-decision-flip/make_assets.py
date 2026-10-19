"""Render this repo's OWN architecture diagram and results chart (not the paper's figures)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
r = json.load(open("checkpoints/prune_results.json"))
fig, ax = plt.subplots(figsize=(13, 4.2), facecolor="#0b0f14"); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 4.2)
boxes = [(0.2, "Unlabeled\ncalibration\nscenes", "#374151"), (2.6, "Dense Gaussian bank\n(384 persistent rows)", "#1d4ed8"), (5.2, "Leave-one-out:\nsubtract row i's voxel\ncontribution, re-argmax", "#7c3aed"),
         (7.8, "Mean-FLIP score\n= avg #voxel decisions\nflipped per row", "#b45309"), (10.2, "Guarded Otsu on log-score\n(abstain if no valley)\n-> compile smaller bank", "#047857")]
for i, (x, t, c) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((x, 1.4), 2.3, 1.5, boxstyle="round,pad=0.05", fc=c, ec="w")); ax.text(x + 1.15, 2.15, t, ha="center", va="center", color="w", fontsize=9)
    if i < len(boxes) - 1: ax.annotate("", xy=(boxes[i + 1][0] - 0.02, 2.15), xytext=(x + 2.32, 2.15), arrowprops=dict(arrowstyle="->", color="w"))
ax.text(6.5, 3.8, "Decision-FLIP pipeline - this repo's own reconstruction diagram (not the paper's figure)", ha="center", color="w", fontsize=11)
ax.text(6.5, 0.6, f"Result on synthetic data: {r['kept_rows']}/{r['dense_rows']} rows kept ({r['removed_pct']:.0f}% removed), zero-flip dead rows: {r['dead_rows_zero_flip']}", ha="center", color="#a7f3d0", fontsize=10)
fig.savefig("assets/architecture.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)
m = r["methods"]; fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#0b0f14"); ax.set_facecolor("#0b0f14")
names = list(m); import numpy as np; x = np.arange(len(names))
ax.bar(x - .2, [m[n]["miou_no_recovery"] for n in names], .4, label="no recovery", color="#34d399"); ax.bar(x + .2, [m[n]["miou_recovered"] for n in names], .4, label="+120-step finetune", color="#60a5fa")
ax.axhline(r["dense_miou"], c="#9ca3af", ls="--", label=f"dense {r['dense_miou']:.3f}"); ax.set_xticks(x); ax.set_xticklabels(names, color="w", fontsize=8); ax.set_ylim(0.6, 0.75)
ax.tick_params(colors="w"); ax.set_ylabel("val mIoU @ 260 rows (synthetic)", color="w"); ax.legend(facecolor="#111827", labelcolor="w", fontsize=8)
fig.savefig("assets/results.png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)
