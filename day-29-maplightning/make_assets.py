"""Render assets/architecture.png (this repo's own diagram, NOT the paper's figure) and assets/results.png from outputs/results.json."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, FG, NEW, OLD = "#0f1419", "#e5e7eb", "#60a5fa", "#f87171"
Path("assets").mkdir(exist_ok=True)


def box(ax, x, y, w, h, t, c, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=c, ec="none", alpha=.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=fs, weight="bold")


def arrow(ax, x1, y1, x2, y2, c=FG):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=c, lw=1.6))


fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2)
ax.text(.2, 5.9, "OLD WAY — dense BEV (needs camera calibration)", color=OLD, fontsize=12, weight="bold")
box(ax, .2, 4.5, 1.6, .9, "Camera\nimage", "#374151"); box(ax, 2.3, 4.5, 1.6, .9, "CNN\nstem", "#4b5563")
box(ax, 4.4, 4.5, 2.4, .9, "Lift to dense BEV grid\n(nominal intrinsics + extrinsics)", OLD, 8); box(ax, 7.3, 4.5, 1.7, .9, "384 BEV\ntokens", "#4b5563")
box(ax, 9.5, 4.5, 2, .9, "Set decoder\n(6 queries)", "#6b7280"); box(ax, 11.8, 4.5, 1, .9, "Polylines", "#374151", 8)
for x1, x2 in ((1.8, 2.3), (3.9, 4.4), (6.8, 7.3), (9, 9.5), (11.5, 11.8)): arrow(ax, x1, 4.95, x2, 4.95)
ax.text(.2, 3.7, "NEW WAY — MapLightning-style 1D map tokens (no camera parameters)", color=NEW, fontsize=12, weight="bold")
box(ax, .2, 2.1, 1.6, .9, "Camera\nimage", "#374151"); box(ax, 2.3, 2.1, 1.6, .9, "CNN stem +\n2x2 patchify", "#4b5563")
box(ax, 4.4, 2.5, 2.4, .6, "72 image tokens", "#4b5563"); box(ax, 4.4, 1.7, 2.4, .6, "24 learned map tokens", NEW)
box(ax, 7.3, 1.9, 1.9, 1.2, "Joint FULL\nself-attention\nx3 layers", NEW, 9); box(ax, 9.5, 2.1, 1.1, .9, "drop image\ntokens", "#b45309", 8)
box(ax, 10.8, 2.1, 1.3, .9, "Set decoder\n(full x-attn)", "#6b7280", 8); box(ax, 12.2, 2.1, .7, .9, "Poly-\nlines", "#374151", 7)
for x1, y1, x2, y2 in ((1.8, 2.55, 2.3, 2.55), (3.9, 2.55, 4.4, 2.8), (6.8, 2.8, 7.3, 2.7), (6.8, 2.0, 7.3, 2.3), (9.2, 2.55, 9.5, 2.55), (10.6, 2.55, 10.8, 2.55), (12.1, 2.55, 12.2, 2.55)): arrow(ax, x1, y1, x2, y2)
ax.text(.2, .8, "Same stem + same set-decoder + same loss in both rows: the only difference is HOW image evidence reaches the queries.\nDiagram of this repo's reconstruction; layer sizes are this repo's defaults, not the paper's.", color="#9ca3af", fontsize=9)
fig.savefig("assets/architecture.png", dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(fig)

if Path("outputs/results.json").exists():
    r = json.load(open("outputs/results.json")); ss = list(r["sweep"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG)
    for ax, key, lab in ((axs[0], "err_m", "matched polyline error (m) ↓"), (axs[1], "f1_1m", "F1 @ 1 m ↑")):
        ax.set_facecolor("#161c24")
        for n, c, l in (("bev_baseline", OLD, "dense BEV (nominal calib)"), ("maplightning", NEW, "1D map tokens")):
            ax.plot([float(s) for s in ss], [r["sweep"][s][n][key] for s in ss], "o-", color=c, label=l, lw=2)
        ax.set_xlabel("extrinsic perturbation (deg std)", color=FG); ax.set_ylabel(lab, color=FG); ax.tick_params(colors=FG)
        ax.legend(facecolor=BG, labelcolor=FG, fontsize=8)
    fig.suptitle("Synthetic data — this repo's own results, not the paper's nuScenes/AV2 numbers", color="#9ca3af", fontsize=9)
    fig.savefig("assets/results.png", dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(fig)
print("assets done")
