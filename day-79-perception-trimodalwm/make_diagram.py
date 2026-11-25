"""Render assets/architecture.png: single-sensor post-training vs the tri-sensor 'one video' world model."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
BG, FG, MUT = "#16181d", "#e8e8e8", "#9aa0a6"
COL = dict(rgb="#4f9dff", th="#ff8a3d", lid="#c86bfa", core="#2a2f38", attn="#ffb347", out="#5ad17a")


def box(ax, x, y, w, h, txt, fc, ec=None, fs=9, tc=FG):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec or fc, lw=1.4))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, a, b, col=MUT):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=col, lw=1.3))


fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.2)
ax.axis("off")
ax.text(0.2, 5.9, "Tri-modal driving world model (toy re-implementation of arXiv 2611.22669, abstract only)", color=FG,
        fontsize=12, weight="bold")
# --- top: old way
ax.text(0.2, 5.4, "OLD: one post-trained model per sensor", color=MUT, fontsize=10)
for i, (nm, c) in enumerate((("RGB", COL["rgb"]), ("thermal", COL["th"]), ("LiDAR depth", COL["lid"]))):
    y = 4.75 - i * 0.55
    box(ax, 0.3, y, 1.5, 0.42, nm + " video", c, fs=8.5, tc="#111")
    box(ax, 2.2, y, 2.6, 0.42, "world model (own copy)", COL["core"], ec=c, fs=8.5)
    box(ax, 5.2, y, 1.4, 0.42, "forecast", COL["core"], ec=c, fs=8.5)
    arrow(ax, (1.8, y + 0.21), (2.2, y + 0.21))
    arrow(ax, (4.8, y + 0.21), (5.2, y + 0.21))
ax.text(2.2, 3.35, "no path between sensors: fog-blind camera cannot borrow from thermal / LiDAR", color=MUT, fontsize=8)
# --- bottom: new way
ax.text(0.2, 2.95, "NEW: all three sensors tiled into ONE video, one model", color=FG, fontsize=10)
x = 0.3
for nm, c in (("RGB", COL["rgb"]), ("thermal x3", COL["th"]), ("LiDAR d,d,valid", COL["lid"])):
    box(ax, x, 2.2, 1.0, 0.5, nm, c, fs=7.5, tc="#111")
    x += 1.02
ax.text(0.3, 1.95, "canvas (3, 48, 192) per frame, 6 observed frames", color=MUT, fontsize=7.5)
box(ax, 3.6, 2.15, 1.5, 0.6, "Tokenizer\n3 convs, stride 4", COL["core"], ec=FG, fs=8)
box(ax, 5.45, 2.15, 1.45, 0.6, "+ 2-D position\n(12 x 48, no sensor id)", COL["core"], ec=FG, fs=7.5)
box(ax, 7.25, 2.15, 1.35, 0.6, "ConvGRU\nlocal motion", COL["core"], ec=FG, fs=8)
box(ax, 8.95, 2.15, 1.75, 0.6, "Canvas attention\nall 576 tokens", COL["attn"], fs=8, tc="#111")
box(ax, 11.05, 2.15, 1.7, 0.6, "Detokenizer\nx_t + delta", COL["core"], ec=FG, fs=8)
for a, b in (((3.36, 2.45), (3.6, 2.45)), ((5.1, 2.45), (5.45, 2.45)), ((6.9, 2.45), (7.25, 2.45)),
             ((8.6, 2.45), (8.95, 2.45)), ((10.7, 2.45), (11.05, 2.45))):
    arrow(ax, a, b, FG)
box(ax, 11.05, 1.0, 1.7, 0.55, "forecast canvas\nt+1", COL["out"], fs=8, tc="#111")
arrow(ax, (11.9, 2.15), (11.9, 1.55), FG)
ax.add_patch(FancyArrowPatch((11.05, 1.27), (4.35, 2.15), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
                             mutation_scale=11, color=COL["out"], lw=1.2, ls="--"))
ax.text(6.4, 1.32, "autoregressive: fed back for 8 future frames (2 s)", color=COL["out"], fontsize=8)
ax.text(8.95, 3.0, "the main route between sensors\n(3x3 ConvGRU only mixes at tile seams)", color=COL["attn"], fontsize=8)
arrow(ax, (9.8, 2.98), (9.8, 2.78), COL["attn"])
ax.text(0.3, 0.45, "Training: base model pretrained on clear-weather RGB -> post-trained per sensor (OLD) or on the tri canvas "
        "(NEW), all 6 weathers.\nLoss: L1 on RGB / thermal; LiDAR L1 only on measured returns (x4) + validity. "
        "Same 0.43M weights for 1 or 3 tiles.", color=MUT, fontsize=8)
fig.savefig(ROOT / "assets/architecture.png", facecolor=BG, bbox_inches="tight")
print("saved assets/architecture.png")
