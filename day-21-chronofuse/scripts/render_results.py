"""Renders the results stat-tile PNG. Kept deliberately as TWO separate
groups of tiles (paper-verified vs. this repo's own synthetic numbers),
per this project's standing convention for not visually conflating the
two when they diverge (introduced Day 19)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
BG = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#898781"

fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")

ax.text(6, 5.7, "ChronoFuse — Reported Results", ha="center", fontsize=16, fontweight="bold", color=INK)


def tile(x, y, w, h, big, label, color, sub=None):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                                    linewidth=2.2, edgecolor=color, facecolor="white")
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h * 0.62, big, ha="center", va="center", fontsize=21, fontweight="bold", color=color)
    ax.text(x + w / 2, y + h * 0.30, label, ha="center", va="center", fontsize=9, color=INK2, wrap=True)
    if sub:
        ax.text(x + w / 2, y + h * 0.10, sub, ha="center", va="center", fontsize=7.3, color=INK3)


ax.text(0.3, 4.85, "Paper-verified (arXiv:2609.26919 abstract)", fontsize=11, fontweight="bold", color=INK)
tile(0.3, 3.55, 2.6, 1.15, "71%", "of accuracy lost to latency\nrecovered on driving data", BLUE)
tile(3.1, 3.55, 2.6, 1.15, "9.3×", "sAP gain, extreme motion\n(EV-Flying: 20.95 vs 2.25)", BLUE)
tile(5.9, 3.55, 2.6, 1.15, "0.17M / 0.84ms", "params / latency added\nby ChronoFuse itself", BLUE)
tile(8.7, 3.55, 2.9, 1.15, "3 benchmarks", "1 Mpx driving data,\nFRED, EV-Flying", BLUE)

ax.text(0.3, 2.85, "This repo's own reconstruction (synthetic data — NOT comparable to the numbers above)",
        fontsize=11, fontweight="bold", color=INK)
tile(0.3, 1.55, 2.9, 1.15, "40.2%", "of old-way error recovered\n(center-distance, synthetic)", ORANGE,
     sub="stale 9.36px -> ChronoFuse 5.60px")
tile(3.4, 1.55, 2.9, 1.15, "352,726", "total params\n(ChronoFuse: 83,440 / 23.7%)", ORANGE)
tile(6.5, 1.55, 2.9, 1.15, "16 / 16", "unit + regression tests\npassing", ORANGE)
tile(9.55, 1.55, 2.15, 1.15, "24-frame", "GIF, real trained\ncheckpoint", ORANGE)

ax.text(6, 0.55, "See SOURCING.md — every number above is labeled paper-verified or this-repo-synthetic; never conflated.",
        ha="center", fontsize=8.5, color=INK3, style="italic")

fig.tight_layout()
fig.savefig("assets/results_stat_tiles.png", dpi=170, facecolor=BG)
print("saved assets/results_stat_tiles.png")
