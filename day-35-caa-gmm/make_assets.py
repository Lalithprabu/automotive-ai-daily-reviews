"""Render this repo's OWN architecture diagram and results chart (not the paper's figures) from results*.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, FG, MUT = "#0e1117", "#e6edf3", "#8b949e"
OLD, NEW, GT = "#f0883e", "#58a6ff", "#3fb950"
os.makedirs("assets", exist_ok=True)


def box(ax, x, y, w, h, txt, c, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=c + "33", ec=c, lw=1.5))
    ax.text(x + w / 2, y + h / 2, txt, color=FG, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, c=MUT):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4))


def architecture():
    fig, ax = plt.subplots(figsize=(14, 6.6), facecolor=BG)
    ax.set_facecolor(BG); ax.set_xlim(0, 14); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.text(0.2, 6.35, "CAA-GMM (this repo's reconstruction of arXiv:2610.09174's described design)", color=FG, fontsize=13, weight="bold")
    ax.text(0.2, 6.02, "Old way (top): raster CNN -> flatten -> MLP.   New (bottom): neighbour tokens + two lightweight attentions + 3-way gate.", color=MUT, fontsize=9)
    # old way
    box(ax, 0.3, 4.55, 1.9, 0.8, "raster\n(4,64,64)", OLD); box(ax, 2.6, 4.55, 2.0, 0.8, "3x conv\nflatten", OLD)
    box(ax, 5.0, 4.55, 1.8, 0.8, "ego GRU\n+ concat", OLD); box(ax, 7.2, 4.55, 1.8, 0.8, "MLP", OLD)
    box(ax, 11.4, 4.55, 2.3, 0.8, "GMM head\nK=6 modes", OLD)
    for a, b in [((2.2, 4.95), (2.6, 4.95)), ((4.6, 4.95), (5.0, 4.95)), ((6.8, 4.95), (7.2, 4.95)), ((9.0, 4.95), (11.4, 4.95))]:
        arrow(ax, a, b, OLD)
    ax.text(9.2, 5.1, "neighbours only as pixels", color=OLD, fontsize=8)
    # new
    box(ax, 0.3, 2.9, 1.9, 0.8, "raster\n(4,64,64)", NEW); box(ax, 2.6, 2.9, 2.0, 0.8, "3x conv -> 64\ncell tokens", NEW)
    box(ax, 0.3, 1.5, 1.9, 0.8, "ego history\n(8x4) GRU", NEW); box(ax, 0.3, 0.3, 1.9, 0.8, "6 neighbour\nhistories GRU", NEW)
    box(ax, 2.6, 0.3, 2.0, 0.8, "+ null token\n+ key mask", NEW)
    box(ax, 5.3, 0.9, 2.2, 1.0, "interaction\nattention\n(agent <- neighbours)", NEW, 8)
    box(ax, 5.3, 2.6, 2.2, 1.0, "context\nattention\n(agent <- map cells)", NEW, 8)
    box(ax, 8.1, 1.6, 2.0, 1.1, "3-way gate\nmotion | interact.\n| context", NEW, 8)
    box(ax, 10.6, 1.6, 1.4, 1.1, "residual\nMLP+LN", NEW, 8); box(ax, 12.3, 1.6, 1.5, 1.1, "GMM head\nK=6 modes", NEW, 8)
    for a, b in [((2.2, 3.3), (2.6, 3.3)), ((4.6, 3.3), (5.3, 3.1)), ((2.2, 1.9), (5.3, 1.5)), ((2.2, 0.7), (2.6, 0.7)),
                 ((4.6, 0.7), (5.3, 1.1)), ((7.5, 1.4), (8.1, 1.9)), ((7.5, 3.1), (8.1, 2.4)), ((10.1, 2.15), (10.6, 2.15)),
                 ((12.0, 2.15), (12.3, 2.15))]:
        arrow(ax, a, b, NEW)
    ax.text(7.2, 0.15, "context query = ego state + interaction context (traffic first, map second)", color=MUT, fontsize=8)
    fig.savefig("assets/architecture.png", dpi=110, facecolor=BG, bbox_inches="tight")


def results():
    R = {t: json.load(open(f"results{t}.json")) for t in ("", "_aug") if os.path.exists(f"results{t}.json")}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=BG)
    for a in axs:
        a.set_facecolor(BG); a.tick_params(colors=MUT)
        for s in a.spines.values(): s.set_color("#30363d")
    ps = ["0.0", "0.3", "0.6", "1.0"]
    styles = {("", "baseline"): (OLD, "--"), ("", "caagmm"): (NEW, "--"), ("_aug", "baseline"): (OLD, "-"), ("_aug", "caagmm"): (NEW, "-")}
    for (tag, m), (c, ls) in styles.items():
        if tag not in R: continue
        lab = ("raster baseline" if m == "baseline" else "CAA-GMM") + (" + outage aug" if tag else " (clean-trained)")
        axs[0].plot([float(p) for p in ps], [R[tag]["robustness"][p][m]["minADE"] for p in ps], ls, color=c, marker="o", label=lab)
        axs[1].plot([float(p) for p in ps], [R[tag]["robustness"][p][m]["top1ADE"] for p in ps], ls, color=c, marker="o", label=lab)
    axs[0].set_title("minADE@6 vs. neighbour-message drop probability (synthetic test set)", color=FG, fontsize=10)
    axs[1].set_title("top-1 ADE vs. drop probability", color=FG, fontsize=10)
    for a in axs:
        a.set_xlabel("P(neighbour message dropped)", color=MUT); a.set_ylabel("metres", color=MUT)
    axs[0].legend(facecolor=BG, labelcolor=FG, fontsize=8, edgecolor="#30363d")
    fig.suptitle("Day 35 reconstruction results - synthetic data, 1 seed, NOT the paper's nuScenes/Argoverse 2 numbers", color=FG, fontsize=10)
    fig.savefig("assets/results.png", dpi=110, facecolor=BG, bbox_inches="tight")


if __name__ == "__main__":
    architecture(); results()
