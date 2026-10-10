"""Render assets/architecture.png (this repo's own diagram, NOT the paper's figure) and
assets/results.png (from results.json produced by train.py)."""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, FG, C1, C2, C3, C4 = "#0d1117", "#c9d1d9", "#58a6ff", "#a371f7", "#3fb950", "#f0883e"


def box(ax, x, y, w, h, text, c, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc="#161b22", ec=c, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=FG, fontsize=fs)


def arrow(ax, x0, y0, x1, y1, c=FG):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=c, lw=1.5))


def architecture(path="assets/architecture.png"):
    fig, ax = plt.subplots(figsize=(14, 6.2), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 6.2)
    ax.text(0.2, 5.9, "MESSENGER-style memory scene-flow forecasting — this repo's reconstruction (abstract-level paper facts only)", color=FG, fontsize=11, weight="bold")
    box(ax, 0.2, 3.6, 2.3, 1.2, "Sweeps  P(t-K..t)\n(no point correspondence)", C1)
    box(ax, 3.1, 4.1, 2.6, 1.0, "PairFlowEstimator\n(observed frames s→s+1)\nflow f_s + state h_s", C1)
    box(ax, 6.3, 4.1, 2.6, 1.0, "MEMORY BUFFER (K=3)\n{P_s, f_s, h_s}", C2)
    box(ax, 3.1, 2.2, 2.6, 1.1, "1  RETRIEVE per age\nchain flows → advance to t\nsoft kNN match to P(t)", C2)
    box(ax, 6.3, 2.2, 2.6, 1.1, "2  UNCERTAINTY REWEIGHT\ng_a = σ(MLP[h, dist, |f_a−f_1|])", C4)
    box(ax, 9.5, 2.2, 2.2, 1.1, "3  FORECAST HEAD\nflow = f_1 + Δ ,  log σ", C3)
    box(ax, 9.5, 0.4, 2.2, 1.0, "Virtual frame\nP + flow  → new memory entry", C2)
    box(ax, 12.1, 2.2, 1.7, 1.1, "Rollout +0.5s…+2.0s\n(no new sweeps)", C3)
    arrow(ax, 2.5, 4.2, 3.1, 4.5, C1); arrow(ax, 5.7, 4.6, 6.3, 4.6, C2); arrow(ax, 7.6, 4.1, 5.0, 3.3, C2)
    arrow(ax, 5.7, 2.75, 6.3, 2.75, C4); arrow(ax, 8.9, 2.75, 9.5, 2.75, C3); arrow(ax, 11.7, 2.75, 12.1, 2.75, C3)
    arrow(ax, 10.6, 2.2, 10.6, 1.4, C2); ax.plot([9.5, 9.2, 9.2], [0.9, 0.9, 4.6], c=C2, lw=1.2, ls="--"); arrow(ax, 9.2, 4.6, 8.9, 4.6, C2)
    ax.text(0.2, 0.3, "Old way = same code with K=1, no gate: carry the last pairwise flow forward + learned residual.", color="#8b949e", fontsize=9)
    fig.savefig(path, facecolor=BG, dpi=110, bbox_inches="tight"); plt.close(fig)


def results(path="assets/results.png", src="results.json"):
    r = json.load(open(src)); o, n = r["old_way_K1"], r["messenger_K3"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=BG)
    hz = [1, 2, 3, 4]
    for a in ax:
        a.set_facecolor(BG); a.tick_params(colors="#8b949e"); [s.set_color("#30363d") for s in a.spines.values()]; a.grid(color="#21262d")
    ax[0].plot(hz, o["epe_cv_carry_moving"], c="#8b949e", marker="s", ls="--", label="constant-velocity carry (no learning)")
    ax[0].plot(hz, o["epe_moving"], c=C4, marker="o", label="old way (K=1)")
    ax[0].plot(hz, n["epe_moving"], c=C1, marker="o", label="MESSENGER-style (K=3 + gate)")
    ax[0].set_title("Extrapolation EPE on moving points (m)", color=FG); ax[0].set_xlabel("horizon (frames × 0.5 s)", color=FG); ax[0].legend(facecolor=BG, labelcolor=FG, edgecolor="#30363d", fontsize=8)
    red = [100 * (1 - b / a) for a, b in zip(o["epe_moving"], n["epe_moving"])]
    ax[1].bar(hz, red, color=[C3 if x > 0 else "#f85149" for x in red]); ax[1].axhline(0, c="#8b949e")
    ax[1].set_title("MESSENGER-style vs old way: % EPE reduction", color=FG); ax[1].set_xlabel("horizon", color=FG)
    for x, v in zip(hz, red): ax[1].text(x, v, f"{v:.1f}%", ha="center", va="bottom" if v > 0 else "top", color=FG, fontsize=9)
    fig.suptitle("Synthetic data, single seed — NOT the paper's nuScenes / Argoverse 2 numbers", color="#f0883e", fontsize=10)
    fig.savefig(path, facecolor=BG, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    architecture()
    if len(sys.argv) > 1 and sys.argv[1] == "all": results()
