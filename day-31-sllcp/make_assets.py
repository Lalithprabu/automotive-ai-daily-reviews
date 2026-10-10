"""Render assets/architecture.png (this repo's own diagram, not the paper's) and assets/results.png from real run data."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, FG = "#0f1419", "#e6edf3"
fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec=FG, lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=9)
def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=FG, lw=1.5))
box(0.2, 3.2, 2.2, 1.2, "Scene + ego plan\n(camera/BEV + trajectory)", "#30363d")
box(3.0, 3.2, 2.4, 1.2, "Frozen predictor\n(VLM in paper; MLP here)\n-> p(safe), p(unsafe)", "#1f6feb")
box(6.0, 3.2, 2.2, 1.2, "score s(x,y)=1-p_y(x)", "#8957e5")
box(3.0, 0.4, 2.4, 1.2, "Calibration set (per label)\nscores + scene embeddings", "#238636")
box(6.0, 0.4, 2.4, 1.2, "Kernel weights w_i(x)\nfrom scene similarity", "#238636")
box(9.0, 1.8, 2.2, 1.4, "Weighted (1-a) quantile\nper label  ->  q_y(x)", "#9e6a03")
box(9.0, 3.6, 3.7, 1.0, "Prediction set {y : s(x,y) <= q_y(x)}\nunsafe in set -> FLAG", "#da3633")
arrow(2.4, 3.8, 3.0, 3.8); arrow(5.4, 3.8, 6.0, 3.8); arrow(5.4, 1.0, 6.0, 1.0); arrow(8.4, 1.0, 9.0, 2.2); arrow(8.2, 3.7, 9.0, 4.0); arrow(10.1, 3.2, 10.1, 3.6)
ax.text(6.5, 4.85, "SLLCP calibration layer (this repo's own diagram - not the paper's figure)", color=FG, ha="center", fontsize=11)
fig.savefig("assets/architecture.png", dpi=110, facecolor=BG); plt.close(fig)

r = json.load(open("checkpoints/results.json"))
names = [k for k in r if not k.startswith("_")]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG)
cols = ["#8b949e", "#58a6ff", "#ff7b72"]
for a in axs: a.set_facecolor(BG); a.tick_params(colors=FG); [s.set_color("#2b3440") for s in a.spines.values()]
regs = ["clear", "dense", "occluded"]
for j, n in enumerate(names):
    axs[0].bar([i + j * 0.27 for i in range(3)], [r[n][f"recall_{g}"] for g in regs], 0.26, color=cols[j], label=n)
    axs[1].bar([i + j * 0.27 for i in range(3)], [r[n][f"fa_{g}"] for g in regs], 0.26, color=cols[j], label=n)
axs[0].axhline(0.9, color="#3fb950", ls=":"); axs[0].set_title("Unsafe-trajectory recall (target 0.90)", color=FG)
axs[1].set_title("False-alarm rate on safe trajectories (cost)", color=FG)
for a in axs: a.set_xticks([i + 0.27 for i in range(3)]); a.set_xticklabels(regs, color=FG)
axs[0].legend(facecolor=BG, edgecolor="#2b3440", labelcolor=FG, fontsize=8)
fig.suptitle("Synthetic held-out results (15,000 scenes, shifted test mix) - NOT the paper's CARLA numbers", color=FG, fontsize=10)
fig.savefig("assets/results.png", dpi=110, facecolor=BG); plt.close(fig)
print("assets written")
