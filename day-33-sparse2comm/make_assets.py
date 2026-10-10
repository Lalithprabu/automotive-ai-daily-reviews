"""Render assets/architecture.png (this repo's own diagram, not the paper's figure) and assets/results.png from real run data."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG, FG, GR = "#0e1117", "#e6edf3", "#30363d"
C = dict(ego="#8b949e", old="#f0883e", new="#ff4d6d", msg="#58a6ff", ok="#3fb950")

def box(ax, x, y, w, h, t, col, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=BG, ec=col, lw=2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color=FG, fontsize=fs)

def arrow(ax, a, b, col=FG):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=col, lw=1.6))

fig, ax = plt.subplots(figsize=(14, 5.2), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 5.2)
ax.text(7, 4.95, "Sparse2comm-style pipeline (this repo's reconstruction)", color=FG, ha="center", fontsize=13, weight="bold")
box(ax, .2, 3.2, 2.2, 1.0, "Collaborator BEV\nfeatures (frames t-d-1, t-d)", C["msg"])
box(ax, .2, 1.6, 2.2, 1.0, "Ego BEV features\n(blind zone)", C["ego"])
box(ax, 2.9, 3.2, 2.0, 1.0, "Random mask-\nsampling of\nforeground cells", C["msg"], 8.5)
ax.text(3.9, 2.75, "bandwidth ~1-2% of cells\n+ packet loss", color=C["msg"], ha="center", fontsize=7.5)
box(ax, 5.3, 3.2, 2.0, 1.0, "1  Sparse Feature\nDecoder\n(completeness)", C["new"], 8.5)
box(ax, 7.6, 3.2, 2.0, 1.0, "2  Latency-Aware\nAlignment: flow\n(temporal)", C["new"], 8.5)
box(ax, 9.9, 3.2, 2.0, 1.0, "3  Self-Calibrating\nFusion: offset\nsearch + gate", C["new"], 8.5)
box(ax, 12.2, 2.4, 1.6, 1.0, "Detection\nheatmap\n(time t)", C["ok"], 8.5)
for a, b in (((2.4, 3.7), (2.9, 3.7)), ((4.9, 3.7), (5.3, 3.7)), ((7.3, 3.7), (7.6, 3.7)), ((9.6, 3.7), (9.9, 3.7))): arrow(ax, a, b)
arrow(ax, (11.9, 3.7), (12.5, 3.4)); arrow(ax, (2.4, 2.1), (10.9, 3.2), C["ego"]); arrow(ax, (2.4, 2.1), (12.2, 2.7), C["ego"])
ax.text(7, 1.0, "OLD WAY: zero-filled sparse message -> plain gated fusion (no 1, 2, 3).   Losses: focal (det) + recon + flow-L1 + label-free calibration.", color=C["old"], ha="center", fontsize=9)
ax.text(7, .5, "Every dimension is this repo's own default (paper read at abstract level only) - see SOURCING.md", color=FG, ha="center", fontsize=8.5)
fig.savefig("assets/architecture.png", dpi=130, facecolor=BG); plt.close(fig)

R = json.load(open("checkpoints/sweeps.json")); mx = R["mixed"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)
names = [("ego_only", "ego only", C["ego"]), ("old_way", "old way", C["old"]), ("sparse2comm", "Sparse2comm-style", C["new"])]
ax = axs[0]
for i, (k, n, c) in enumerate(names):
    ax.bar(i - .2, mx[k]["AP@r1"], .4, color=c); ax.bar(i + .2, mx[k]["AP@r2"], .4, color=c, alpha=.55)
    ax.text(i - .2, mx[k]["AP@r1"] + .01, f"{mx[k]['AP@r1']:.3f}", ha="center", color=FG, fontsize=8); ax.text(i + .2, mx[k]["AP@r2"] + .01, f"{mx[k]['AP@r2']:.3f}", ha="center", color=FG, fontsize=8)
ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n, _ in names]); ax.set_ylim(0, 1); ax.set_title("Mixed degradations: AP@r1 (solid) / AP@r2 (light)", color=FG, fontsize=9)
ax = axs[1]
for k, n, c in names: ax.plot(range(4), R["delay"][k], "o-", c=c, lw=2, label=n)
ax.set_xlabel("transmission delay (frames)", color=FG); ax.set_title("AP@r2 vs delay", color=FG, fontsize=9); ax.set_ylim(.5, 1); ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GR, fontsize=8)
ax = axs[2]
for k, n, c in names[1:]: ax.plot([.1, .2, .3, .45, .6], R["keep"][k], "o-", c=c, lw=2, label=n)
ax.axhline(R["keep"]["ego_only"][0], c=C["ego"], ls="--", label="ego only"); ax.set_xlabel("keep ratio of foreground cells", color=FG); ax.set_title("AP@r2 vs bandwidth", color=FG, fontsize=9); ax.set_ylim(.5, 1); ax.legend(facecolor=BG, labelcolor=FG, edgecolor=GR, fontsize=8)
for a in axs: a.set_facecolor(BG); a.tick_params(colors=FG); [s.set_color(GR) for s in a.spines.values()]; a.grid(color=GR, lw=.4)
fig.suptitle("Synthetic held-out results - this repo's simulator, centre-distance AP (NOT the paper's AP@IoU on DAIR-V2X/OpenV2V/V2V4Real)", color=FG, fontsize=10)
fig.savefig("assets/results.png", dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(fig)
