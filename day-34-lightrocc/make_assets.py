"""Render assets/architecture.png (this repo's own diagram, NOT the paper's figure) and assets/results.png (from checkpoints/results.json)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
C = dict(img="#8b949e", enc="#a371f7", q="#58a6ff", lift="#3fb950", gauss="#f0883e", prop="#f778ba", old="#f0883e")
os.makedirs("assets", exist_ok=True)


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=BG, ec=color, lw=2))
    ax.text(x + w / 2, y + h / 2, text, color=FG, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, color=FG):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.6))


def architecture():
    fig, ax = plt.subplots(figsize=(15, 6.2), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 6.2)
    ax.text(7.5, 5.85, "Instance-centric Gaussian occupancy forecasting (this repo's reconstruction)", color=FG, ha="center", fontsize=13, weight="bold")
    ax.text(7.5, 5.5, "OLD: dense BEV voxel grid  vs  NEW: ~6 object queries → 3D Gaussians → displaced in time", color="#8b949e", ha="center", fontsize=9)
    box(ax, 0.2, 3.2, 1.9, 1.2, "2 past camera\nfeature maps\n(12×24×48)", C["img"])
    box(ax, 2.6, 3.2, 1.9, 1.2, "Conv + Transformer\nimage encoder\n(288 tokens, d=64)", C["enc"])
    box(ax, 5.0, 3.2, 2.0, 1.2, "6 object queries\ncross-attend\n(2 decoder layers)", C["q"])
    box(ax, 7.5, 3.2, 2.2, 1.2, "FORWARD LIFTING\n(u,v) + depth + pinhole\n→ 3D center", C["lift"])
    box(ax, 10.2, 3.2, 2.0, 1.2, "3 anisotropic 3D\nGaussians / instance\n+ yaw, size", C["gauss"])
    box(ax, 12.6, 3.2, 2.2, 1.2, "displacements\nΔx,Δy,Δyaw × 4 steps\n→ rigid propagation", C["prop"])
    for x0, x1 in [(2.1, 2.6), (4.5, 5.0), (7.0, 7.5), (9.7, 10.2), (12.2, 12.6)]:
        arrow(ax, x0, 3.8, x1, 3.8)
    box(ax, 5.0, 1.55, 9.8, 0.9, "noisy-OR over Gaussians & instances → occupancy probability at every voxel, t = 0 … +2 s  (one forward pass)", C["gauss"], 9)
    arrow(ax, 13.7, 3.2, 13.7, 2.45, C["prop"]); arrow(ax, 11.2, 3.2, 11.2, 2.45, C["gauss"])
    box(ax, 2.6, 0.2, 4.4, 0.9, "OLD WAY: 16×16 BEV queries → conv upsample\n→ (T+1)×Z logits per 32×32 cell", C["old"], 8.5)
    arrow(ax, 3.5, 3.2, 3.5, 1.1, C["old"])
    fig.savefig("assets/architecture.png", dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(fig)


def results():
    r = json.load(open("checkpoints/results.json"))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)
    h = [0, 0.5, 1.0, 1.5, 2.0]
    spec = [("Voxel IoU (↑)", "iou"), ("Instance F1 @2 m (↑)", "f1"), ("Matched-center error, m (↓)", "ade")]
    for ax, (ttl, key) in zip(axs, spec):
        ax.set_facecolor(BG); ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.plot(h, r["dense"][key], color=C["old"], marker="o", alpha=0.5, label="dense BEV, same 2.5k steps")
        if "dense_long" in r:
            ax.plot(h, r["dense_long"][key], color=C["old"], marker="s", label="dense BEV, 7.5k steps (~equal wall-clock)")
        ax.plot(h, r["lightrocc"][key], color=C["q"], marker="o", label="instance Gaussians (new)")
        if key == "iou":
            ax.plot(h, r["lightrocc_zero_disp_iou"], color="#8b949e", ls="--", marker=".", label="new, displacement off")
        ax.set_title(ttl, color=FG, fontsize=10); ax.set_xlabel("forecast horizon (s)", color=FG)
        ax.grid(color=GRID, lw=0.5)
    axs[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    fig.suptitle("This repo's synthetic benchmark (600 held-out toy scenes, 1 seed) — NOT the paper's nuScenes results", color=FG, fontsize=11)
    fig.savefig("assets/results.png", dpi=130, facecolor=BG, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    architecture()
    if os.path.exists("checkpoints/results.json"): results()
    print("assets rendered")
