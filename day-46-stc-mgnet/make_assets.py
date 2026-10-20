"""Render assets/architecture.png (this repo's own diagram, not the paper's figure) and assets/results.png from results.json."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG, FG = "#0d1117", "#e6edf3"
r = json.load(open("results.json"))
fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=BG); ax.set_facecolor(BG)
names = ["Old way\npose-only ST-GCN", "+ STC attention\n(pose-only)", "STC-MGNet\n(+ cross-modal gates)"]
vals = [r[k]["test_acc"] * 100 for k in ("baseline", "pose_stc", "stc_mgnet")]
b = ax.bar(names, vals, color=["#f85149", "#d29922", "#58a6ff"])
for x, v in zip(b, vals): ax.text(x.get_x() + x.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", color=FG)
ax.set_ylim(50, 100); ax.set_ylabel("test accuracy (synthetic, 1 seed)", color=FG); ax.tick_params(colors=FG, labelsize=8)
ax.set_title("This repo's synthetic result - NOT the paper's JAAD/PIE numbers", color=FG, fontsize=9)
[s.set_color("#30363d") for s in ax.spines.values()]; fig.tight_layout(); fig.savefig("assets/results.png", facecolor=BG, dpi=130)

fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.2)
def box(x, y, w, h, t, c): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="#30363d")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=8)
box(0.1, 2.9, 1.6, 0.9, "Pose graph\n(B,4,T,14)", "#21262d")
for i in range(3):
    x = 2.2 + i * 2.6
    box(x, 2.9, 1.5, 0.9, f"Stage {i+1}\nGraphConv\n+ temporal conv", "#1f6feb")
    box(x, 1.7, 1.5, 0.8, "STC attention\njoint | frame | channel\nlearnable fusion w", "#8957e5")
    box(x, 0.5, 1.5, 0.8, "Cross-modal\nchannel gate", "#bf8700")
    ax.annotate("", xy=(x + 0.75, 2.9), xytext=(x + 0.75, 2.5), arrowprops=dict(arrowstyle="<-", color=FG))
    ax.annotate("", xy=(x + 0.75, 1.3), xytext=(x + 0.75, 1.7), arrowprops=dict(arrowstyle="->", color=FG))
    if i < 2: ax.annotate("", xy=(x + 2.6, 3.35), xytext=(x + 1.5, 3.35), arrowprops=dict(arrowstyle="->", color=FG))
ax.annotate("", xy=(2.2, 3.35), xytext=(1.7, 3.35), arrowprops=dict(arrowstyle="->", color=FG))
box(0.1, 0.5, 1.6, 0.8, "Visual + ego\nembeddings", "#9e6a03"); ax.annotate("", xy=(2.2, 0.9), xytext=(1.7, 0.9), arrowprops=dict(arrowstyle="->", color=FG))
box(10.2, 2.9, 1.6, 0.9, "Pool -> MLP\ncross / no-cross", "#238636"); ax.annotate("", xy=(10.2, 3.35), xytext=(9.5, 3.35), arrowprops=dict(arrowstyle="->", color=FG))
ax.text(6, 4.05, "This repo's reconstruction of the STC-MGNet idea (abstract-level)", color=FG, ha="center", fontsize=10)
fig.savefig("assets/architecture.png", facecolor=BG, dpi=120)
