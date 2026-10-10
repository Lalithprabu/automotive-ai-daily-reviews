"""assets/architecture.png (this repo's own diagram, not the paper's figure) and assets/results.png (from results.json)."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG, FG = "#0f1419", "#e6edf3"

def box(ax, x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=c, ec=FG, lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="white", fontsize=8.5)

def arrow(ax, a, b): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=FG, lw=1.4))

fig, ax = plt.subplots(figsize=(13, 5.2), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2)
ax.text(0.2, 4.95, "MoCAR-style reconstruction (this repo's own diagram — not the paper's figure)", color=FG, fontsize=11, weight="bold")
box(ax, 0.2, 3.2, 2.0, 0.9, "History xy+heading\n(20 steps → 4 segments)", "#1f6feb"); box(ax, 0.2, 1.6, 2.0, 0.9, "Map lane polylines\n(PointNet tokens)", "#8957e5")
box(ax, 2.8, 3.2, 2.2, 0.9, "Motion-Code Tokenizer\nsegment geometry +\nframe transition → z", "#238636"); arrow(ax, (2.2, 3.65), (2.8, 3.65))
box(ax, 5.6, 3.2, 2.4, 0.9, "Code tokens +\nPose embedding\n(coordinate-aware)", "#9e6a03"); arrow(ax, (5.0, 3.65), (5.6, 3.65))
box(ax, 8.6, 2.4, 2.4, 1.7, "Causal decoder-only\nTransformer × M modes\n(cross-attn to map)", "#da3633"); arrow(ax, (8.0, 3.65), (8.6, 3.4)); arrow(ax, (2.2, 2.05), (8.6, 2.7))
box(ax, 11.4, 3.2, 1.5, 0.9, "next code ĉ\n(continuous)", "#238636"); arrow(ax, (11.0, 3.4), (11.4, 3.65))
ax.annotate("", xy=(6.8, 3.1), xytext=(12.15, 3.15), arrowprops=dict(arrowstyle="->", color="#ff7b72", lw=1.6, connectionstyle="arc3,rad=-0.35"))
ax.text(8.2, 0.95, "ĉ appended to latent sequence; end pose re-derived from decoded codes (no xy re-tokenization)", color="#ff7b72", fontsize=8.5)
box(ax, 2.8, 0.15, 5.0, 0.7, "Tokenizer decoder → chain segment poses → xy trajectory (3 modes + scores)", "#1f6feb"); arrow(ax, (11.4, 3.2), (7.8, 0.5))
plt.savefig("assets/architecture.png", dpi=110, facecolor=BG, bbox_inches="tight"); plt.close()

R = json.load(open("results.json"))["results"]
names = [("oneshot", "One-shot\n(old way)", "#8b949e"), ("mocar_nocoord", "MoCAR\nno coord-aware", "#d2a8ff"), ("mocar", "MoCAR\n(full)", "#ff7b72")]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=BG)
for ax, (key, ttl) in zip(axs, [("all", "All scenes"), ("turns", "Turn-heavy (left+right)"), ("junction_straight", "Junction, straight")]):
    ax.set_facecolor(BG); vals = [R[n][key]["minADE3"] for n, _, _ in names]
    bars = ax.bar([l for _, l, _ in names], vals, color=[c for *_, c in names])
    for b_, v in zip(bars, vals): ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", color=FG, fontsize=9)
    ax.set_title(f"{ttl}: minADE@3 (m, lower=better)", color=FG, fontsize=9); ax.tick_params(colors=FG, labelsize=8)
    for s in ax.spines.values(): s.set_color("#2b3440")
fig.suptitle("Synthetic held-out test (1,200 scenes) — NOT the paper's Argoverse numbers", color=FG, fontsize=10)
plt.savefig("assets/results.png", dpi=110, facecolor=BG, bbox_inches="tight")
