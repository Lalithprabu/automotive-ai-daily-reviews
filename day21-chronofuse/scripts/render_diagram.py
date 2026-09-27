"""One-off script to render the architecture diagram PNG for this repo's
README/LinkedIn assets. Not part of the shipped package's runtime code."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
BG = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#898781"

fig, ax = plt.subplots(figsize=(13, 8.6), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 8.6)
ax.axis("off")


def box(x, y, w, h, text, color, fontsize=9.5, text_color=None, lw=2.2, fontweight="bold"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=lw, edgecolor=color, facecolor="white", zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color or INK, zorder=4, fontweight=fontweight, wrap=True)


def arrow(x0, y0, x1, y1, color=INK3, lw=1.8, style="-"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=style, shrinkA=2, shrinkB=2),
        zorder=2,
    )


ax.text(6.5, 8.3, "ChronoFuse (reconstruction) — Latency-Compensated Event Detection",
        ha="center", fontsize=15, color=INK, fontweight="bold")
ax.text(6.5, 7.95, "Reconstructed from arXiv:2609.26919 abstract-level description — architecture wiring is this project's own design (see SOURCING.md)",
        ha="center", fontsize=8.5, color=INK3, style="italic")

# --- Input event sequence ---
box(0.3, 6.5, 2.6, 1.0, "Event Frame\nSequence\n[B,T,2,H,W]\n(ON/OFF polarity)", ORANGE, fontsize=9)

# --- Shared backbone ---
box(3.4, 6.5, 2.5, 1.0, "EventFrameEncoder\n(shared weights,\nper timestep)\n3-scale pyramid", BLUE, fontsize=9)
arrow(2.9, 7.0, 3.4, 7.0)

# Cache vs current split
box(6.4, 6.65, 2.4, 0.75, "Current features F_t\n(last observed step)", BLUE, fontsize=8.3)
box(6.4, 5.55, 2.4, 0.75, "Cached features\n{F_t-1 ... F_t-K}", ORANGE, fontsize=8.3)
arrow(5.9, 7.15, 6.4, 7.0)
arrow(5.9, 6.85, 6.4, 5.95)

# Latency embedding
box(3.2, 3.05, 2.6, 0.9, "Latency Embedding\nsin/cos + MLP\nΔt (frames ahead) → [B,D]", YELLOW, fontsize=8.5)

# ChronoFuse block (star of the show)
box(6.2, 2.7, 2.9, 2.5,
    "ChronoFuseBlock  (× 3 scales)\n\n"
    "1. velocity = F_t − F_t-1\n"
    "2. causal temporal-attn\n    over cache → summary\n"
    "3. fuse[F_t, velocity, summary]\n"
    "4. gate = σ(MLP(latency))\n"
    "5. out = norm(F_t + gate·correction)",
    BLUE, fontsize=8.0, fontweight="normal")

arrow(7.6, 6.65, 7.6, 5.2, style="-")
arrow(7.6, 5.55, 7.6, 5.2, style="-")
arrow(5.8, 3.5, 6.2, 3.7)

# Predicted future features
box(9.9, 3.6, 2.6, 1.0, "Predicted future\nmulti-scale features\n(state @ t + Δt)", AQUA, fontsize=8.5)
arrow(9.1, 4.0, 9.9, 4.05)

# FPN merge
box(9.9, 2.1, 2.6, 1.0, "Top-down FPN merge\n(stride16→8→4)", AQUA, fontsize=8.5)
arrow(11.2, 3.6, 11.2, 3.1)

# Detection head
box(9.9, 0.6, 2.6, 1.0, "CenterDetectionHead\nheatmap / size / offset\n@ stride 4", AQUA, fontsize=8.5)
arrow(11.2, 2.1, 11.2, 1.6)

# Output
ax.text(9.9, 0.15, "→ Bounding boxes predicted for object state\n    at the moment the output is actually available",
        ha="left", fontsize=8.3, color=INK2)

# Ablation note
ax.text(
    0.3, 1.4,
    "Ablation baseline (\"old way\"): the SAME trained weights, with the\n"
    "ChronoFuse block bypassed entirely (identity pass-through of F_t).\n"
    "Represents a standard single-frame event detector with zero\n"
    "latency compensation — supervised to report the CURRENT\n"
    "(stale) position rather than the future one.",
    fontsize=8.3, color=INK2,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f3ef", edgecolor=INK3, linewidth=1),
)

# Legend
legend_handles = [
    mlines.Line2D([], [], color=ORANGE, marker='s', linestyle='None', markersize=10, markerfacecolor='white', markeredgewidth=2, label="Input / cached (past)"),
    mlines.Line2D([], [], color=BLUE, marker='s', linestyle='None', markersize=10, markerfacecolor='white', markeredgewidth=2, label="Backbone + ChronoFuse (core contribution)"),
    mlines.Line2D([], [], color=YELLOW, marker='s', linestyle='None', markersize=10, markerfacecolor='white', markeredgewidth=2, label="Latency conditioning"),
    mlines.Line2D([], [], color=AQUA, marker='s', linestyle='None', markersize=10, markerfacecolor='white', markeredgewidth=2, label="Output / detection path"),
]
ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig("assets/architecture_diagram.png", dpi=170, facecolor=BG)
print("saved assets/architecture_diagram.png")
