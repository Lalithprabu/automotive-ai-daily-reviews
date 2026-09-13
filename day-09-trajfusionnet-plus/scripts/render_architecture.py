"""Renders the TrajFusionNet+ architecture diagram as a PNG, following the
project's validated categorical palette (Reference Matrix §5)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SECOND, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"


def box(ax, xy, w, h, text, color, fontsize=9.5, text_color="white", lw=1.6, dashed=False):
    b = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=INK, facecolor=color, zorder=3,
        linestyle="dashed" if dashed else "solid",
    )
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, zorder=4, wrap=True, linespacing=1.35)
    return b


def arrow(ax, p0, p1, color=INK_SECOND, lw=1.8, style="-|>"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, zorder=2)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(12, 5.6), facecolor=SURFACE)
ax.set_xlim(0, 12)
ax.set_ylim(2.6, 8.4)
ax.axis("off")
ax.set_facecolor(SURFACE)

fig.text(0.5, 0.965, "TrajFusionNet+  —  Pedestrian Crossing-Intention Prediction",
          ha="center", fontsize=15, color=INK, weight="bold")
fig.text(0.5, 0.94, "arXiv:2609.10806  ·  builds on TrajFusionNet (arXiv:2508.19866) by adding the Graph Attention Module",
          ha="center", fontsize=9.5, color=INK_SECOND)

# ---- Inputs ----
box(ax, (0.3, 6.6), 2.1, 0.85, "Past trajectory\n(15 frames × 5-dim\nbbox + speed)", INK_MUTED, fontsize=8.5)
box(ax, (0.3, 5.2), 2.1, 0.85, "Observed + predicted\nframe overlays\n(bbox-rendered RGB)", INK_MUTED, fontsize=8.5)
box(ax, (0.3, 3.5), 2.1, 1.0, "Segmented scene\n(pedestrian + traffic\nelements → scene graph)", INK_MUTED, fontsize=8.5)

# ---- SAM branch (verified, base TrajFusionNet) ----
box(ax, (3.0, 6.5), 2.7, 1.1,
    "SAM — Sequence Attention\nTrajectory Transformer (8+8 layers)\n→ Classification Transformer (6L/8H)\nd_model=128  [VERIFIED]",
    BLUE, fontsize=8.3)
arrow(ax, (2.4, 7.0), (3.0, 7.05))

# ---- VAM branch (verified, base TrajFusionNet) ----
box(ax, (3.0, 5.05), 2.7, 1.1,
    "VAM — Visual Attention\n2× Large-Kernel-Attention CNN\n(VAN-B2 style, dual branch)\n[VERIFIED]",
    BLUE, fontsize=8.3)
arrow(ax, (2.4, 5.6), (3.0, 5.6))

# ---- GAM branch (new, reconstructed) ----
box(ax, (3.0, 3.35), 2.7, 1.3,
    "GAM — Graph Attention\n(NEW in TrajFusionNet+)\nPedestrian-centric graph:\nnode encoder → 2× multi-head\ngraph attention → hub readout\n[RECONSTRUCTED]",
    ORANGE, fontsize=8.2, dashed=True)
arrow(ax, (2.4, 4.0), (3.0, 4.0), color=ORANGE, lw=2.2)

# Branch projection labels
for y, label in [(7.0, "proj → 40-d"), (5.55, "proj → 40-d"), (3.95, "proj → 40-d")]:
    ax.text(5.85, y, label, fontsize=7.5, color=INK_SECOND, ha="left", va="center", style="italic")

# ---- Late fusion ----
box(ax, (7.2, 5.1), 2.0, 1.55, "Late-Fusion Trunk\nconcat(120-d)\n→ Dense 120→60\n→ Dense 60→2", AQUA,
    fontsize=8.8, text_color=INK)
arrow(ax, (5.7, 7.05), (7.2, 6.1), lw=1.6)
arrow(ax, (5.7, 5.6), (7.2, 5.85), lw=1.6)
arrow(ax, (5.7, 4.0), (7.2, 5.55), lw=1.6, color=ORANGE)

# ---- Outputs ----
box(ax, (9.7, 6.15), 2.0, 0.9, "Crossing-intention\nlogits [B, 2]\n(cross / no-cross)", YELLOW, fontsize=8.6, text_color=INK)
box(ax, (9.7, 4.95), 2.0, 0.9, "Predicted future\ntrajectory [B, 60, 5]", YELLOW, fontsize=8.6, text_color=INK)
arrow(ax, (9.2, 6.6), (9.7, 6.6))
arrow(ax, (9.2, 5.4), (9.7, 5.4))
arrow(ax, (5.7, 7.05), (9.7, 5.4), color="none")  # spacer (invisible) to keep layout balanced

# direct SAM->trajectory output path (trajectory head lives inside SAM, not the fusion trunk)
arrow(ax, (5.7, 6.8), (9.7, 5.35), lw=1.4, color=BLUE, style="-|>")

# ---- Legend ----
legend_handles = [
    mpatches.Patch(color=BLUE, label="Verified from TrajFusionNet full text (base branches)"),
    mpatches.Patch(color=ORANGE, label="New / reconstructed this cycle (GAM — abstract-level only)"),
    mpatches.Patch(color=AQUA, label="Fusion"),
    mpatches.Patch(color=YELLOW, label="Output heads"),
]
ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.14),
          ncol=2, fontsize=8.5, frameon=False)

plt.tight_layout(rect=[0, 0.06, 1, 0.90])
plt.savefig("assets/trajfusionnetplus_architecture.png", dpi=170, facecolor=SURFACE)
print("Saved assets/trajfusionnetplus_architecture.png")
