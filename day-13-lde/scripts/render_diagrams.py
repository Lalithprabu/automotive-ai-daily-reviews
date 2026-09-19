"""
Renders assets/architecture_diagram.png -- a schematic of the LDE
Mean-Teacher collaborative-perception self-training pipeline. No results
chart is produced here (see README "Sourcing note": no paper-attributed
numeric results were recoverable, so none are fabricated or plotted).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def box(ax, xy, w, h, text, color, text_color="white", fontsize=9.5, lw=0):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                 linewidth=lw, edgecolor=INK, facecolor=color, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4, wrap=True)


def arrow(ax, p1, p2, color=INK_SECONDARY, style="-|>", lw=1.6, ls="solid"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, color=color,
                         linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6.5, 7.6, "LDE: Collaborative Perception for Automated Model Adaptation",
            ha="center", fontsize=15, color=INK, fontweight="bold")
    ax.text(6.5, 7.15, "Mean-Teacher self-training driven by a bandwidth-budgeted V2X link",
            ha="center", fontsize=10.5, color=INK_SECONDARY)

    # -- Ego (student) branch --
    box(ax, (0.4, 5.6), 2.3, 0.8, "Ego Vehicle\nRaw BEV Input", INK_MUTED)
    box(ax, (0.4, 4.2), 2.3, 0.8, "BEVDetector\n(STUDENT)", BLUE)
    box(ax, (0.4, 2.8), 2.3, 0.8, "Student cls/reg\nheads (logits)", BLUE)

    arrow(ax, (1.55, 5.6), (1.55, 5.0))
    arrow(ax, (1.55, 4.2), (1.55, 3.6))

    # -- Collaborator (teacher) branch --
    box(ax, (10.3, 5.6), 2.3, 0.8, "Collaborator Vehicle\nRaw BEV Input", INK_MUTED)
    box(ax, (10.3, 4.2), 2.3, 0.8, "BEVDetector\n(TEACHER, EMA)", ORANGE)
    arrow(ax, (11.45, 5.6), (11.45, 5.0))

    box(ax, (10.3, 2.9), 2.3, 0.7, "AdaptiveFeatureGate\n(top-k, spatial only,\nno channel mixing)", YELLOW,
        text_color=INK, fontsize=8.5)
    arrow(ax, (11.45, 4.2), (11.45, 3.6))

    box(ax, (10.3, 1.7), 2.3, 0.7, "FoVAligner\n(affine warp into\nego frame + range mask)", AQUA,
        text_color=INK, fontsize=8.5)
    arrow(ax, (11.45, 2.9), (11.45, 2.4))

    box(ax, (10.3, 0.5), 2.3, 0.7, "Teacher's OWN\ncls_head decodes\npseudo-labels", ORANGE, fontsize=8.5)
    arrow(ax, (11.45, 1.7), (11.45, 1.2))

    # -- Curriculum + fusion in the middle --
    box(ax, (4.9, 0.5), 3.2, 0.75, "CurriculumScheduler\ncosine threshold 0.90 -> 0.50", YELLOW,
        text_color=INK, fontsize=9)
    arrow(ax, (10.3, 0.85), (8.1, 0.85))

    box(ax, (4.9, 1.7), 3.2, 0.8, "Confident pseudo-labels\n(gated AND in-FoV AND\nabove curriculum threshold)", "#ffffff",
        text_color=INK, fontsize=8.5, lw=1.4)
    arrow(ax, (6.5, 1.25), (6.5, 1.7))

    box(ax, (4.9, 2.9), 3.2, 0.8, "self_training_loss\n(class_balanced_bce_loss)", INK, fontsize=9.5)
    arrow(ax, (6.5, 2.5), (6.5, 2.9))
    arrow(ax, (2.7, 3.2), (4.9, 3.2))   # student logits -> loss
    ax.text(3.7, 3.35, "student logits", fontsize=7.5, color=INK_SECONDARY, ha="center")

    arrow(ax, (6.5, 3.7), (2.1, 4.2), color=BLUE, lw=1.8)
    ax.text(4.1, 4.15, "gradient update", fontsize=7.5, color=BLUE, ha="center")

    # EMA update loop
    arrow(ax, (2.7, 4.6), (10.3, 4.6), color=ORANGE, lw=1.6, ls=(0, (5, 3)))
    ax.text(6.5, 4.8, "EMA update: teacher <- momentum * teacher + (1 - momentum) * student",
            fontsize=8.5, color=ORANGE, ha="center", style="italic")

    legend_items = [
        Line2D([0], [0], color=BLUE, lw=6, label="Student (ego, trained by SGD)"),
        Line2D([0], [0], color=ORANGE, lw=6, label="Teacher (EMA of student)"),
        Line2D([0], [0], color=YELLOW, lw=6, label="Bandwidth / curriculum gating"),
        Line2D([0], [0], color=AQUA, lw=6, label="FoV alignment"),
    ]
    ax.legend(handles=legend_items, loc="lower center", bbox_to_anchor=(0.5, -0.06),
              ncol=4, frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "assets", "architecture_diagram.png")
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
