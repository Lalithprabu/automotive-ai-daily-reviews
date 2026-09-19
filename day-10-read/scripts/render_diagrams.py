"""Renders assets/architecture_diagram.png for the READ package.

No results.png is produced for this day: no hard PDMS/collision-rate
numbers were recoverable from any source, and this project's policy is
to never fabricate a results chart from invented numbers (see README
"Sourcing note").
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
COLOR_AQUA = "#1baf7a"
COLOR_YELLOW = "#eda100"
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_INK_MUTED = "#898781"


def box(ax, xy, w, h, text, color, text_color="white", fontsize=9.5, sub=None):
    x, y = xy
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                        linewidth=1.2, edgecolor=COLOR_INK, facecolor=color, zorder=3)
    ax.add_patch(b)
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=fontsize,
                 color=text_color, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=fontsize - 2.5,
                 color=text_color, zorder=4, wrap=True)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                 color=text_color, fontweight="bold", zorder=4)
    return (x, y, w, h)


def arrow(ax, start, end, color=COLOR_INK_SECONDARY, style="-|>", lw=1.6, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, linewidth=lw,
                         color=color, zorder=2, connectionstyle=connectionstyle)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13, 8.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(6.5, 8.15, "READ — Risk-Informed Fields for End-to-End Autonomous Driving",
            ha="center", fontsize=15, color=COLOR_INK, fontweight="bold")
    ax.text(6.5, 7.78, "arXiv:2609.12371 · Tsinghua University · learned continuous spatiotemporal risk field r(x, y, t)",
            ha="center", fontsize=9.5, color=COLOR_INK_MUTED)

    # ---- Row 1: inputs ----
    b_bev = box(ax, (0.5, 6.3), 2.6, 0.95, "BEV Raster", COLOR_BLUE, sub="lane / occupancy grid  [B,C,H,W]")
    b_hist = box(ax, (3.5, 6.3), 2.6, 0.95, "Agent History", COLOR_BLUE, sub="(x,y,heading,speed) x T  [B,A,T,F]")

    # ---- Row 2: scene encoder ----
    b_cnn = box(ax, (0.5, 4.95), 2.6, 0.85, "BEV-Raster CNN", COLOR_AQUA, sub="conv stack -> map tokens")
    b_gru = box(ax, (3.5, 4.95), 2.6, 0.85, "Agent-History GRU", COLOR_AQUA, sub="per-agent token")
    arrow(ax, (1.8, 6.3), (1.8, 5.8))
    arrow(ax, (4.8, 6.3), (4.8, 5.8))

    b_fuse = box(ax, (1.5, 3.75), 3.5, 0.85, "Shallow Fusion Transformer", COLOR_ORANGE,
                 sub="SceneEncoder: map + agent tokens -> scene context")
    arrow(ax, (1.8, 4.95), (2.6, 4.6))
    arrow(ax, (4.8, 4.95), (4.2, 4.6))

    b_tokens = box(ax, (2.35, 2.75), 1.8, 0.65, "scene_tokens", COLOR_INK_SECONDARY, fontsize=9, sub=None)
    arrow(ax, (3.25, 3.75), (3.25, 3.4))

    # ---- query coordinates branch ----
    b_query = box(ax, (7.2, 6.3), 2.6, 0.95, "Query (x, y, t)", COLOR_BLUE, sub="candidate probe / waypoint")
    b_fourier = box(ax, (7.2, 4.95), 2.6, 0.85, "Fourier Positional Enc.", COLOR_AQUA,
                     sub="max freq 4.0 (bug-fixed, see README)")
    arrow(ax, (8.5, 6.3), (8.5, 5.8))

    # ---- Risk Field Network cross-attention ----
    b_risk = box(ax, (4.7, 1.55), 5.6, 1.1, "RiskFieldNetwork: Cross-Attention", COLOR_ORANGE,
                 sub="query cross-attends scene_tokens x2 layers -> risk_head -> sigmoid")
    arrow(ax, (3.25, 2.75), (5.6, 2.65), connectionstyle="arc3,rad=-0.15")
    arrow(ax, (8.5, 4.95), (7.6, 2.65), connectionstyle="arc3,rad=0.15")

    b_out = box(ax, (5.7, 0.55), 3.6, 0.65, "risk(x, y, t)  in [0, 1]", COLOR_YELLOW, text_color=COLOR_INK)
    arrow(ax, (7.5, 1.55), (7.5, 1.2))

    # ---- trajectory refinement branch ----
    b_refine = box(ax, (10.1, 1.55), 2.6, 1.55, "Differentiable\nTrajectory Refinement",
                    COLOR_INK, sub="grad descent on (x,y)\nvia autograd through\nthe fixed risk field")
    arrow(ax, (10.3, 2.1), (10.3, 2.1))
    arrow(ax, (7.5, 0.87), (9.9, 1.9), connectionstyle="arc3,rad=-0.2")
    arrow(ax, (11.4, 3.1), (11.4, 6.3), connectionstyle="arc3,rad=0.35", color=COLOR_INK_MUTED)

    b_planner = box(ax, (9.7, 6.3), 3.0, 0.95, "Planner / VLA\nIntegration Point", COLOR_INK_SECONDARY,
                     sub="risk-refined trajectory\nconsumed downstream")

    ax.text(0.5, 0.15,
            "Old way: fixed-shape Gaussian \"safety bubble\" per obstacle, re-discretized every perception update, no gradient path to the trajectory.",
            fontsize=8.3, color=COLOR_INK_MUTED)
    ax.text(0.5, -0.1, "", fontsize=1)

    legend_elems = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_BLUE, markersize=10, label="Inputs"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_AQUA, markersize=10, label="Encoders"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_ORANGE, markersize=10, label="Fusion / Risk Field"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_YELLOW, markersize=10, label="Output"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=False, fontsize=8.5,
              bbox_to_anchor=(1.0, -0.02))

    plt.tight_layout()
    fig.savefig("assets/architecture_diagram.png", dpi=170, facecolor=COLOR_SURFACE)
    print("Saved assets/architecture_diagram.png")


if __name__ == "__main__":
    main()
