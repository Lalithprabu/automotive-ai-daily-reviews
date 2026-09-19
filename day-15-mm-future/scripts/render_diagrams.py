"""Renders the two required visual assets for this project (see
AUTOMOTIVE AI TECHNICAL REFERENCE MATRIX, section 5):
  - assets/architecture_diagram.png : the MM-Future pipeline, boxes+arrows
  - assets/results.png              : a stat-tile row of the paper's own
                                       reported NAVSIM/HUGSIM numbers
                                       (recovered from a third-party mirror;
                                       see README "Sourcing note" — nothing
                                       here is fabricated)
Run: python scripts/render_diagrams.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#898781"


def box(ax, xy, w, h, text, color, text_color="white", fontsize=9, ls="-", lw=1.8, alpha=1.0):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=color, facecolor=color, alpha=alpha, linestyle=ls,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             color=text_color, fontsize=fontsize, wrap=True, fontweight="medium")
    return rect


def arrow(ax, p1, p2, color=INK3, style="-|>", lw=1.6, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, color=color, lw=lw,
                          mutation_scale=14, connectionstyle=connectionstyle)
    ax.add_patch(a)


def render_architecture():
    fig, ax = plt.subplots(figsize=(13, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6.5, 7.65, "MM-Future — Multi-Mode Joint World-Action Modeling",
             ha="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(6.5, 7.28, "reconstruction of arXiv:2609.20377  ·  bidirectional conditional-flow-matching over paired action + scene streams",
             ha="center", fontsize=9.5, color=INK2)

    # --- Inputs ---
    box(ax, (0.3, 6.0), 2.3, 0.8, "History cameras\n(4 views, 2s @ 2Hz)", INK2, fontsize=8.5)
    box(ax, (0.3, 4.5), 2.3, 0.8, "Ground-truth future\ncameras (training only)", INK3, fontsize=8.5)

    # --- Tokenizer ---
    box(ax, (3.0, 5.5), 2.5, 1.6, "MM-Tokenizer\nCNN backbone → register-token\ncross-attention → chunk compressor",
        BLUE, fontsize=8.7)
    arrow(ax, (2.6, 6.4), (3.0, 6.3))
    arrow(ax, (2.6, 4.9), (3.0, 5.7))

    box(ax, (5.9, 6.35), 2.1, 0.7, "History MM-Tokens\n(clean prefix)", BLUE, fontsize=8.3)
    box(ax, (5.9, 5.35), 2.1, 0.7, "Scene z₁ target\n(flow-matching target, stop-grad)", INK3, fontsize=7.8)
    arrow(ax, (5.5, 6.5), (5.9, 6.7))
    arrow(ax, (5.5, 5.9), (5.9, 5.7))

    # --- Action prior ---
    box(ax, (0.3, 3.1), 2.3, 0.9, "Action Prior\nK-means Gaussian-mixture\nover training trajectories",
        YELLOW, text_color=INK, fontsize=8.2)
    box(ax, (3.0, 3.1), 2.5, 0.9, "z₀ action noise\n(M proposals, one GMM\ncluster each → diverse intents)",
        YELLOW, text_color=INK, fontsize=8.2)
    arrow(ax, (2.6, 3.55), (3.0, 3.55))
    box(ax, (3.0, 1.9), 2.5, 0.9, "z₀ scene noise\n(M proposals, N(0, I))", INK3, fontsize=8.2)

    # --- Flow transformer ---
    box(ax, (6.3, 2.6), 3.1, 3.0,
        "Bidirectional Flow-Matching\nTransformer\n\nasymmetric attention mask:\nhistory → history only\naction+scene → history + both streams\n\nseparate FFN branch per modality,\nshared self-attention\n\nz_t = (1−t)z₀ + t·z₁ , t ~ U(0,1)",
        ORANGE, fontsize=8.3)
    arrow(ax, (7.0, 6.35), (7.0, 5.6), connectionstyle="arc3,rad=-0.3")
    arrow(ax, (7.0, 5.35), (7.0, 5.6), connectionstyle="arc3,rad=0.3")
    arrow(ax, (5.5, 3.55), (6.3, 4.0))
    arrow(ax, (5.5, 2.35), (6.3, 3.2))

    box(ax, (10.1, 3.85), 2.4, 0.8, "predicted velocity\nv_action  [M, T_f, 4]", ORANGE, fontsize=8)
    box(ax, (10.1, 2.75), 2.4, 0.8, "predicted velocity\nv_scene  [M, T_s, D]", ORANGE, fontsize=8)
    arrow(ax, (9.4, 4.3), (10.1, 4.25))
    arrow(ax, (9.4, 3.4), (10.1, 3.15))

    # --- one-step estimate / trajectory decode ---
    box(ax, (10.1, 1.55), 2.4, 0.9, "one-step z₁ estimate\n+ cumsum → M candidate\ntrajectories", BLUE, fontsize=7.8)
    arrow(ax, (11.3, 3.75), (11.3, 2.45))

    # --- best of many (training only) ---
    box(ax, (7.6, 0.25), 3.0, 0.85, "Best-of-many selection (training)\nm* = argmin dist(trajᵐ, GT traj)\nonly m* backprops CFM loss",
        INK3, fontsize=7.6)
    arrow(ax, (10.1, 1.75), (10.6, 1.1))

    # --- scorer ---
    box(ax, (10.1, 0.25), 2.6, 1.1,
        "Proposal Scorer\nper-proposal summary →\nself-attention across M →\nranked scores",
        AQUA, fontsize=8)
    arrow(ax, (11.5, 1.55), (11.5, 1.35))

    # --- aux BEV decoder ---
    box(ax, (6.3, 0.25), 1.0, 0.85, "Aux BEV\ndecoder*", INK3, fontsize=7, ls="--")
    arrow(ax, (10.1, 3.15), (7.3, 0.7), connectionstyle="arc3,rad=0.25", color=INK3, lw=1.2)

    ax.text(0.3, 0.15,
            "* not part of the paper — this project's own visualization-only head (scene tokens → coarse occupancy grid) used by simulate.py.",
            fontsize=7, color=INK3, style="italic")

    # legend
    legend_items = [
        (BLUE, "Visual encoding"), (YELLOW, "Action prior"), (ORANGE, "Flow-matching transformer"),
        (AQUA, "Proposal scorer"), (INK3, "Training-only / auxiliary"),
    ]
    for i, (c, label) in enumerate(legend_items):
        y = 7.6 - i * 0.0
    lx = 0.3
    ly = 0.9
    for i, (c, label) in enumerate(legend_items):
        ax.add_patch(mpatches.Rectangle((lx, ly - i * 0.28 + 1.0), 0.18, 0.14, color=c))
        ax.text(lx + 0.26, ly - i * 0.28 + 1.07, label, fontsize=7.3, color=INK2, va="center")

    plt.tight_layout()
    plt.savefig("assets/architecture_diagram.png", dpi=170, facecolor=SURFACE)
    plt.close()
    print("Saved assets/architecture_diagram.png")


def render_results():
    # Real numbers reported in the paper (arXiv:2609.20377), recovered via a
    # third-party mirror after arXiv's own /abs, /pdf and /html endpoints
    # rate-limited every direct fetch attempt this session (see README
    # "Sourcing note"). Nothing here is fabricated or estimated.
    tiles = [
        ("94.0", "NAVSIM-v1 PDMS", "navtest, trainval\n+3.3 vs. DriveFuture", BLUE),
        ("91.5", "NAVSIM-v2 EPDMS", "navtest\n+1.4 vs. UniTeD", ORANGE),
        ("32.3", "HUGSIM HD-Score", "zero-shot closed-loop\nvs. 28.9 (Latent-WAM)", AQUA),
        ("233 ms", "End-to-end latency", "NVIDIA H800 GPU\n64-proposal config", YELLOW),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), facecolor=SURFACE)
    fig.suptitle("MM-Future — reported results (arXiv:2609.20377)", fontsize=13.5,
                 fontweight="bold", color=INK, y=1.03)
    for ax, (value, label, sub, color) in zip(axes, tiles):
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        rect = mpatches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94, transform=ax.transAxes,
                                         boxstyle="round,pad=0.02,rounding_size=0.05",
                                         linewidth=2.2, edgecolor=color, facecolor="white")
        ax.add_patch(rect)
        ax.text(0.5, 0.62, value, transform=ax.transAxes, ha="center", va="center",
                 fontsize=26, fontweight="bold", color=color)
        ax.text(0.5, 0.38, label, transform=ax.transAxes, ha="center", va="center",
                 fontsize=10.5, fontweight="medium", color=INK)
        ax.text(0.5, 0.18, sub, transform=ax.transAxes, ha="center", va="center",
                 fontsize=8, color=INK2)
    plt.tight_layout()
    plt.savefig("assets/results.png", dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close()
    print("Saved assets/results.png")


if __name__ == "__main__":
    import os
    os.makedirs("assets", exist_ok=True)
    render_architecture()
    render_results()
