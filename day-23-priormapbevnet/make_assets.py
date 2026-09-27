"""Renders the required static visual assets (architecture diagram + results
stat tiles) as PNGs, per this project's standing visual-asset requirement.
Deterministic matplotlib output -- no photorealistic image generator.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def box(ax, xy, w, h, text, color, fontsize=9.5, text_color="white"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, fontweight="bold", wrap=True)


def arrow(ax, p0, p1, color=INK2):
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, shrinkA=2, shrinkB=2),
    )


def architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title(
        "PriorMapBEVNet — vision-built map priors fused with live camera BEV\n"
        "(this project's own disclosed reconstruction of arXiv:2609.26325)",
        fontsize=12, color=INK, fontweight="bold", loc="left",
    )

    # Prior branch (top row)
    box(ax, (0.15, 5.55), 2.55, 1.0, "Static prior\npoint cloud\n(Pi3X + DINOv3)", MUTED, fontsize=8.5, text_color="white")
    box(ax, (3.05, 5.55), 2.55, 1.0, "Sparse Voxel\nPrior Encoder", BLUE, fontsize=9)
    arrow(ax, (2.7, 6.05), (3.05, 6.05))

    # Live branch (bottom row)
    box(ax, (0.15, 3.55), 2.55, 1.0, "Live multi-camera\nfeatures\n(front/left/right)", MUTED, fontsize=8.5, text_color="white")
    box(ax, (3.05, 3.55), 2.55, 1.0, "Camera BEV\nLifter\n(pinhole sampling)", ORANGE, fontsize=8.5)
    arrow(ax, (2.7, 4.05), (3.05, 4.05))

    # Fusion
    box(ax, (5.95, 4.55), 2.55, 1.1, "Prior-Map\nBEV Fusion\n(learned gate +\nresidual conv)", AQUA, fontsize=8.5)
    arrow(ax, (3.6, 6.05), (5.95, 5.35))
    arrow(ax, (3.6, 4.05), (5.95, 4.9))

    # Heads
    box(ax, (8.85, 5.35), 2.0, 1.0, "Detection Head\nheatmap + boxes", BLUE, fontsize=8.7)
    box(ax, (8.85, 3.3), 2.0, 1.0, "Map Head\nlane / curb\nheatmap", YELLOW, fontsize=8.7, text_color=INK)
    arrow(ax, (8.5, 5.1), (8.85, 5.75))
    arrow(ax, (8.5, 5.0), (8.85, 3.9))

    ax.text(5.5, 2.4,
            "Old way: live-camera-only BEV perception — depth/parallax ambiguity means\n"
            "occluded lane geometry is simply missing from a single traversal.\n"
            "New way: a persistent, vision-built scene memory supplies the geometry the\n"
            "live cameras occluded — without ever needing LiDAR to build or query it.",
            fontsize=9.5, color=INK2, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=MUTED, linewidth=0.8))

    legend_items = [
        (MUTED, "Input"), (BLUE, "Prior-map path"), (ORANGE, "Live-camera path"),
        (AQUA, "Fusion"), (YELLOW, "Output head"),
    ]
    for i, (c, label) in enumerate(legend_items):
        ax.add_patch(mpatches.Rectangle((0.3 + i * 2.05, 0.3), 0.3, 0.3, facecolor=c))
        ax.text(0.68 + i * 2.05, 0.45, label, fontsize=8.5, color=INK, va="center")

    fig.savefig("assets/architecture.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def results_tiles():
    with open("metrics.json") as f:
        m = json.load(f)

    tiles = [
        ("Map IoU — with prior", f"{m['with_prior']['map_iou']:.3f}", AQUA),
        ("Map IoU — no prior", f"{m['without_prior']['map_iou']:.3f}", MUTED),
        ("Det. F1 — with prior", f"{m['with_prior']['det_f1']:.3f}", BLUE),
        ("Det. F1 — no prior", f"{m['without_prior']['det_f1']:.3f}", ORANGE),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), facecolor=SURFACE)
    for ax, (label, value, color) in zip(axes, tiles):
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                                              facecolor="white", edgecolor=color, linewidth=2.5))
        ax.text(0.5, 0.58, value, ha="center", va="center", fontsize=26, color=INK, fontweight="bold")
        ax.text(0.5, 0.22, label, ha="center", va="center", fontsize=10, color=INK2, wrap=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(
        "This repo's own synthetic-data measurements (NOT the paper's Argoverse 2 CDS/mAP numbers)",
        fontsize=10, color=INK2,
    )
    fig.savefig("assets/results.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    architecture_diagram()
    results_tiles()
    print("Saved assets/architecture.png and assets/results.png")
