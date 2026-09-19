"""Render assets/architecture_diagram.png and assets/results.png.

Usage:
    python scripts/render_diagrams.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Palette (series-wide)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(HERE, "assets")


def _box(ax, xy, w, h, text, facecolor, textcolor=SURFACE, fontsize=9.5, weight="bold"):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0, facecolor=facecolor, zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
        fontsize=fontsize, color=textcolor, fontweight=weight, zorder=4,
    )
    return box


def _arrow(ax, p0, p1, color=INK_SECONDARY, style="-|>", lw=1.6, ls="-"):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=13, color=color,
        linewidth=lw, linestyle=ls, zorder=2,
    )
    ax.add_patch(arr)


def render_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14.5, 9.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    ax.text(7.5, 9.15, "DRiF: Data-Driven Risk Fields", ha="center", fontsize=17,
             fontweight="bold", color=INK)
    ax.text(7.5, 8.65, "one shared BEV feature F_t  →  three parallel heads, trained jointly",
             ha="center", fontsize=10.5, color=INK_MUTED)

    # --- Row 1: input -> shared encoder --------------------------------
    _box(ax, (0.5, 7.1), 2.4, 0.9, "Multi-sensor\nBEV rasterization", INK_SECONDARY, fontsize=9.5)
    _arrow(ax, (2.9, 7.55), (4.9, 7.55))

    _box(ax, (4.9, 6.75), 5.2, 1.6,
         "Shared BEV Encoder\nconv stem  →  downsample ×4\n→  self-attention bottleneck\n\noutput: F_t",
         BLUE, fontsize=10)

    encoder_bottom = (7.5, 6.75)

    # --- Row 2: three parallel heads ------------------------------------
    head_centers_x = [2.0, 7.5, 13.0]
    head_y0, head_h, head_w = 4.55, 1.25, 3.6
    head_defs = [
        ("StaticMapHead\n(map segmentation)", ORANGE),
        ("DynamicRiskHead\n(pairwise-ranked risk field)\n★ paper's core contribution", AQUA),
        ("PlanningHead\n(trajectory waypoints)", YELLOW),
    ]
    for cx, (label, color) in zip(head_centers_x, head_defs):
        _arrow(ax, encoder_bottom, (cx, head_y0 + head_h), color=INK_MUTED, lw=1.3)
        _box(ax, (cx - head_w / 2, head_y0), head_w, head_h, label, color, fontsize=9.3,
             textcolor=SURFACE if color != YELLOW else INK)

    # --- Row 3: per-branch losses ----------------------------------------
    loss_y0, loss_h, loss_w = 2.55, 1.05, 3.6
    loss_defs = [
        "L_map\nCE + Dice",
        "L_rank + L_tv\nhinge ranking +\nsmoothness reg.",
        "L_plan\nSmooth-L1",
    ]
    for cx, label in zip(head_centers_x, loss_defs):
        _arrow(ax, (cx, head_y0), (cx, loss_y0 + loss_h), color=INK_MUTED, lw=1.2)
        _box(ax, (cx - loss_w / 2, loss_y0), loss_w, loss_h, label, "#e9e6df",
             textcolor=INK, fontsize=8.8, weight="normal")

    # --- Training-only pairwise supervision, feeding the risk branch -----
    sup_cx = head_centers_x[1]
    _box(ax, (sup_cx - 2.6, 0.35), 5.2, 1.35,
         "Training-only pairwise supervision\nsampled point pairs (i, j)  →  priority tiers:\n"
         "overlap > corridor > occupancy-risk  →  y_ij ∈ {+1, 0, −1}",
         "#f4f2ec", textcolor=INK, fontsize=8.8, weight="normal")
    _arrow(ax, (sup_cx, 1.7), (sup_cx, loss_y0), color=AQUA, lw=2.0)

    # --- Combined loss, off to the side, collecting all three branches ---
    _box(ax, (0.5, 0.35), 3.0, 1.6,
         "L_total =\n3.0·L_rank + 0.4·L_map\n+ 0.3·L_plan + 0.08·L_tv", INK, fontsize=9.2)
    for cx in head_centers_x:
        _arrow(ax, (cx, loss_y0), (2.0, 1.95), color=INK_MUTED, lw=0.9, ls="--")

    fig.savefig(os.path.join(ASSETS, "architecture_diagram.png"), dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved assets/architecture_diagram.png")


def render_results():
    """Stat-tile row for the 3 REAL Bench2Drive headline numbers (verified,
    see README sourcing note) -- incompatible scales, so tiles not bars.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), facecolor=SURFACE)

    tiles = [
        {
            "title": "Driving Score",
            "old": "85.65", "new": "88.78", "delta": "+3.13",
            "color": BLUE,
        },
        {
            "title": "Success Rate",
            "old": "69.09%", "new": "75.91%", "delta": "+6.82 pts",
            "color": AQUA,
        },
        {
            "title": "Collisions / km",
            "old": "2.822", "new": "1.753", "delta": "−38%",
            "color": ORANGE,
        },
    ]

    for ax, tile in zip(axes, tiles):
        ax.set_facecolor(SURFACE)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        box = FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0.02,rounding_size=0.05",
                              linewidth=1.4, edgecolor=tile["color"], facecolor="white", zorder=1)
        ax.add_patch(box)

        ax.text(0.5, 0.82, tile["title"], ha="center", va="center", fontsize=11,
                 color=INK_SECONDARY, fontweight="bold")
        ax.text(0.5, 0.52, tile["new"], ha="center", va="center", fontsize=26,
                 color=tile["color"], fontweight="bold")
        ax.text(0.5, 0.30, f"vs TF++ baseline: {tile['old']}", ha="center", va="center",
                 fontsize=9, color=INK_MUTED)
        ax.text(0.5, 0.14, tile["delta"], ha="center", va="center", fontsize=11,
                 color=tile["color"], fontweight="bold")

    fig.suptitle("DRiF vs TF++ baseline -- Bench2Drive closed-loop evaluation (paper-reported)",
                  fontsize=10.5, color=INK, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "results.png"), dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved assets/results.png")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    render_architecture_diagram()
    render_results()
