"""Render assets/architecture_diagram.png for BeamTransFuser.

No results.png is rendered -- no paper-attributed metrics were recoverable
for this reconstruction (see README "Sourcing note").
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---- palette (shared across the whole daily series) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

MODALITY_COLORS = {
    "Camera": BLUE,
    "LiDAR": ORANGE,
    "Radar": AQUA,
    "GPS": YELLOW,
}

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "architecture_diagram.png")


def box(ax, x, y, w, h, text, fc, ec=INK, fontsize=10.5, textcolor="white", lw=1.4, zorder=3):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, zorder=zorder + 1, fontweight="bold",
             linespacing=1.35)
    return patch


def arrow(ax, x0, y0, x1, y1, color=INK_SECONDARY, lw=1.6, style="-|>", zorder=2, ls="-"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                         linewidth=lw, color=color, zorder=zorder, linestyle=ls)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13.5, 9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(6.75, 8.65, "BeamTransFuser", ha="center", va="center",
             fontsize=19, fontweight="bold", color=INK)
    ax.text(6.75, 8.25, "Hierarchical Transformer Fusion for Robust V2X Beam Prediction",
             ha="center", va="center", fontsize=11, color=INK_SECONDARY)

    # ---- Row 1: raw modality streams ----
    modalities = ["Camera", "LiDAR", "Radar", "GPS"]
    stream_y = 7.15
    stream_w, stream_h = 2.55, 0.62
    xs = [0.45, 3.4, 6.35, 9.3]
    for name, x in zip(modalities, xs):
        box(ax, x, stream_y, stream_w, stream_h, f"{name} stream", MODALITY_COLORS[name], fontsize=11)

    # ---- Row 2: per-modality encoders ----
    enc_y = 6.05
    enc_h = 0.62
    enc_labels = ["Camera\nConv Stem", "LiDAR\nConv Stem", "Radar\nConv Stem", "GPS MLP"]
    for name, x, label in zip(modalities, xs, enc_labels):
        box(ax, x, enc_y, stream_w, enc_h, label, "white", ec=MODALITY_COLORS[name],
            textcolor=INK, fontsize=9.5)
        arrow(ax, x + stream_w / 2, stream_y, x + stream_w / 2, enc_y + enc_h, color=MODALITY_COLORS[name])

    ax.text(11.95, 6.05 + enc_h / 2, "shared\nd_model\ntoken space", ha="left", va="center",
            fontsize=8.5, color=INK_MUTED, style="italic")

    # ---- Row 3: ModalityImputer ----
    imp_y = 5.05
    imp_x, imp_w, imp_h = 1.6, 8.4, 0.78
    box(ax, imp_x, imp_y, imp_w, imp_h,
        "ModalityImputer\ncross-attention: synthesizes a token for any FULLY-ABSENT modality,\nconditioned on the modalities that ARE present",
        INK, fontsize=9.8, textcolor="white")
    for name, x in zip(modalities, xs):
        arrow(ax, x + stream_w / 2, enc_y, x + stream_w / 2, imp_y + imp_h, color=MODALITY_COLORS[name])
    # dashed "branch in for dropped modality" annotation, kept outside the box column
    ax.annotate("sensor dropout ->\nimputer fills the gap", xy=(imp_x, imp_y + imp_h * 0.5),
                xytext=(0.15, imp_y + imp_h * 0.5), fontsize=8.3, color=INK_SECONDARY, style="italic",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=INK_MUTED, linestyle="--", lw=1.2,
                                 connectionstyle="arc3,rad=0.25"))

    # ---- Row 4: 4x CrossModalFusionBlock (stacked) ----
    fb_w, fb_h, fb_gap = 8.4, 0.55, 0.14
    fb_x = 1.6
    fb_top = imp_y - 0.35        # top edge of the whole 4-block stack
    stage_labels = ["cross-attn", "self-attn", "FFN"]
    block_ys = []
    for i in range(4):
        # i=0 drawn at the top (closest to imputer) down to i=3 at the bottom
        y = fb_top - fb_h - i * (fb_h + fb_gap)
        block_ys.append(y)
        fc = BLUE if i % 2 == 0 else "#3f8ade"
        label = f"CrossModalFusionBlock {i + 1}/4   ({'  ->  '.join(stage_labels)})"
        box(ax, fb_x, y, fb_w, fb_h, label, fc, fontsize=9)
    stack_bottom = block_ys[-1]
    stack_top = block_ys[0] + fb_h
    arrow(ax, imp_x + imp_w / 2, imp_y, fb_x + fb_w / 2, stack_top, color=INK_SECONDARY)

    # progressive-fusion side bracket (simple vertical line + arrowhead, no overlap)
    bracket_x = fb_x - 0.28
    ax.plot([bracket_x, bracket_x], [stack_bottom + 0.05, stack_top - 0.05],
            color=INK_MUTED, lw=1.2, zorder=2)
    arrow(ax, bracket_x, stack_top - 0.05, bracket_x, stack_bottom + 0.05, color=INK_MUTED, lw=1.2)
    ax.text(bracket_x - 0.16, (stack_top + stack_bottom) / 2, "progressive\nfusion",
            ha="center", va="center", fontsize=7.8, color=INK_MUTED, rotation=90)

    # ---- Row 5: pooled classification head ----
    head_y = 1.15
    head_x, head_w, head_h = 3.6, 4.4, 0.65
    box(ax, head_x, head_y, head_w, head_h, "Mean-pool -> LayerNorm -> MLP\nBeam Classification Head",
        ORANGE, fontsize=10)
    arrow(ax, fb_x + fb_w / 2, stack_bottom, head_x + head_w / 2, head_y + head_h, color=INK_SECONDARY)

    out_y = 0.15
    box(ax, 4.85, out_y, 1.9, 0.6, "Predicted\nBeam Index", "white", ec=ORANGE, textcolor=INK, fontsize=9.5)
    arrow(ax, head_x + head_w / 2, head_y, 4.85 + 0.95, out_y + 0.6, color=ORANGE)

    # legend
    legend_elems = [Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markersize=12, label=name)
                     for name, c in MODALITY_COLORS.items()]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              frameon=False, fontsize=9, ncol=1, labelcolor=INK_SECONDARY,
              bbox_transform=ax.transAxes)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
