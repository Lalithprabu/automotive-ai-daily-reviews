"""assets/architecture.png: glance-only scoring (old way) vs. glance -> gaze -> re-projected scoring (GGDriver-style)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent
INK, MUTED, BLUE, RED, CYAN = "#1f2937", "#6b7280", "#2563eb", "#dc2626", "#0891b2"


def box(ax, x, y, w, h, txt, col, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=col, lw=1.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, a, b, col=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=col, lw=1.4))


def main():
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(0.2, 5.9, "Old way: every candidate is refined and scored against the SAME compressed scene", fontsize=11,
            color=RED, weight="bold")
    box(ax, 0.2, 4.5, 1.7, 0.9, "camera\n96 x 160", MUTED)
    box(ax, 2.3, 4.5, 1.9, 0.9, "conv backbone\n240 patches @ /8", MUTED)
    box(ax, 4.6, 4.5, 2.1, 0.9, "8 registers + ego\n(9 shared tokens)", RED)
    box(ax, 7.1, 4.5, 2.6, 0.9, "45 anchors -> glance\nrefine on 9 tokens", RED)
    box(ax, 10.1, 4.5, 2.7, 0.9, "score on 9 tokens\n(same for all)", RED)
    for a, b in (((1.9, 4.95), (2.3, 4.95)), ((4.2, 4.95), (4.6, 4.95)), ((6.7, 4.95), (7.1, 4.95)),
                 ((9.7, 4.95), (10.1, 4.95))):
        arrow(ax, a, b)
    ax.text(0.2, 3.7, "GGDriver-style: glance at what is shared, gaze at what differs", fontsize=11, color=BLUE,
            weight="bold")
    box(ax, 0.2, 2.2, 1.7, 0.9, "camera\n+ calibration", MUTED)
    box(ax, 2.3, 2.2, 1.9, 0.9, "conv backbone\n240 patches @ /8", MUTED)
    box(ax, 4.6, 2.2, 2.1, 0.9, "GLANCE\nregisters -> top-8", BLUE)
    box(ax, 7.1, 2.2, 2.6, 0.9, "GAZE: project corridor\n-> patch mask (~17%)\n-> refine (dd, dv)", CYAN, 8.5)
    box(ax, 10.1, 2.2, 2.7, 0.9, "RE-PROJECT refined path\n-> new mask -> NC/DAC/EP\n-> rank, pick best", CYAN, 8.5)
    for a, b in (((1.9, 2.65), (2.3, 2.65)), ((4.2, 2.65), (4.6, 2.65)), ((6.7, 2.65), (7.1, 2.65)),
                 ((9.7, 2.65), (10.1, 2.65))):
        arrow(ax, a, b, BLUE)
    ax.plot([3.25, 3.25, 11.45], [2.2, 1.75, 1.75], color=CYAN, lw=1.4)
    arrow(ax, (8.4, 1.75), (8.4, 2.2), CYAN)
    arrow(ax, (11.45, 1.75), (11.45, 2.2), CYAN)
    ax.text(3.4, 1.45,"native-resolution patches, masked per candidate (no copies: mask on the K x N logits)",
            fontsize=8.5, color=CYAN)
    ax.text(0.2, 0.7, "Ablations in this repo: random mask with the same patch count | attention over all 240 patches | "
            "gaze for refinement only | gaze for scoring only", fontsize=9, color=MUTED)
    ax.text(0.2, 0.3, "Toy re-implementation from the abstract of arXiv 2611.27330 (synthetic scenes, ~1.0M params)",
            fontsize=8.5, color=MUTED, style="italic")
    out = ROOT / "assets" / "architecture.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
