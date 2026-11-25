"""assets/architecture.png: the three pipelines compared (clean-trained, + input denoiser, noise-augmented retrain)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
BG, FG, MUTED = "#16181d", "#e8e8e8", "#9aa0a6"
RED, BLUE, PURPLE, GREEN, ORANGE = "#e05050", "#3d9be9", "#b77fe0", "#5cc85c", "#f0a030"


def box(ax, x, y, w, h, txt, c, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=BG, ec=c, lw=1.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", color=FG, fontsize=fs)


def arrow(ax, x0, y0, x1, y1, c=MUTED):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=c, lw=1.5))


def main():
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(0.2, 5.9, "Trajectory prediction under sensing + V2X noise: where the robustness comes from", color=FG,
            fontsize=13, weight="bold")
    ax.text(0.2, 5.6, "toy re-implementation of the setting in arXiv 2611.28128 (abstract only); same predictor in all rows",
            color=MUTED, fontsize=9)
    # shared input
    box(ax, 0.2, 2.3, 2.1, 1.5, "noisy object states\n(x, y, heading, speed)\n2 s @ 10 Hz, 8 agents\n"
        "sensor: AR(1) jitter\nV2X: GNSS bias + loss", ORANGE, 8.5)
    rows = [(4.25, "A  clean-trained (curated data)", RED, None),
            (2.85, "B  + input denoising module", PURPLE, "denoiser\n(Kalman filter or\nlearned Transformer)"),
            (1.5, "C  noise-augmented retrain", BLUE, None)]
    ax.plot([2.3, 2.7], [3.05, 3.05], color=MUTED, lw=1.5)
    ax.plot([2.7, 2.7], [1.9, 4.65], color=MUTED, lw=1.5)
    for y, lab, c, pre in rows:
        arrow(ax, 2.7, y + 0.4, 3.05 if pre else 5.35, y + 0.4)
        x = 3.1
        if pre:
            box(ax, x, y, 1.9, 0.8, pre, c, 8)
            arrow(ax, x + 1.95, y + 0.4, x + 2.25, y + 0.4)
        x = 5.4
        box(ax, x, y, 1.6, 0.8, "temporal\nTransformer\n(per agent, 2 L)", c, 8)
        arrow(ax, x + 1.65, y + 0.4, x + 1.95, y + 0.4)
        box(ax, x + 2.0, y, 1.6, 0.8, "social\nTransformer\n(8 agents, 2 L)", c, 8)
        arrow(ax, x + 3.65, y + 0.4, x + 3.95, y + 0.4)
        box(ax, x + 4.0, y, 1.6, 0.8, "6 mode queries\n-> 3 s offsets\n+ mode scores", c, 8)
        ax.text(3.1, y + 0.9, lab, color=c, fontsize=9.5, weight="bold")
        tr = {"A": "trained on CLEAN histories", "B": "predictor = A (frozen); denoiser trained noisy -> clean",
              "C": "trained on histories with noise level ~ U(0, 4)"}[lab[0]]
        ax.text(11.15, y + 0.4, tr, color=MUTED, fontsize=8, va="center", wrap=True)
    ax.text(0.2, 0.55, "anchor = last observed (x, y) of whatever history the predictor receives -> a clean-trained model "
            "trusts it blindly;\nthe augmented model learns how far to trust a stale or biased observation (it also "
            "sees fresh / staleness / V2X flags).", color=FG, fontsize=8.5)
    out = ROOT / "assets" / "architecture.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=BG, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
