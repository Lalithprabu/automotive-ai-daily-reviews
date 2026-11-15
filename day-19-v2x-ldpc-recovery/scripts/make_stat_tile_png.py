"""
Generates:
  assets/results_stat_tile.png            -- the required asset: a SINGLE stat
      tile showing ONLY the one verified paper number (87.27% @ 0.75dB). No
      other numbers, no comparison bars, per the project brief.
  assets/repo_reconstruction_stat_tile.png -- a SEPARATE, distinctly labeled
      tile for this repo's own synthetic-reconstruction recovery rate, kept
      in its own file specifically so it is never visually conflated with the
      paper-reported tile above as directly comparable data.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
YELLOW = "#eda100"
BG = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"


def make_tile(out_path, accent, big_text, label_lines, sub_lines, tag_text, tag_color, figsize=(6.2, 5.6)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150, facecolor=BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(BG)

    card = FancyBboxPatch((0.4, 0.4), 9.2, 9.2, boxstyle="round,pad=0.02,rounding_size=0.25",
                            linewidth=2.4, edgecolor=accent, facecolor="white", zorder=1)
    ax.add_patch(card)
    bar = FancyBboxPatch((0.4, 9.1), 9.2, 0.5, boxstyle="round,pad=0.0,rounding_size=0.12",
                           linewidth=0, facecolor=accent, zorder=2)
    ax.add_patch(bar)

    ax.text(5.0, 6.4, big_text, ha="center", va="center", fontsize=54, fontweight="bold",
            color=accent, zorder=3)
    y = 4.35
    for line in label_lines:
        ax.text(5.0, y, line, ha="center", va="center", fontsize=15, color=INK, fontweight="medium", zorder=3)
        y -= 0.68
    y -= 0.15
    for line in sub_lines:
        ax.text(5.0, y, line, ha="center", va="center", fontsize=10.5, color=MUTED, zorder=3)
        y -= 0.52

    tag_fontsize = 9.5 if len(tag_text) < 40 else 7.6
    tag = FancyBboxPatch((0.75, 0.85), 8.5, 0.75, boxstyle="round,pad=0.02,rounding_size=0.35",
                           linewidth=1.5, edgecolor=tag_color, facecolor="white", zorder=3)
    ax.add_patch(tag)
    ax.text(5.0, 1.225, tag_text, ha="center", va="center", fontsize=tag_fontsize, color=tag_color,
            fontweight="bold", zorder=4)

    plt.tight_layout(pad=0.8)
    plt.savefig(out_path, facecolor=BG, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


make_tile(
    "assets/results_stat_tile.png",
    GREEN,
    "87.27%",
    ["of failed messages recovered"],
    ["at Eb/N0 = 0.75 dB"],
    "PAPER-REPORTED (arXiv:2609.25609)",
    GREEN,
)

make_tile(
    "assets/repo_reconstruction_stat_tile.png",
    ORANGE,
    "12.5%",
    ["of first-attempt failures recovered", "by the prediction-aided 2nd pass"],
    ["at Eb/N0 = 0.75 dB -- 500 synthetic messages, 5 seeds", "this repo's own small GRU predictor + from-scratch LDPC (n=256)"],
    "THIS REPO'S SYNTHETIC RECONSTRUCTION -- NOT THE PAPER'S NUMBERS",
    ORANGE,
)
