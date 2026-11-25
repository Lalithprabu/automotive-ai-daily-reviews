"""assets/architecture.png - block diagram of the toy RadarOSV-style network vs the imaging-only baseline."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
BG, FG, MUTED = "#16181c", "#e8e8e8", "#9aa0a6"
BLUE, ORANGE, GREEN, RED = "#4aa3ff", "#ffad33", "#5cd65c", "#ff6b6b"


def box(ax, x, y, w, h, text, col, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=BG, ec=col, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=FG, fontsize=fs)


def arrow(ax, a, b, col=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=col, lw=1.3))


fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.2)
ax.axis("off")
ax.text(0.2, 5.9, "RadarOSV-style dual-radar network (toy re-implementation of arXiv 2611.30078, from the abstract)",
        color=FG, fontsize=12, weight="bold")

box(ax, 0.2, 4.3, 2.2, 0.9, "Imaging radar\npolar (D16, R64, A64)\nDoppler wraps at +-2.5 m/s", BLUE, 8)
box(ax, 0.2, 2.6, 2.2, 0.9, "Single-chip radar\npolar (D32, R32, A16)\nunaliased +-24 m/s, coarse", ORANGE, 8)
box(ax, 2.8, 4.3, 1.6, 0.9, "Polar encoder\nconv over (R, A)", BLUE, 8)
box(ax, 2.8, 2.6, 1.6, 0.9, "Polar encoder\nconv over (R, A)", ORANGE, 8)
box(ax, 4.8, 4.3, 2.0, 0.9, "PCG\nfixed (r, theta) sampling\n+ 4 deformable offsets", BLUE, 8)
box(ax, 4.8, 2.6, 2.0, 0.9, "PCG\nfixed + deformable", ORANGE, 8)
box(ax, 4.8, 1.0, 2.0, 0.9, "Ego-velocity head\nDoppler-vs-azimuth\n-> v_ego (no odometry)", GREEN, 8)
box(ax, 7.3, 2.9, 1.7, 1.8, "Cartesian BEV\n64 x 64 x 0.4 m\nconcat +\nv_ego cos(theta)\n-> U-shaped\nconv trunk", FG, 8)
box(ax, 9.5, 4.5, 3.3, 0.8, "Semantics: free / static / car /\ntruck / cyclist / pedestrian", FG, 8)
box(ax, 9.5, 3.2, 3.3, 1.0, "Velocity readout\nwrapped phase -> v_w in [-2.5, 2.5)\ncandidates v_w + 5k, k = -4..4\nclassifier SELECTS k", RED, 8)
box(ax, 9.5, 1.9, 3.3, 0.8, "v_gr = v_rel + v_ego cos(theta)\nground-relative radial velocity", GREEN, 8)

for a, b, c in (((2.4, 4.75), (2.8, 4.75), BLUE), ((2.4, 3.05), (2.8, 3.05), ORANGE), ((4.4, 4.75), (4.8, 4.75), BLUE),
                ((4.4, 3.05), (4.8, 3.05), ORANGE), ((6.8, 4.75), (7.3, 4.2), BLUE), ((6.8, 3.05), (7.3, 3.4), ORANGE),
                ((3.6, 2.6), (4.8, 1.5), GREEN), ((6.8, 1.45), (7.6, 2.9), GREEN), ((9.0, 4.2), (9.5, 4.9), FG),
                ((9.0, 3.7), (9.5, 3.7), FG), ((11.15, 3.2), (11.15, 2.7), RED)):
    arrow(ax, a, b, c)

ax.text(0.2, 0.45, "Baseline (paper's comparison): imaging radar only, ego velocity GIVEN from outside. Same PCG, trunk "
        "and readout, but the k classifier has no unaliased Doppler to look at.", color=MUTED, fontsize=9)
ax.text(0.2, 0.15, "The abstract names PCG, candidate selection and radar-estimated ego motion; sizes, losses, 2D BEV "
        "instead of 3D voxels and the phase/candidate parameterisation are my construction.", color=MUTED, fontsize=8)
out = ROOT / "assets" / "architecture.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
print(out)
