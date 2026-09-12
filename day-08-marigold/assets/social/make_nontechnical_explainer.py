"""Plain-language, icon-driven explainer graphic for a non-technical LinkedIn
audience. No jargon (no 'latent', 'U-Net', 'VAE') -- plain sentences + simple
hand-drawn-style vector icons built from matplotlib primitives (no emoji
fonts, for reliable rendering everywhere).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Wedge
import numpy as np

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"

fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=200)  # square, LinkedIn-friendly
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 10.8)
ax.set_ylim(0, 10.8)
ax.axis("off")

# ---------- Header ----------
ax.text(5.4, 10.35, "How Can an AI Judge Distance", ha="center", fontsize=22, weight="bold", color=INK)
ax.text(5.4, 9.85, "From Just ONE Regular Photo?", ha="center", fontsize=22, weight="bold", color=BLUE)
ax.text(5.4, 9.35, "No lasers. No radar. Just a camera and an idea borrowed from AI art tools.",
        ha="center", fontsize=12.5, color=INK2, style="italic")


def card(cx, cy, w, h, color):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.08,rounding_size=0.22",
                 linewidth=2, edgecolor=INK, facecolor="white", zorder=2))
    ax.add_patch(FancyBboxPatch((cx - w/2, cy + h/2 - 0.22), w, 0.22,
                 boxstyle="round,pad=0.0,rounding_size=0.0",
                 linewidth=0, facecolor=color, zorder=3))


def step_label(cx, top_y, number, title, color):
    ax.add_patch(Circle((cx - 3.55, top_y), 0.26, facecolor=color, edgecolor=INK, linewidth=1.6, zorder=5))
    ax.text(cx - 3.55, top_y, str(number), ha="center", va="center", fontsize=13, weight="bold", color="white", zorder=6)
    ax.text(cx - 3.1, top_y, title, ha="left", va="center", fontsize=14.5, weight="bold", color=INK, zorder=6)


# ---------- Row Y positions ----------
rows_y = [7.55, 5.55, 3.55, 1.35]
card_h = 1.55
card_w = 8.6
cx = 5.4

# STEP 1 — Take a photo
card(cx, rows_y[0], card_w, card_h, BLUE)
step_label(cx, rows_y[0] + 0.95, 1, "Start with an ordinary photo", BLUE)
# camera icon
icx, icy = 2.35, rows_y[0] - 0.15
ax.add_patch(FancyBboxPatch((icx - 0.55, icy - 0.35), 1.1, 0.7, boxstyle="round,pad=0.03,rounding_size=0.08",
             facecolor=BLUE, edgecolor=INK, linewidth=1.5, zorder=4))
ax.add_patch(Rectangle((icx - 0.2, icy + 0.32), 0.4, 0.18, facecolor=BLUE, edgecolor=INK, linewidth=1.5, zorder=4))
ax.add_patch(Circle((icx, icy), 0.24, facecolor="white", edgecolor=INK, linewidth=1.5, zorder=5))
ax.add_patch(Circle((icx, icy), 0.11, facecolor=INK, zorder=6))
ax.text(6.4, rows_y[0] - 0.15, "Just a single camera image — the kind\nyour phone or a car's windshield camera takes every day.",
        ha="center", va="center", fontsize=11.3, color=INK2)

# STEP 2 — Starts from noise (like TV static) and cleans it up
card(cx, rows_y[1], card_w, card_h, ORANGE)
step_label(cx, rows_y[1] + 0.95, 2, "The AI starts from random \u201cTV static\u201d", ORANGE)
icx, icy = 2.35, rows_y[1] - 0.15
np.random.seed(3)
grid = np.random.rand(14, 14)
ax.imshow(grid, extent=(icx - 0.55, icx + 0.55, icy - 0.42, icy + 0.42), cmap="gray", zorder=4)
ax.add_patch(Rectangle((icx - 0.55, icy - 0.42), 1.1, 0.84, fill=False, edgecolor=INK, linewidth=1.5, zorder=5))
ax.text(6.4, rows_y[1] - 0.15,
        "Same trick as AI image generators: begin with pure\nrandom noise, then gradually refine it — a few quick passes\ninstead of drawing from scratch.",
        ha="center", va="center", fontsize=11.3, color=INK2)

# STEP 3 — borrows knowledge from a giant art AI
card(cx, rows_y[2], card_w, card_h, AQUA)
step_label(cx, rows_y[2] + 0.95, 3, "...guided by a giant AI that\nalready understands images", AQUA)
icx, icy = 2.35, rows_y[2] - 0.2
# simple "brain" icon: two overlapping circles + lines
ax.add_patch(Circle((icx - 0.18, icy), 0.4, facecolor=AQUA, edgecolor=INK, linewidth=1.5, zorder=4))
ax.add_patch(Circle((icx + 0.18, icy), 0.4, facecolor=AQUA, edgecolor=INK, linewidth=1.5, zorder=4, alpha=0.85))
for dx in np.linspace(-0.5, 0.5, 5):
    ax.plot([icx + dx, icx + dx * 0.6], [icy - 0.35, icy + 0.3], color="white", linewidth=1.2, zorder=5)
ax.text(6.4, rows_y[2] - 0.2,
        "It reuses a model already trained on billions of\neveryday images (the same kind that power AI art tools) —\nso it recognizes shapes, edges, and scenes it's never seen.",
        ha="center", va="center", fontsize=11.3, color=INK2)

# STEP 4 — Output: near/far map
card(cx, rows_y[3], card_w, card_h, YELLOW)
step_label(cx, rows_y[3] + 0.95, 4, "Out comes a \u201cnear vs. far\u201d map", "#c98400")
icx, icy = 2.35, rows_y[3] - 0.15
# simple layered "depth" icon: 3 overlapping mountain-ish shapes light->dark
shapes = [
    (Polygon([(icx-0.55,icy-0.4),(icx-0.1,icy+0.35),(icx+0.35,icy-0.4)], closed=True), "#f4d9a0"),
    (Polygon([(icx-0.35,icy-0.4),(icx+0.05,icy+0.1),(icx+0.5,icy-0.4)], closed=True), "#c98400"),
]
for poly, c in shapes:
    poly.set_facecolor(c); poly.set_edgecolor(INK); poly.set_linewidth(1.2); poly.set_zorder(4)
    ax.add_patch(poly)
ax.add_patch(Rectangle((icx - 0.55, icy - 0.42), 1.1, 0.84, fill=False, edgecolor=INK, linewidth=1.5, zorder=5))
ax.text(6.4, rows_y[3] - 0.15,
        "Brighter = closer, darker = farther — a distance map from\na single camera. Handy for cheaper self-driving perception,\nbut it shows relative distance, not exact meters (yet).",
        ha="center", va="center", fontsize=11.3, color=INK2)

# Connecting arrows between cards
for y1, y2 in zip(rows_y[:-1], rows_y[1:]):
    ax.add_patch(FancyArrowPatch((cx, y1 - card_h/2 - 0.03), (cx, y2 + card_h/2 + 0.03),
                 arrowstyle="-|>", mutation_scale=20, linewidth=2.2, color=INK3, zorder=1))

# Footer callout
ax.add_patch(FancyBboxPatch((0.5, 0.15), 9.8, 0.55, boxstyle="round,pad=0.05,rounding_size=0.12",
             facecolor=INK, edgecolor="none", zorder=5))
ax.text(5.4, 0.42, "Marigold  —  one photo in, a distance map out, powered by the same AI behind image generators.",
        ha="center", va="center", fontsize=11.5, color="white", weight="bold", zorder=6)

plt.tight_layout()
plt.savefig("marigold_explainer_nontechnical.png", facecolor=SURFACE, bbox_inches="tight")
print("saved marigold_explainer_nontechnical.png")
