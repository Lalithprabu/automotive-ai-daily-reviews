"""Renders the Marigold architecture diagram to assets/marigold_architecture.png
using the project's validated categorical palette (Reference Matrix §5)."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"

fig, ax = plt.subplots(figsize=(13, 7.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.axis("off")


def box(x, y, w, h, text, color, fontsize=10.5, text_color="white"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.4, edgecolor=INK, facecolor=color, alpha=0.95, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, weight="bold", zorder=4, wrap=True)


def arrow(x1, y1, x2, y2, color=INK2, style="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color=color, linestyle=style, zorder=2)
    ax.add_patch(a)


# Title
ax.text(6.5, 7.15, "Marigold — Latent-Diffusion Monocular Depth Estimation",
         ha="center", fontsize=15.5, weight="bold", color=INK)
ax.text(6.5, 6.78, "Repurposing a frozen Stable-Diffusion VAE + fine-tuned U-Net for affine-invariant depth",
         ha="center", fontsize=10, color=INK2, style="italic")

# Input image
box(0.4, 4.6, 2.0, 1.0, "RGB Image\n[B,3,H,W]", INK3, text_color="white")

# Frozen VAE encoder
box(3.0, 4.6, 2.1, 1.0, "Frozen VAE\nEncoder", BLUE)
arrow(2.4, 5.1, 3.0, 5.1)

# Image latent
box(5.6, 4.6, 2.0, 1.0, "Image Latent\n[B,4,H/8,W/8]", BLUE, fontsize=9.5)
arrow(5.1, 5.1, 5.6, 5.1)

# Noise -> depth latent
box(5.6, 2.3, 2.0, 1.0, "Noisy Depth\nLatent  x_t", ORANGE, fontsize=9.5)
box(0.4, 2.3, 2.0, 1.0, "Gaussian Noise\nx_T ~ N(0,I)", INK3)
arrow(1.4, 3.3, 1.4, 3.9)
arrow(2.4, 2.8, 5.6, 2.8)

# Concatenation node (core innovation)
box(8.1, 3.45, 1.7, 1.5, "Channel\nConcat\n[B,8,h,w]", YELLOW, text_color=INK, fontsize=10)
arrow(7.6, 5.1, 8.1, 4.5)
arrow(7.6, 2.8, 8.1, 3.9)

# U-Net
box(10.3, 3.2, 2.3, 2.0, "Fine-Tuned\nDenoising U-Net\n(ResBlocks +\nself-attn, t-embed)", AQUA, fontsize=9.5)
arrow(9.8, 4.2, 10.3, 4.2)

# Predicted noise / trailing-DDIM loop
box(8.1, 0.7, 2.3, 1.4, "Predicted\nNoise  ε̂\n(trailing-DDIM\nupdate)", ORANGE, fontsize=9.3)
arrow(10.4, 3.2, 9.6, 2.1)
arrow(8.1, 1.4, 5.9, 2.3, color=ORANGE, style="--")
ax.text(6.8, 1.75, "repeat 1–4 steps\n(trailing schedule)", ha="center", fontsize=8.3,
        color=ORANGE, style="italic")

# Decode
box(10.6, 0.7, 2.1, 1.2, "Frozen VAE\nDecoder", BLUE, fontsize=10)
arrow(11.15, 3.2, 11.65, 1.9)

box(10.3, -0.55, 2.5, 1.0, "Affine-Invariant\nDepth Map [B,1,H,W]", INK, fontsize=9.5)
ax.set_ylim(-1.0, 7.5)
arrow(11.65, 0.7, 11.55, 0.45)

# Ensembling note
ax.text(6.5, -0.75, "N independent noise seeds → per-pair least-squares affine alignment → averaged ensemble prediction",
        ha="center", fontsize=9, color=INK2)

# Legend
legend_items = [
    mpatches.Patch(color=BLUE, label="Frozen VAE (image ↔ latent)"),
    mpatches.Patch(color=AQUA, label="Fine-tuned denoising U-Net"),
    mpatches.Patch(color=ORANGE, label="Diffusion state (noise / depth latent)"),
    mpatches.Patch(color=YELLOW, label="Core innovation: latent concatenation"),
]
ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(0.0, 1.06),
          fontsize=8.6, frameon=False, ncol=1)

plt.tight_layout()
plt.savefig("marigold_architecture.png", facecolor=SURFACE, bbox_inches="tight")
print("saved marigold_architecture.png")
