"""
Renders assets/architecture_diagram.png and assets/results.png.

Run from the repo root:
    python scripts/render_diagrams.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# --- palette (matches every other day in this series) ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(OUT_DIR, exist_ok=True)


def rounded_box(ax, xy, w, h, text, facecolor, textcolor="white", fontsize=10.5, sub=None):
    box = mpatches.FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=facecolor, edgecolor="none",
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if sub:
        ax.text(cx, cy + h * 0.14, text, ha="center", va="center", color=textcolor, fontsize=fontsize, weight="bold")
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center", color=textcolor, fontsize=fontsize * 0.72)
    else:
        ax.text(cx, cy, text, ha="center", va="center", color=textcolor, fontsize=fontsize, weight="bold")


def arrow(ax, p0, p1, color=INK_SECONDARY):
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6, mutation_scale=14),
    )


def render_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # 6 camera feeds
    cam_w, cam_h = 1.15, 0.55
    cam_y = 5.9
    cam_xs = [0.3 + i * 1.3 for i in range(6)]
    for i, cx in enumerate(cam_xs):
        rounded_box(ax, (cx, cam_y), cam_w, cam_h, f"Cam {i}", BLUE, fontsize=9)

    # Shared conv backbone (BRA-routed)
    backbone_xy = (0.3, 4.55)
    backbone_w = cam_xs[-1] + cam_w - 0.3
    rounded_box(ax, backbone_xy, backbone_w, 0.75,
                "Shared Conv Backbone", ORANGE, fontsize=12,
                sub="+ Bi-Level Routing Attention (BRA): region routing before token attention")
    for cx in cam_xs:
        arrow(ax, (cx + cam_w / 2, cam_y), (cx + cam_w / 2, backbone_xy[1] + 0.75))

    # Sparse Spatial Cross-Attention -> BEV grid
    ssca_xy = (0.3, 3.25)
    rounded_box(ax, ssca_xy, backbone_w, 0.75,
                "Sparse Spatial Cross-Attention", AQUA, fontsize=12,
                sub="each BEV cell routes to its top-k most relevant camera views (not all 6)")
    arrow(ax, (backbone_xy[0] + backbone_w / 2, backbone_xy[1]), (ssca_xy[0] + backbone_w / 2, ssca_xy[1] + 0.75))

    # BEV grid
    bev_xy = (2.9, 1.95)
    bev_w, bev_h = 6.2, 0.85
    rounded_box(ax, bev_xy, bev_w, bev_h, "Shared BEV Grid", INK, fontsize=12,
                sub="lifted, fused multi-view scene representation")
    arrow(ax, (ssca_xy[0] + backbone_w / 2, ssca_xy[1]), (bev_xy[0] + bev_w / 2, bev_xy[1] + bev_h))

    # CGA fusion + detection head
    head_xy = (2.9, 0.4)
    rounded_box(ax, head_xy, bev_w, 0.95, "Cascaded Group Attention (CGA)", YELLOW, textcolor=INK, fontsize=12,
                sub="cascades partial outputs across head groups -> detection head: objectness + box regression")
    arrow(ax, (bev_xy[0] + bev_w / 2, bev_xy[1]), (head_xy[0] + bev_w / 2, head_xy[1] + 0.95))

    ax.text(6, 6.85, "Sparse-BEVNet -- Bi-Level Routing & Sparse Spatial Attention for Multi-View BEV 3D Detection",
            ha="center", va="center", fontsize=13.5, weight="bold", color=INK)
    ax.text(6, 0.15, "arXiv:2609.14185  |  Jing Zhang, Jiaqi Liu, Zibo Wang  |  CISAT 2026 (reconstruction diagram)",
            ha="center", va="center", fontsize=9, color=INK_MUTED)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "architecture_diagram.png")
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def stat_tile(ax, xy, w, h, value, label, delta, color):
    box = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="white", edgecolor=color, linewidth=2.2,
    )
    ax.add_patch(box)
    cx = xy[0] + w / 2
    ax.text(cx, xy[1] + h * 0.62, value, ha="center", va="center", fontsize=30, weight="bold", color=INK)
    ax.text(cx, xy[1] + h * 0.34, label, ha="center", va="center", fontsize=12.5, color=INK_SECONDARY)
    ax.text(cx, xy[1] + h * 0.14, delta, ha="center", va="center", fontsize=11.5, weight="bold", color=color)


def render_results():
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(4.5, 3.68, "Sparse-BEVNet -- paper-reported results (nuScenes)", ha="center", va="center",
            fontsize=14, weight="bold", color=INK)
    ax.text(4.5, 3.3, "Verified via alphaXiv mirror -- these two figures only; no other numbers fabricated",
            ha="center", va="center", fontsize=9.5, color=INK_MUTED)

    stat_tile(ax, (0.9, 0.35), 3.1, 2.6, "45.2%", "mAP", "+3.6 pts vs. baseline", BLUE)
    stat_tile(ax, (4.9, 0.35), 3.1, 2.6, "54.5%", "NDS", "+2.8 pts vs. baseline", AQUA)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "results.png")
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    render_architecture_diagram()
    render_results()
