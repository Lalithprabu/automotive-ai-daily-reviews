#!/usr/bin/env python3
"""
make_assets.py -- renders two static PNGs from REAL run data (checkpoints/metrics.json
and assets/simulation_summary.json, both produced by train.py / simulate.py):

  assets/architecture.png -- a block/arrow sketch of THIS PROJECT'S reconstruction
      pipeline (AnchorGenerator -> AnchorConditionedPredictor -> TrustRegionCEM).
      This is explicitly NOT the paper's real figure (no such figure was accessible
      to this project -- see SOURCING.md); it is a diagram of the code in this repo.

  assets/results.png -- a stat-tile summary image built from the real numbers
      measured in train.py / simulate.py. No fabricated numbers: if metrics.json or
      simulation_summary.json is missing, this script fails loudly instead of
      inventing values.

Run AFTER train.py and simulate.py.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
BLOCK_FACE = "#eef3fb"
BLOCK_EDGE = "#2e5fa3"
ACCENT_NEW = "#1f9d55"
ACCENT_OLD = "#b0446a"
ACCENT_NAIVE = "#9a9a9a"


def draw_block(ax, xy, w, h, text, subtext=None, face=BLOCK_FACE, edge=BLOCK_EDGE, fontsize=10.5):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.8, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2 + (0.06 if subtext else 0), text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=INK, zorder=3)
    if subtext:
        ax.text(x + w / 2, y + h / 2 - 0.14, subtext, ha="center", va="center",
                fontsize=8, color=MUTED, zorder=3)
    return (x, y, w, h)


def arrow(ax, p_from, p_to, color=INK, style="-|>", lw=1.8, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(p_from, p_to, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                         connectionstyle=connectionstyle, zorder=1)
    ax.add_patch(a)


def make_architecture_png(out_path: str, sim_summary: dict):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("INTERACT reconstruction -- pipeline of THIS repo (not the paper's own figure)",
                  fontsize=11, color=INK, loc="left")

    # Row 1: inputs
    b_ego = draw_block(ax, (0.3, 5.6), 2.4, 1.0, "Ego history", "[H,4]: lat, spd, dlat, dspd")
    b_other = draw_block(ax, (3.0, 5.6), 2.4, 1.0, "Other-agent history", "[H,4]")
    b_map = draw_block(ax, (5.7, 5.6), 2.6, 1.0, "Map / goal geometry", "gap-to-merge, lane width")

    # Row 2: AnchorGenerator
    b_anchor = draw_block(ax, (5.7, 4.1), 2.6, 1.0, "AnchorGenerator", "K=5 intent vectors (cond)",
                           face="#fdeee0", edge="#c8781f")
    arrow(ax, (6.0 + 0.9, 5.6), (6.0 + 0.9, 5.1))

    # Row 3: AnchorConditionedPredictor (core module)
    b_pred = draw_block(ax, (1.6, 2.5), 4.6, 1.15, "AnchorConditionedPredictor",
                         "GRU encoder + FiLM(cond) + cross-attn decoder  ->  [T,2] other-agent future",
                         face="#e8f6ec", edge=ACCENT_NEW, fontsize=11)
    arrow(ax, (1.5, 5.6), (2.8, 3.65), connectionstyle="arc3,rad=-0.15")
    arrow(ax, (4.2, 5.6), (3.9, 3.65), connectionstyle="arc3,rad=0.05")
    arrow(ax, (6.5, 4.1), (5.2, 3.4), connectionstyle="arc3,rad=0.15")
    ax.text(3.9, 3.9, "queried ONCE per anchor\n(new way) --or-- once per\ncandidate (naive old way B)",
            fontsize=7.3, color=MUTED, ha="center")

    # Row 4: TrustRegionCEM
    b_cem = draw_block(ax, (1.6, 0.5), 4.6, 1.15, "TrustRegionCEM",
                        "sample candidates -> cost(collision, comfort, progress, trust-region) -> CEM update",
                        face="#fdeef2", edge=ACCENT_OLD, fontsize=11)
    arrow(ax, (3.9, 2.5), (3.9, 1.65))
    ax.text(4.85, 2.05, "cached prediction reused\nacross whole CEM population", fontsize=7.3, color=MUTED)

    # Output
    b_out = draw_block(ax, (7.2, 0.5), 3.1, 1.15, "Best anchor + refined\nego trajectory",
                        "argmin CEM cost across K anchors", face="#f4f4f4", edge="#555555")
    arrow(ax, (6.2, 1.05), (7.2, 1.05))

    # small legend / caption using REAL measured numbers
    if sim_summary:
        cap = (f"Measured this run: New Way = {sim_summary['new_way_predictor_calls']} predictor calls "
               f"(1/anchor)  |  Old Way A = {sim_summary['old_way_a_predictor_calls']} call (non-reactive)  |  "
               f"Old Way B (naive, single anchor) = {sim_summary['old_way_b_naive_calls_single_anchor']} calls")
        ax.text(0.3, 0.05, cap, fontsize=8, color=MUTED, ha="left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)


def make_results_png(out_path: str, metrics: dict, sim_summary: dict):
    held_out = metrics["held_out_bucketed"]
    c_int, u_int = held_out["conditioned"]["interactive"], held_out["unconditional"]["interactive"]
    ade_reduction_interactive = 100.0 * (u_int["ade"] - c_int["ade"]) / u_int["ade"]

    tiles = [
        ("Held-out ADE reduction\n(interactive bucket)", f"{ade_reduction_interactive:+.1f}%", ACCENT_NEW),
        ("New-way predictor calls\nper planning cycle (K=5)", f"{sim_summary['new_way_predictor_calls']}", ACCENT_NEW),
        ("Naive re-query calls\n(single anchor, this run)", f"{sim_summary['old_way_b_naive_calls_single_anchor']}", ACCENT_NAIVE),
        ("Mean forecast ADE reduction\n(sim episode, new vs old-A)", f"{sim_summary['mean_ade_reduction_pct']:.1f}%", ACCENT_NEW),
        ("Final train loss (MSE)\nconditioned model", f"{metrics['final_train_loss']['conditioned']:.5f}", "#2e5fa3"),
        ("Training wall time\n(CPU, this run)", f"{metrics['train_wall_time_s']:.1f}s", "#2e5fa3"),
    ]

    fig = plt.figure(figsize=(11, 7.2), facecolor=BG)
    fig.suptitle("Real, measured results -- this repo's synthetic ablation (not the paper's own numbers)",
                 fontsize=12, x=0.03, ha="left", color=INK, y=0.985)

    # --- stat tiles (2 rows x 3 cols) ---
    tile_axes = fig.subplots(2, 3, gridspec_kw={"top": 0.86, "bottom": 0.52, "left": 0.04, "right": 0.98,
                                                 "hspace": 0.5, "wspace": 0.3})
    for ax, (label, value, color) in zip(tile_axes.flat, tiles):
        ax.axis("off")
        box = FancyBboxPatch((0.03, 0.05), 0.94, 0.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                              linewidth=1.5, edgecolor=color, facecolor="#fafafa", transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=19, fontweight="bold",
                color=color, transform=ax.transAxes)
        ax.text(0.5, 0.24, label, ha="center", va="center", fontsize=8.3, color=MUTED,
                transform=ax.transAxes)

    # --- bar chart: ADE per anchor, new vs old way (bottom panel) ---
    ax_bar = fig.add_axes([0.08, 0.06, 0.86, 0.36])
    per_anchor = sim_summary["per_anchor_forecast_error"]
    names = list(per_anchor.keys())
    new_vals = [per_anchor[n]["new_way_ade"] for n in names]
    old_vals = [per_anchor[n]["old_way_ade"] for n in names]
    x = range(len(names))
    width = 0.35
    ax_bar.bar([xi - width / 2 for xi in x], new_vals, width=width, color=ACCENT_NEW, label="New Way (anchor-conditioned)")
    ax_bar.bar([xi + width / 2 for xi in x], old_vals, width=width, color=ACCENT_OLD, label="Old Way A (non-reactive)")
    ax_bar.set_xticks(list(x))
    ax_bar.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax_bar.set_ylabel("forecast ADE", fontsize=9)
    ax_bar.set_title("Forecast error by committed anchor (this run's simulated scenario)", fontsize=10)
    ax_bar.legend(fontsize=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)


def main():
    ckpt_dir = "checkpoints"
    assets_dir = "assets"
    os.makedirs(assets_dir, exist_ok=True)

    metrics_path = os.path.join(ckpt_dir, "metrics.json")
    sim_path = os.path.join(assets_dir, "simulation_summary.json")
    assert os.path.exists(metrics_path), f"missing {metrics_path} -- run train.py first"
    assert os.path.exists(sim_path), f"missing {sim_path} -- run simulate.py first"

    with open(metrics_path) as f:
        metrics = json.load(f)
    with open(sim_path) as f:
        sim_summary = json.load(f)

    arch_path = os.path.join(assets_dir, "architecture.png")
    results_path = os.path.join(assets_dir, "results.png")
    make_architecture_png(arch_path, sim_summary)
    make_results_png(results_path, metrics, sim_summary)
    print(f"Wrote {arch_path}")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
