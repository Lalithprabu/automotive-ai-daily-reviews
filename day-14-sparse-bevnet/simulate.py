"""
Sparse-BEVNet real-time simulation dashboard.

Renders a 2x2 animated dashboard from a real trained checkpoint, using the
post-hoc calibrated decision threshold saved in `threshold.json`:

  Panel 1 (top-left):     Camera 0's grayscale feed with a heatmap glow
                           marking BRA (Bi-Level Routing Attention) routed
                           regions -- where the backbone is choosing to
                           look.
  Panel 2 (top-right):    Top-down BEV grid. Ground-truth boxes are cyan
                           outlines; predicted cells (at the CALIBRATED
                           threshold) are filled green (true positive /
                           hit), red (false negative / miss), or orange
                           (false positive).
  Panel 3 (bottom-left):  Live bar chart of the 6 per-camera
                           attention-gate weights from
                           SparseSpatialCrossAttention.
  Panel 4 (bottom-right): Scrolling recall-over-time line graph.

Usage:
    python simulate.py --config config.yaml --frames 30
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from src.dataset import generate_moving_sequence
from src.model import SparseBEVNet
from train import load_config

# --- palette (matches the rest of the series) ---
COLOR_BLUE = "#2a78d6"
COLOR_ORANGE = "#eb6834"
COLOR_AQUA = "#1baf7a"
COLOR_YELLOW = "#eda100"
COLOR_SURFACE = "#fcfcfb"
COLOR_INK = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_INK_MUTED = "#898781"
COLOR_CYAN = "#17b8c4"
COLOR_RED = "#d6432a"
COLOR_GREEN = "#1baf7a"


def load_model_and_threshold(config: dict):
    ckpt = torch.load(config["checkpoint_path"], map_location="cpu")
    model = SparseBEVNet(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(config["threshold_path"]) as f:
        thr = json.load(f)
    return model, thr["threshold"], thr


def run_inference(model, images: torch.Tensor):
    """images: (num_cameras, 3, H, W) -> single-frame forward pass."""
    with torch.no_grad():
        out = model(images.unsqueeze(0))
        probs = torch.sigmoid(out["obj_logits"])[0].numpy()  # (bev, bev)
        cam_gate = out["cam_gate"][0].numpy()  # (num_cameras,)
    return probs, cam_gate


def bra_routing_heatmap(model, images: torch.Tensor, camera_index: int, image_size: int):
    """Compute the BRA routing-glow heatmap for one camera view, upsampled
    to image_size x image_size for overlay."""
    with torch.no_grad():
        cam_img = images[camera_index : camera_index + 1]  # (1, 3, H, W)
        feat = model.backbone.stem(cam_img)  # (1, C, fh, fw)
        energy = model.backbone.bra.region_energy(feat)[0].numpy()  # (rg, rg)
    rg = energy.shape[0]
    reps = image_size // rg
    heat = np.kron(energy, np.ones((reps, reps)))
    if heat.shape[0] != image_size:
        heat = np.pad(heat, ((0, image_size - heat.shape[0]), (0, image_size - heat.shape[1])), mode="edge")
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def classify_cells(probs: np.ndarray, obj_grid: np.ndarray, threshold: float):
    pred = probs >= threshold
    gt = obj_grid > 0.5
    hit = pred & gt
    miss = (~pred) & gt
    false_pos = pred & (~gt)
    return hit, miss, false_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", type=str, default="assets/sparse_bevnet_simulation.gif")
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    model, threshold, thr_meta = load_model_and_threshold(config)
    print(f"Loaded checkpoint + calibrated threshold {threshold:.2f} "
          f"(val precision {thr_meta['precision']:.3f}, recall {thr_meta['recall']:.3f}, f1 {thr_meta['f1']:.3f})")

    frames_data = generate_moving_sequence(args.frames, config, seed=12345)

    # Precompute everything up front (inference + heatmaps + recall) so the
    # animation callback is a pure function of the frame index -- FuncAnimation
    # can call a given frame index more than once while saving, so accumulating
    # state (like a running recall list) *inside* the callback is unsafe.
    precomputed = []
    recall_running = []
    for t, frame in enumerate(frames_data):
        images = frame["images"]
        obj_grid = frame["obj_grid"].numpy()
        probs, cam_gate = run_inference(model, images)
        hit, miss, false_pos = classify_cells(probs, obj_grid, threshold)
        n_gt = int((obj_grid > 0.5).sum())
        n_hit = int(hit.sum())
        recall = n_hit / n_gt if n_gt > 0 else 1.0
        recall_running.append(recall)
        heat = bra_routing_heatmap(model, images, args.camera_index, config["image_size"])
        precomputed.append(
            dict(
                images=images,
                obj_grid=obj_grid,
                hit=hit,
                miss=miss,
                false_pos=false_pos,
                cam_gate=cam_gate,
                recall=recall,
                recall_history=list(recall_running),
                heat=heat,
            )
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), facecolor=COLOR_SURFACE)
    ax_cam, ax_bev, ax_bar, ax_recall = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for ax in axes.flat:
        ax.set_facecolor(COLOR_SURFACE)

    bev_size = config["bev_size"]
    bev_range = config["bev_range"]
    image_size = config["image_size"]
    num_cameras = config["num_cameras"]

    def render_frame(t: int):
        for ax in axes.flat:
            ax.clear()
            ax.set_facecolor(COLOR_SURFACE)

        pc = precomputed[t]
        images = pc["images"]
        obj_grid = pc["obj_grid"]
        hit, miss, false_pos = pc["hit"], pc["miss"], pc["false_pos"]
        cam_gate = pc["cam_gate"]
        recall = pc["recall"]

        # --- Panel 1: camera feed + BRA routing glow ---
        cam_img = images[args.camera_index].numpy()
        gray = cam_img.mean(axis=0)
        heat = pc["heat"]
        ax_cam.imshow(gray, cmap="gray", vmin=0, vmax=1, extent=(0, image_size, image_size, 0))
        ax_cam.imshow(heat, cmap="inferno", alpha=0.45, extent=(0, image_size, image_size, 0))
        ax_cam.set_title(f"Camera {args.camera_index} + BRA routing glow", color=COLOR_INK, fontsize=11, weight="bold")
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])

        # --- Panel 2: BEV grid ---
        cell = 2 * bev_range / bev_size
        for gy in range(bev_size):
            for gx in range(bev_size):
                x0 = -bev_range + gx * cell
                y0 = -bev_range + gy * cell
                facecolor = None
                if hit[gy, gx]:
                    facecolor = COLOR_GREEN
                elif miss[gy, gx]:
                    facecolor = COLOR_RED
                elif false_pos[gy, gx]:
                    facecolor = COLOR_ORANGE
                if facecolor is not None:
                    ax_bev.add_patch(
                        mpatches.Rectangle((x0, y0), cell, cell, facecolor=facecolor, edgecolor="none", alpha=0.85)
                    )
                if obj_grid[gy, gx] > 0.5:
                    ax_bev.add_patch(
                        mpatches.Rectangle((x0, y0), cell, cell, facecolor="none", edgecolor=COLOR_CYAN, linewidth=1.6)
                    )
        ax_bev.set_xlim(-bev_range, bev_range)
        ax_bev.set_ylim(-bev_range, bev_range)
        ax_bev.set_aspect("equal")
        ax_bev.set_title("BEV grid: GT (cyan) vs. predictions", color=COLOR_INK, fontsize=11, weight="bold")
        ax_bev.tick_params(colors=COLOR_INK_MUTED, labelsize=8)
        legend_handles = [
            mpatches.Patch(facecolor=COLOR_GREEN, label="hit"),
            mpatches.Patch(facecolor=COLOR_RED, label="miss"),
            mpatches.Patch(facecolor=COLOR_ORANGE, label="false positive"),
            mpatches.Patch(facecolor="none", edgecolor=COLOR_CYAN, label="ground truth"),
        ]
        ax_bev.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

        # --- Panel 3: per-camera attention gate weights ---
        cams = [f"C{c}" for c in range(num_cameras)]
        colors = [COLOR_BLUE, COLOR_ORANGE, COLOR_AQUA, COLOR_YELLOW, COLOR_BLUE, COLOR_ORANGE]
        ax_bar.bar(cams, cam_gate, color=colors[:num_cameras])
        ax_bar.set_ylim(0, max(0.5, float(cam_gate.max()) * 1.3))
        ax_bar.set_title("Live per-camera attention-gate weights", color=COLOR_INK, fontsize=11, weight="bold")
        ax_bar.tick_params(colors=COLOR_INK_SECONDARY, labelsize=9)

        # --- Panel 4: scrolling recall over time ---
        window = pc["recall_history"][-30:]
        xs = list(range(max(0, t - len(window) + 1), t + 1))
        ax_recall.plot(xs, window, color=COLOR_BLUE, linewidth=2)
        ax_recall.scatter([t], [recall], color=COLOR_BLUE, s=25, zorder=5)
        ax_recall.axhline(thr_meta["recall"], color=COLOR_INK_MUTED, linestyle="--", linewidth=1,
                           label=f"val recall @ calibrated thr ({thr_meta['recall']:.2f})")
        ax_recall.set_ylim(-0.05, 1.05)
        ax_recall.set_title("Recall over time (this scene)", color=COLOR_INK, fontsize=11, weight="bold")
        ax_recall.set_xlabel("frame", color=COLOR_INK_SECONDARY, fontsize=9)
        ax_recall.tick_params(colors=COLOR_INK_SECONDARY, labelsize=8)
        ax_recall.legend(loc="lower left", fontsize=7)

        fig.suptitle(
            f"Sparse-BEVNet reconstruction -- frame {t+1}/{len(frames_data)} "
            f"(calibrated threshold {threshold:.2f})",
            color=COLOR_INK, fontsize=13, weight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return []

    anim = FuncAnimation(fig, render_frame, frames=len(frames_data), blit=False)
    writer = PillowWriter(fps=4)
    anim.save(args.out, writer=writer)
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out}")


if __name__ == "__main__":
    main()
