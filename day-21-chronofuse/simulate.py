"""
Renders a real-time-style synthetic simulation GIF from the trained
ChronoFuse checkpoint: for a sequence of held-out driving scenes it
overlays (a) the ground-truth future object position/box, (b) the
'old way' stale/no-compensation detection (same weights, ChronoFuse
disabled), and (c) the ChronoFuse-compensated prediction — plus a live
telemetry sub-panel tracking each method's running mean center-error.

Usage:
    python simulate.py --config config.yaml --checkpoint checkpoints/chronofuse.pt
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from PIL import Image

from src.data.synthetic_events import SyntheticDrivingEventDataset, CLASS_NAMES
from src.models.model import ChronoFuseDetector
from src.utils.losses import STRIDE
from src.utils.metrics import decode_centers, center_distance_error, stale_baseline_error

# Palette per this project's standing visual-asset convention.
COLOR_GT = "#1baf7a"        # aqua -- ground truth
COLOR_STALE = "#eb6834"     # orange -- old way / no compensation
COLOR_CHRONO = "#2a78d6"    # blue -- ChronoFuse
COLOR_BG = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def draw_box(ax, cx, cy, w, h, color, label, linestyle="-"):
    rect = mpatches.Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        linewidth=2, edgecolor=color, facecolor="none", linestyle=linestyle, zorder=5,
    )
    ax.add_patch(rect)
    ax.text(cx - w / 2, cy - h / 2 - 3, label, color=color, fontsize=7, fontweight="bold", zorder=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/chronofuse.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ChronoFuseDetector(
        in_channels=cfg["model"]["in_channels"],
        backbone_channels=tuple(cfg["model"]["backbone_channels"]),
        fpn_channels=cfg["model"]["fpn_channels"],
        num_classes=cfg["model"]["num_classes"],
        cache_len=cfg["model"]["cache_len"],
        latency_dim=cfg["model"]["latency_dim"],
        max_latency=cfg["model"]["max_latency"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_frames = cfg["simulate"]["num_frames"]
    ds = SyntheticDrivingEventDataset(
        num_samples=num_frames,
        height=cfg["data"]["height"], width=cfg["data"]["width"], t_obs=cfg["data"]["t_obs"],
        max_agents=cfg["data"]["max_agents"], max_latency_steps=cfg["data"]["max_latency_steps"],
        seed=9999,  # disjoint seed from train/val splits
    )

    stale_running, chrono_running = [], []
    gif_frames = []

    for i in range(num_frames):
        sample = ds[i]
        event_seq = sample.event_seq.unsqueeze(0)
        latency_steps = sample.latency_steps.unsqueeze(0)

        with torch.no_grad():
            hm_on, size_on, off_on = model(event_seq, latency_steps, enable_chronofuse=True)
            hm_off, size_off, off_off = model(event_seq, latency_steps, enable_chronofuse=False)

        chrono_err = center_distance_error(hm_on[0], off_on[0], sample.future_centers, sample.classes)
        stale_err = stale_baseline_error(sample.stale_centers, sample.future_centers)
        chrono_running.append(chrono_err)
        stale_running.append(stale_err)

        num_per_class = [int((sample.classes == c).sum().item()) for c in range(cfg["model"]["num_classes"])]
        dets_chrono = decode_centers(hm_on[0], off_on[0], num_per_class)
        dets_stale_learned = decode_centers(hm_off[0], off_off[0], num_per_class)

        # --- Render this GIF frame -----------------------------------
        fig, (ax_scene, ax_tel) = plt.subplots(
            1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.3, 1]}, facecolor=COLOR_BG
        )

        # Background: sum of ON/OFF polarity channels of the last observed frame.
        last_frame = event_seq[0, -1].sum(dim=0).numpy()
        ax_scene.imshow(last_frame, cmap="Greys", vmin=0, vmax=1.2, origin="upper")
        ax_scene.set_facecolor(COLOR_BG)
        ax_scene.set_title(
            f"Scene {i+1}/{num_frames}  |  requested latency = {int(latency_steps.item())} frame(s) ahead",
            fontsize=10, color=INK_PRIMARY,
        )
        ax_scene.set_xticks([]); ax_scene.set_yticks([])

        # Ground-truth future boxes (aqua).
        for j in range(sample.future_centers.shape[0]):
            cx, cy = sample.future_centers[j].tolist()
            w, h = sample.sizes[j].tolist()
            cls = CLASS_NAMES[sample.classes[j].item()]
            draw_box(ax_scene, cx, cy, w, h, COLOR_GT, f"GT:{cls}")

        # Stale / old-way boxes (orange, dashed) -- the learned no-compensation head's own detections.
        for cls, x, y in dets_stale_learned:
            w, h = cfg["data"]["width"] * 0.10, cfg["data"]["height"] * 0.10
            draw_box(ax_scene, x, y, w, h, COLOR_STALE, f"OLD:{CLASS_NAMES[cls]}", linestyle="--")

        # ChronoFuse boxes (blue, solid).
        for cls, x, y in dets_chrono:
            w, h = cfg["data"]["width"] * 0.10, cfg["data"]["height"] * 0.10
            draw_box(ax_scene, x, y, w, h, COLOR_CHRONO, f"ChronoFuse:{CLASS_NAMES[cls]}")

        legend_handles = [
            mpatches.Patch(edgecolor=COLOR_GT, facecolor="none", label="Ground truth (future)"),
            mpatches.Patch(edgecolor=COLOR_STALE, facecolor="none", label="Old way (no compensation)"),
            mpatches.Patch(edgecolor=COLOR_CHRONO, facecolor="none", label="ChronoFuse (compensated)"),
        ]
        ax_scene.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

        # --- Telemetry panel: running mean center-error ------------------
        ax_tel.set_facecolor(COLOR_BG)
        frames_x = np.arange(1, len(stale_running) + 1)
        stale_cum = np.cumsum(stale_running) / frames_x
        chrono_cum = np.cumsum(chrono_running) / frames_x
        ax_tel.plot(frames_x, stale_cum, color=COLOR_STALE, linewidth=2, label="Old way (running mean)")
        ax_tel.plot(frames_x, chrono_cum, color=COLOR_CHRONO, linewidth=2, label="ChronoFuse (running mean)")
        ax_tel.set_xlim(1, num_frames)
        ax_tel.set_ylim(0, max(max(stale_cum), max(chrono_cum)) * 1.25 + 1e-3)
        ax_tel.set_xlabel("scene #", fontsize=8, color=INK_SECONDARY)
        ax_tel.set_ylabel("running mean center error (px)", fontsize=8, color=INK_SECONDARY)
        ax_tel.set_title("Live telemetry: latency-induced error, old vs. compensated", fontsize=9, color=INK_PRIMARY)
        ax_tel.legend(fontsize=7, loc="upper right")
        ax_tel.tick_params(colors=INK_MUTED, labelsize=7)
        for spine in ax_tel.spines.values():
            spine.set_color(INK_MUTED)

        recovered_pct = 100.0 * (stale_cum[-1] - chrono_cum[-1]) / max(stale_cum[-1], 1e-6)
        ax_tel.text(
            0.02, 0.02,
            f"so far: {recovered_pct:+.1f}% of old-way error recovered\n(this repo's synthetic metric -- see SOURCING.md)",
            transform=ax_tel.transAxes, fontsize=7, color=INK_SECONDARY, va="bottom",
        )

        fig.tight_layout()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        gif_frames.append(Image.fromarray(img))
        plt.close(fig)

    out_path = cfg["simulate"]["output_path"]
    duration_ms = int(1000 / cfg["simulate"]["fps"])
    gif_frames[0].save(
        out_path, save_all=True, append_images=gif_frames[1:], duration=duration_ms, loop=0
    )
    print(f"Rendered {len(gif_frames)} frames -> {out_path}")
    print(f"Final running mean error: old way = {np.mean(stale_running):.2f}px, "
          f"ChronoFuse = {np.mean(chrono_running):.2f}px")


if __name__ == "__main__":
    main()
