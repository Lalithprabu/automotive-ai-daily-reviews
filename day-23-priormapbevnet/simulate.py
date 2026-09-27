"""Render a real GIF from the trained PriorMapBEVNet checkpoint, showing:
  - a BEV panel: prior-map points, ground-truth map/vehicles, and the
    model's own predicted map heatmap + detected boxes, for both the
    with-prior and without-prior ("old way") settings side by side;
  - three live-camera panels (channel-norm visualization of the synthetic
    per-camera features actually fed to the model);
  - a telemetry strip with running detection-F1 / map-IoU for both settings.

Usage:
    python simulate.py --config config.yaml --checkpoint checkpoint.pt --out priormapbevnet_simulation.gif
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from src.dataset import SyntheticSceneDataset, collate_scenes
from src.geometry import BEVGrid, default_camera_rig
from src.model import PriorMapBEVNet
from train import detection_prf1, extract_peaks, map_iou

PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
}


def render_frame(model, grid, scene_batch, history, frame_idx):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), facecolor=PALETTE["surface"])
    fig.suptitle(
        "PriorMapBEVNet — vision-built map priors vs. live-only camera BEV",
        fontsize=13,
        color=PALETTE["ink"],
        fontweight="bold",
    )

    with torch.no_grad():
        out_on = model(scene_batch["prior_points"], scene_batch["camera_feats"], prior_enabled=True)
        out_off = model(scene_batch["prior_points"], scene_batch["camera_feats"], prior_enabled=False)

    p_on, r_on, f1_on = detection_prf1(out_on["det_heatmap"], scene_batch["det_mask"])
    p_off, r_off, f1_off = detection_prf1(out_off["det_heatmap"], scene_batch["det_mask"])
    iou_on = map_iou(out_on["map_heatmap"], scene_batch["map_heatmap"])
    iou_off = map_iou(out_off["map_heatmap"], scene_batch["map_heatmap"])
    history["f1_on"].append(f1_on)
    history["f1_off"].append(f1_off)
    history["iou_on"].append(iou_on)
    history["iou_off"].append(iou_off)

    h, w = grid.grid_h, grid.grid_w

    def draw_bev(ax, title, det_heatmap, map_heatmap, prior_enabled):
        ax.set_facecolor(PALETTE["surface"])
        ax.set_title(title, fontsize=9.5, color=PALETTE["ink"])
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

        if prior_enabled:
            pts = scene_batch["prior_points"][0]
            rc = grid.world_to_cell(pts[:, :2]).numpy()
            ax.scatter(rc[:, 1], rc[:, 0], s=2, c=PALETTE["muted"], alpha=0.35, label="prior points")

        map_np = map_heatmap[0].numpy()
        lane_mask = map_np[0] > 0.5
        curb_mask = map_np[1] > 0.5
        lane_rc = np.argwhere(lane_mask)
        curb_rc = np.argwhere(curb_mask)
        if lane_rc.size:
            ax.scatter(lane_rc[:, 1], lane_rc[:, 0], s=6, c=PALETTE["blue"], label="pred lane")
        if curb_rc.size:
            ax.scatter(curb_rc[:, 1], curb_rc[:, 0], s=6, c=PALETTE["yellow"], label="pred curb")

        gt_map = scene_batch["map_heatmap"][0].numpy()
        gt_rc = np.argwhere(gt_map.sum(0) > 0.5)
        if gt_rc.size:
            ax.scatter(
                gt_rc[:, 1], gt_rc[:, 0], s=14, facecolors="none", edgecolors=PALETTE["ink2"],
                linewidths=0.5, label="GT map",
            )

        for x, y, vw, vl, heading in scene_batch["vehicles"][0].tolist():
            rc = grid.world_to_cell(torch.tensor([x, y])).tolist()
            ax.add_patch(
                plt.Circle((rc[1], rc[0]), 1.4, fill=False, edgecolor=PALETTE["aqua"], linewidth=1.6)
            )

        peaks = extract_peaks(det_heatmap[0])
        for r, c, score in peaks:
            ax.add_patch(plt.Circle((c, r), 1.0, fill=False, edgecolor=PALETTE["orange"], linewidth=1.8))
            ax.text(c + 1.2, r, f"{score:.2f}", fontsize=6, color=PALETTE["orange"])

    draw_bev(axes[0, 0], "With prior map (fused)", out_on["det_heatmap"], out_on["map_heatmap"], True)
    draw_bev(axes[1, 0], "Without prior (live-only, \"old way\")", out_off["det_heatmap"], out_off["map_heatmap"], False)

    cam_names = ["front", "left", "right"]
    for col, cam_name in enumerate(cam_names):
        feat = scene_batch["camera_feats"][0][col]
        norm_img = feat.norm(dim=0).numpy()
        ax = axes[0, col + 1]
        ax.set_facecolor(PALETTE["surface"])
        ax.imshow(norm_img, cmap="inferno")
        ax.set_title(f"live cam: {cam_name}", fontsize=9, color=PALETTE["ink"])
        ax.set_xticks([])
        ax.set_yticks([])
        axes[1, col + 1].axis("off")

    axes[1, 1].axis("off")
    ax_tel = fig.add_axes([0.53, 0.06, 0.44, 0.36])
    ax_tel.set_facecolor(PALETTE["surface"])
    xs = list(range(len(history["f1_on"])))
    ax_tel.plot(xs, history["f1_on"], color=PALETTE["blue"], label="det F1 (with prior)")
    ax_tel.plot(xs, history["f1_off"], color=PALETTE["blue"], linestyle="--", label="det F1 (no prior)")
    ax_tel.plot(xs, history["iou_on"], color=PALETTE["aqua"], label="map IoU (with prior)")
    ax_tel.plot(xs, history["iou_off"], color=PALETTE["aqua"], linestyle="--", label="map IoU (no prior)")
    ax_tel.set_ylim(0, 1)
    ax_tel.set_title("running telemetry", fontsize=9, color=PALETTE["ink"])
    ax_tel.legend(fontsize=6, loc="upper right", framealpha=0.9)
    ax_tel.tick_params(labelsize=7)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf).convert("RGB")
    plt.close(fig)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--out", default="priormapbevnet_simulation.gif")
    parser.add_argument("--frames", type=int, default=14)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    grid = BEVGrid(**cfg["grid"])
    model = PriorMapBEVNet(default_camera_rig(), grid=grid, channels=cfg["model"]["channels"])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = SyntheticSceneDataset(args.frames, grid, seed=999, occlusion=cfg["data"]["occlusion"])
    history = {"f1_on": [], "f1_off": [], "iou_on": [], "iou_off": []}
    frames = []
    for i in range(args.frames):
        batch = collate_scenes([ds[i]])
        frames.append(render_frame(model, grid, batch, history, i))

    frames[0].save(
        args.out, save_all=True, append_images=frames[1:], duration=900, loop=0,
    )
    print(f"Saved simulation GIF to {args.out} ({len(frames)} frames)")
    print("Final telemetry:", {k: round(v[-1], 3) for k, v in history.items()})


if __name__ == "__main__":
    main()
