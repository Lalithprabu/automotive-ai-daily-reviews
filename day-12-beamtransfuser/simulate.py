"""
Render a real-time simulation of BeamTransFuser tracking a vehicle drive-by
past an RSU, including a mid-sequence sensor-dropout event where the
ModalityImputer visibly kicks in.

Panels:
  1. Live camera-feed panel, with per-modality availability shown.
  2. BEV panel: ground-truth beam cone (dashed) vs. the model's top-1
     predicted beam cone (solid, green=hit / red=miss).
  3. Running top-3-accuracy telemetry sparkline.
  4. Modality-availability strip across time, highlighting the drop window.

Usage:
    python simulate.py --config config.yaml
Writes: assets/beam_prediction_simulation.gif
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
import yaml

from src.models.beam_transfuser import build_model_from_config
from src.utils.synthetic_data import generate_drive_sequence, MODALITIES, X_RANGE_M, LANE_OFFSET_M

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GREEN_HIT = "#1baf7a"
RED_MISS = "#d64545"

MODALITY_COLORS = {"camera": BLUE, "lidar": ORANGE, "radar": AQUA, "gps": YELLOW}


def load_model(cfg, ckpt_path):
    model = build_model_from_config(cfg)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        print(f"loaded checkpoint {ckpt_path}")
    else:
        print(f"WARNING: no checkpoint at {ckpt_path}, using randomly-initialized weights")
    model.eval()
    return model


def beam_to_angle_range(beam_idx, num_beams, fov=(0.0, math.pi)):
    lo, hi = fov
    width = (hi - lo) / num_beams
    a0 = lo + beam_idx * width
    a1 = a0 + width
    return a0, a1


def run_simulation(cfg, ckpt_path, out_path):
    scfg = cfg["simulate"]
    n_frames = scfg["n_frames"]
    drop_start = scfg["drop_start_frame"]
    drop_end = scfg["drop_end_frame"]
    drop_modality = scfg["drop_modality"]
    fps = scfg["fps"]
    num_beams = cfg["model"]["num_beams"]

    model = load_model(cfg, ckpt_path)
    frames = generate_drive_sequence(n_frames, num_beams=num_beams, seed=7)

    # run inference frame by frame, applying the scripted modality drop
    preds, top3_hits, presences = [], [], []
    running_hits = []
    for i, fr in enumerate(frames):
        camera = torch.from_numpy(fr["camera"]).unsqueeze(0)
        lidar = torch.from_numpy(fr["lidar"]).unsqueeze(0)
        radar = torch.from_numpy(fr["radar"]).unsqueeze(0)
        gps = torch.from_numpy(fr["gps"]).unsqueeze(0)

        presence = {m: torch.ones(1, dtype=torch.bool) for m in MODALITIES}
        dropped_now = drop_start <= i < drop_end
        if dropped_now:
            presence[drop_modality][0] = False
            if drop_modality == "camera":
                camera = torch.zeros_like(camera)
            elif drop_modality == "lidar":
                lidar = torch.zeros_like(lidar)
            elif drop_modality == "radar":
                radar = torch.zeros_like(radar)
            elif drop_modality == "gps":
                gps = torch.zeros_like(gps)

        with torch.no_grad():
            logits = model(camera, lidar, radar, gps, presence)
            probs = torch.softmax(logits, dim=-1)[0]
            top1 = int(torch.argmax(probs).item())
            top3 = set(torch.topk(probs, 3).indices.tolist())

        hit = fr["label"] in top3
        running_hits.append(hit)
        preds.append(top1)
        top3_hits.append(sum(running_hits) / len(running_hits))
        presences.append({m: (not (dropped_now and m == drop_modality)) for m in MODALITIES})

    # ---- figure layout ----
    fig = plt.figure(figsize=(11, 8.4), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.0, 0.55], width_ratios=[1, 1],
                           hspace=0.55, wspace=0.28, left=0.07, right=0.96, top=0.90, bottom=0.07)

    ax_cam = fig.add_subplot(gs[0, 0])
    ax_bev = fig.add_subplot(gs[0, 1])
    ax_spark = fig.add_subplot(gs[1, :])
    ax_strip = fig.add_subplot(gs[2, :])

    fig.suptitle("BeamTransFuser -- V2X Beam Prediction Drive-By Simulation",
                 fontsize=14, fontweight="bold", color=INK, y=0.965)

    def draw_frame(i):
        fr = frames[i]
        pres = presences[i]

        # --- camera panel ---
        ax_cam.clear()
        ax_cam.set_facecolor("black")
        cam_img = fr["camera"].copy()
        disp = np.transpose(cam_img, (1, 2, 0))
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
        if not pres["camera"]:
            disp[:] = 0.03
        ax_cam.imshow(disp)
        ax_cam.set_xticks([]); ax_cam.set_yticks([])
        status_txt = "LIVE" if pres["camera"] else "DROPPED -- imputer active"
        status_col = GREEN_HIT if pres["camera"] else RED_MISS
        ax_cam.set_title(f"Camera feed [{status_txt}]", fontsize=10.5, color=status_col, fontweight="bold")
        # modality chips
        chip_x = 0.02
        for m in MODALITIES:
            c = MODALITY_COLORS[m] if pres[m] else "#4a4a4a"
            ax_cam.add_patch(mpatches.FancyBboxPatch(
                (chip_x, 0.02), 0.17, 0.09, transform=ax_cam.transAxes,
                boxstyle="round,pad=0.01,rounding_size=0.02", facecolor=c, edgecolor="white", linewidth=0.8,
                clip_on=False))
            ax_cam.text(chip_x + 0.085, 0.065, m[:3].upper(), transform=ax_cam.transAxes,
                        ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
            chip_x += 0.19

        # --- BEV panel ---
        ax_bev.clear()
        ax_bev.set_facecolor(SURFACE)
        rsu_x, rsu_y = 0.0, 0.0
        ax_bev.scatter([rsu_x], [rsu_y], marker="^", s=180, color=INK, zorder=5, label="RSU")
        # lane
        ax_bev.plot([-X_RANGE_M, X_RANGE_M], [LANE_OFFSET_M, LANE_OFFSET_M],
                    color=INK_MUTED, lw=1, ls=":", zorder=1)
        # vehicle
        ax_bev.scatter([fr["x"]], [fr["y"]], marker="s", s=110,
                        color=BLUE if pres["gps"] else INK_MUTED, zorder=6, edgecolor="white", linewidth=1)

        radius = 62
        gt_a0, gt_a1 = beam_to_angle_range(fr["label"], num_beams)
        pred_a0, pred_a1 = beam_to_angle_range(preds[i], num_beams)
        hit = preds[i] in {fr["label"]} or top3_hits  # top1 vs gt for coloring cone
        pred_color = GREEN_HIT if preds[i] == fr["label"] else RED_MISS

        def cone(a0, a1, color, ls, lw, alpha, fillalpha, label):
            th = np.linspace(a0, a1, 24)
            xs = radius * np.cos(th)
            ys = radius * np.sin(th)
            poly_x = np.concatenate([[0], xs, [0]])
            poly_y = np.concatenate([[0], ys, [0]])
            ax_bev.fill(poly_x, poly_y, color=color, alpha=fillalpha, zorder=2)
            ax_bev.plot(poly_x, poly_y, color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                        zorder=3, label=label)

        cone(gt_a0, gt_a1, INK_SECONDARY, "--", 1.6, 0.9, 0.05, "Ground-truth beam")
        cone(pred_a0, pred_a1, pred_color, "-", 2.0, 0.95, 0.12,
             f"Predicted (top-1) {'HIT' if preds[i] == fr['label'] else 'MISS'}")

        ax_bev.set_xlim(-X_RANGE_M - 5, X_RANGE_M + 5)
        ax_bev.set_ylim(-5, radius + 5)
        ax_bev.set_aspect("equal")
        ax_bev.set_xticks([]); ax_bev.set_yticks([])
        ax_bev.set_title("BEV: ground-truth vs. predicted beam cone", fontsize=10.5, color=INK, fontweight="bold")
        ax_bev.legend(loc="upper right", fontsize=7.5, frameon=False)

        # --- sparkline panel ---
        ax_spark.clear()
        xs_sp = np.arange(i + 1)
        ax_spark.plot(xs_sp, top3_hits[: i + 1], color=BLUE, lw=2)
        ax_spark.fill_between(xs_sp, 0, top3_hits[: i + 1], color=BLUE, alpha=0.12)
        ax_spark.axvspan(drop_start, min(drop_end, n_frames) - 1, color=RED_MISS, alpha=0.08)
        ax_spark.set_xlim(0, n_frames - 1)
        ax_spark.set_ylim(0, 1.02)
        ax_spark.set_ylabel("running\ntop-3 acc", fontsize=8.5, color=INK_SECONDARY)
        ax_spark.set_title(f"Telemetry -- running top-3 accuracy: {top3_hits[i]:.3f}",
                            fontsize=10, color=INK, fontweight="bold", loc="left")
        ax_spark.tick_params(labelsize=7.5, colors=INK_SECONDARY)
        for spine in ["top", "right"]:
            ax_spark.spines[spine].set_visible(False)

        # --- modality-availability strip ---
        ax_strip.clear()
        ax_strip.set_xlim(0, n_frames - 1)
        ax_strip.set_ylim(0, len(MODALITIES))
        for row, m in enumerate(MODALITIES):
            for f in range(n_frames):
                p = not (drop_start <= f < drop_end and m == drop_modality)
                c = MODALITY_COLORS[m] if p else "#e3e0da"
                ax_strip.add_patch(mpatches.Rectangle((f, row), 1, 0.85, color=c, linewidth=0))
            ax_strip.text(-2, row + 0.42, m.upper(), ha="right", va="center", fontsize=7.5, color=INK_SECONDARY)
        ax_strip.axvline(i, color=INK, lw=1.4, zorder=5)
        ax_strip.set_xticks([]); ax_strip.set_yticks([])
        for spine in ax_strip.spines.values():
            spine.set_visible(False)
        ax_strip.set_title("Modality availability over time (grey = dropped, imputer active)",
                            fontsize=9, color=INK_SECONDARY, loc="left")

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 / fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"final running top-3 accuracy over sequence: {top3_hits[-1]:.4f}")
    print(f"drop window: frames [{drop_start}, {drop_end}) on modality '{drop_modality}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="beam_transfuser_checkpoint.pt")
    parser.add_argument("--out", type=str, default="assets/beam_prediction_simulation.gif")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_simulation(cfg, args.checkpoint, args.out)


if __name__ == "__main__":
    main()
