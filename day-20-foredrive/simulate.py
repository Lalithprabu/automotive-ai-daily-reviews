"""
simulate.py -- real-time-style synthetic simulation using the REAL trained
ForeDrive checkpoint (from train.py). Renders a 3-panel matplotlib animation
and saves it as a GIF:

  Panel 1: current front-view raster INPUT feed (what the model actually
           sees), with other agents boxed.
  Panel 2: BEV panel -- ground-truth future ego trajectory (green) vs. the
           DiT planner's sampled predicted trajectory (blue), plus a bar
           strip visualizing the world model's per-horizon confidence
           ("reliability") gating weights.
  Panel 3: live telemetry strip -- running displacement error (predicted vs.
           ground truth waypoints), updating every frame.

Every number in this animation comes from the actual trained model's real
outputs on real (synthetic) scenes -- no fabricated numbers.

Usage:
    python3 simulate.py [--checkpoint checkpoint.pt] [--out simulation.gif] [--num-frames 24]
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
import yaml

from models.foredrive import ForeDriveModel
from src.data.synthetic_dataset import (
    SyntheticDrivingDataset, X_RANGE_M, Y_RANGE_M, VEHICLE_LENGTH_M, VEHICLE_WIDTH_M,
)


def load_model(checkpoint_path: str) -> ForeDriveModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ForeDriveModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--out", type=str, default="simulation.gif")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=3)
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint)

    # One held-out scene per animation frame, so each frame shows a genuinely
    # different scenario (this reads as a "live feed" of varied situations
    # rather than one scene looping) -- still using the SAME trained model.
    sim_ds = SyntheticDrivingDataset(
        num_scenes=args.num_frames, scene_len=cfg["data"]["scene_len"], dt=cfg["data"]["dt"],
        raster_size=cfg["data"]["raster_size"], horizons=cfg["data"]["horizons"],
        num_waypoints=cfg["data"]["num_waypoints"], context_t=cfg["data"]["context_len"] - 1,
        num_agents_min=cfg["data"]["num_agents_min"], num_agents_max=cfg["data"]["num_agents_max"],
        seed=999999,   # disjoint from train/val seeds used in train.py
    )

    # --- Run the real model on every frame's scene up front ------------------
    frames = []
    running_pred_err = []
    with torch.no_grad():
        for i in range(len(sim_ds)):
            item = sim_ds[i]
            current = item["current_frame"].unsqueeze(0)
            ego = item["ego_context"].unsqueeze(0)
            expert = item["expert_waypoints"]
            cv_baseline = item["cv_baseline_waypoints"]

            traj, confidence = model.sample_trajectory(current, ego)
            traj = traj.squeeze(0)                       # (num_waypoints, 2)
            confidence = confidence.squeeze(0)             # (num_horizons, N)
            horizon_confidence = confidence.mean(dim=-1)      # (num_horizons,) -- avg over tokens, for the bar strip

            err = (traj - expert).norm(dim=-1).mean().item()
            running_pred_err.append(err)

            scene = sim_ds.get_scene(i)
            t = sim_ds.context_t
            frames.append({
                "raster": item["current_frame"].numpy(),                 # (3, H, W)
                "expert_traj": expert.numpy(),                              # (T, 2) relative to ego@t
                "pred_traj": traj.numpy(),                                    # (T, 2)
                "cv_traj": cv_baseline.numpy(),                                 # (T, 2)
                "horizon_confidence": horizon_confidence.numpy(),                 # (H,)
                "num_agents": scene.num_agents,
                "agents_x_rel": (scene.agents_x[:, t] - scene.ego_x[t]),           # (num_agents,)
                "agents_y": scene.agents_y[:, t],                                    # (num_agents,)
                "ego_y": scene.ego_y[t],
            })

    print(f"Ran trained model on {len(frames)} held-out scenes. "
          f"Mean displacement error over these frames: {np.mean(running_pred_err):.3f} m")

    # --- Figure / axes --------------------------------------------------------
    fig, (ax_raster, ax_bev, ax_tel) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ForeDrive Reconstruction -- Real-Time Synthetic Simulation "
                 "(input / ground-truth / model-prediction overlay)", fontsize=11)

    raster_size = cfg["data"]["raster_size"]
    horizons = cfg["data"]["horizons"]

    def raster_to_rgb(raster: np.ndarray) -> np.ndarray:
        """Channels (lane, ego, agents) -> a simple RGB visualization."""
        rgb = np.zeros((raster.shape[1], raster.shape[2], 3), dtype=np.float32)
        rgb[..., 1] = raster[0] * 0.5           # lane markings -> dim green
        rgb[..., 1] += raster[1] * 0.9            # ego -> bright green
        rgb[..., 0] = raster[2]                     # other agents -> red
        return np.clip(rgb, 0.0, 1.0)

    def world_x_to_px(x_rel):
        return np.asarray(x_rel) / X_RANGE_M * raster_size

    def world_y_to_px(y):
        return (np.asarray(y) + Y_RANGE_M / 2.0) / Y_RANGE_M * raster_size

    def update(frame_idx: int):
        ax_raster.clear()
        ax_bev.clear()
        ax_tel.clear()

        f = frames[frame_idx]

        # --- Panel 1: raw input raster with agent boxes -----------------------
        ax_raster.imshow(raster_to_rgb(f["raster"]), origin="lower",
                          extent=[0, raster_size, 0, raster_size])
        for a in range(f["num_agents"]):
            ax_agent_x = world_x_to_px(f["agents_x_rel"][a] - VEHICLE_LENGTH_M / 2.0)
            ax_agent_y = world_y_to_px(f["agents_y"][a] - VEHICLE_WIDTH_M / 2.0)
            w = world_x_to_px(VEHICLE_LENGTH_M)
            h = world_y_to_px(VEHICLE_WIDTH_M) - world_y_to_px(0)
            if 0 <= f["agents_x_rel"][a] <= X_RANGE_M:
                ax_raster.add_patch(Rectangle((ax_agent_x, ax_agent_y), w, h,
                                               fill=False, edgecolor="yellow", linewidth=1.5))
        ax_raster.set_title(f"Panel 1: INPUT (front-view raster proxy)\nframe {frame_idx+1}/{len(frames)}")
        ax_raster.set_xticks([])
        ax_raster.set_yticks([])

        # --- Panel 2: BEV trajectories + confidence bars -----------------------
        expert_xy = f["expert_traj"]
        pred_xy = f["pred_traj"]
        cv_xy = f["cv_traj"]
        ax_bev.plot([0] + list(expert_xy[:, 0]), [0] + list(expert_xy[:, 1]),
                    "o-", color="green", label="Ground truth (expert)", linewidth=2)
        ax_bev.plot([0] + list(pred_xy[:, 0]), [0] + list(pred_xy[:, 1]),
                    "o-", color="blue", label="Model prediction (DiT sample)", linewidth=2)
        ax_bev.plot([0] + list(cv_xy[:, 0]), [0] + list(cv_xy[:, 1]),
                    "--", color="gray", label="Constant-velocity baseline", linewidth=1, alpha=0.7)
        ax_bev.scatter([0], [0], color="black", marker="s", s=40, zorder=5, label="Ego (t=0)")
        ax_bev.set_xlim(-1, X_RANGE_M)
        ax_bev.set_ylim(-6, 6)
        ax_bev.set_xlabel("forward (m)")
        ax_bev.set_ylabel("lateral (m)")
        ax_bev.set_title("Panel 2: BEV trajectory overlay")
        ax_bev.legend(loc="upper left", fontsize=7)
        ax_bev.grid(alpha=0.3)

        # Per-horizon confidence bars, inset at the bottom of the BEV panel.
        n_h = len(horizons)
        bar_x = np.linspace(0.55, 0.95, n_h)
        for hi, (bx, conf) in enumerate(zip(bar_x, f["horizon_confidence"])):
            ax_bev.add_patch(Rectangle((bx * X_RANGE_M, -5.7), 0.06 * X_RANGE_M, conf * 1.5,
                                        transform=ax_bev.transData, color="orange", alpha=0.8))
            ax_bev.text(bx * X_RANGE_M + 0.03 * X_RANGE_M, -5.9, f"h{horizons[hi]}",
                        ha="center", fontsize=6)
        ax_bev.text(0.55 * X_RANGE_M, -4.0, "future-horizon\nreliability", fontsize=6, color="darkorange")

        # --- Panel 3: running displacement error telemetry -----------------------
        xs = list(range(1, frame_idx + 2))
        ax_tel.plot(xs, running_pred_err[:frame_idx + 1], "-o", color="blue",
                    label="Model ADE (m)", markersize=3)
        cv_err_per_frame = [
            float(np.linalg.norm(frames[i]["cv_traj"] - frames[i]["expert_traj"], axis=-1).mean())
            for i in range(frame_idx + 1)
        ]
        ax_tel.plot(xs, cv_err_per_frame, "--", color="gray", label="CV baseline ADE (m)")
        ax_tel.set_xlim(1, len(frames))
        ax_tel.set_xlabel("frame")
        ax_tel.set_ylabel("mean displacement error (m)")
        ax_tel.set_title("Panel 3: live telemetry (real, measured per frame)")
        ax_tel.legend(loc="upper right", fontsize=7)
        ax_tel.grid(alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    writer = PillowWriter(fps=args.fps)
    anim.save(args.out, writer=writer)
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out}")


if __name__ == "__main__":
    main()
