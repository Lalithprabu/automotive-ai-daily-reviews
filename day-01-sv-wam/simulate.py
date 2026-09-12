"""
Animated top-down simulation of a predicted SV-WAM trajectory: the ego
vehicle's footprint (a rotated rectangle) moves through the predicted
waypoints, leaving a trail behind it. Saved as an animated GIF -- GitHub
renders GIFs inline in READMEs, so this is meant to be embedded there
directly (no video codec / ffmpeg dependency needed).

Run with: python simulate.py --config tiny_cfg.yaml
"""
import argparse

import torch
import yaml
from matplotlib.patches import Polygon

from src.models.sv_wam import SVWAM
from src.utils.geometry import vehicle_corners_to_world


def build_vehicle_polygon(ax, color="#2a78d6"):
    polygon = Polygon([[0, 0]] * 4, closed=True, facecolor=color, edgecolor="#0b0b0b",
                       linewidth=1.2, zorder=5)
    ax.add_patch(polygon)
    return polygon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sv_wam_base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="trajectory_simulation.gif")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    device = torch.device("cpu")   # simulation is a one-off, CPU is plenty
    model = SVWAM(**cfg["model"], img_size=tuple(cfg["data"]["img_size"])).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    images = torch.randn(1, cfg["model"]["n_cams"], 3, *cfg["data"]["img_size"])
    with torch.no_grad():
        trajectory, _ = model(images, predict_action_only=True)   # [1, A, 3]
    trajectory = trajectory[0]   # [A, 3] -- (x, y, heading) per waypoint

    hw, hl = cfg["loss"]["vehicle_width"] / 2.0, cfg["loss"]["vehicle_length"] / 2.0
    local_corners = torch.tensor([[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]])
    all_corners = vehicle_corners_to_world(trajectory.unsqueeze(0), local_corners)[0]  # [A, 4, 2]

    xs, ys = trajectory[:, 0].tolist(), trajectory[:, 1].tolist()
    pad = max(cfg["loss"]["vehicle_length"], 2.0)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_xlabel("x (meters, ego frame)")
    ax.set_ylabel("y (meters, ego frame)")
    weights_note = "trained checkpoint" if args.checkpoint else "untrained/random weights -- expect mostly rotation, little travel"
    ax.set_title(f"SV-WAM predicted trajectory -- top-down simulation\n({weights_note})", fontsize=10.5)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)

    trail_line, = ax.plot([], [], "-o", color="#eb6834", linewidth=2, markersize=4, zorder=4)
    vehicle_polygon = build_vehicle_polygon(ax)
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)

    # Corner ordering from local_corners is (front-left, front-right, rear-left,
    # rear-right); reorder into a proper polygon winding (FL, FR, RR, RL).
    poly_order = [0, 1, 3, 2]

    def update(frame_idx):
        corners = all_corners[frame_idx][poly_order].numpy()
        vehicle_polygon.set_xy(corners)
        trail_line.set_data(xs[: frame_idx + 1], ys[: frame_idx + 1])
        step_text.set_text(f"waypoint {frame_idx + 1}/{len(xs)}")
        return vehicle_polygon, trail_line, step_text

    anim = FuncAnimation(fig, update, frames=len(xs), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out} ({len(xs)} frames)")


if __name__ == "__main__":
    main()
