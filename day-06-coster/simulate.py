"""
Animated top-down simulation of COSTER's time-reversed rollout, played back
in the natural forward-time direction: the model generates a trajectory
ordered [collision instant, ..., earliest state], which this script
reverses so playback shows a vehicle approaching, ending in the collision
at the final frame -- matching how a human would actually watch the
scenario unfold. Saved as an animated GIF -- GitHub renders GIFs inline in
READMEs.

Run with: python simulate.py --config tiny_coster_cfg.yaml
"""
import argparse

import torch
import yaml
from matplotlib.patches import Polygon

from src.models.coster import COSTERConfig, COSTERModel

BLUE = "#2a78d6"
ORANGE = "#eb6834"
VEHICLE_LENGTH, VEHICLE_WIDTH = 4.5, 1.9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tiny_coster_cfg.yaml")
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

    torch.manual_seed(0)
    device = torch.device("cpu")

    coster_cfg = COSTERConfig(**cfg["model"])
    model = COSTERModel(coster_cfg).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    n_agents, n_poly, n_pts = cfg["sim"]["n_agents"], cfg["sim"]["n_polylines"], cfg["sim"]["n_points"]
    polylines = torch.randn(1, n_poly, n_pts, coster_cfg.map_point_dim)
    point_mask = torch.ones(1, n_poly, n_pts, dtype=torch.bool)
    agent_histories = torch.randn(1, n_agents, coster_cfg.n_history_steps, coster_cfg.agent_state_dim)
    target_idx = torch.zeros(1, dtype=torch.long)
    target_future_states = torch.randn(1, coster_cfg.n_collision_time_bins, coster_cfg.state_dim)

    with torch.no_grad():
        out = model(polylines, point_mask, agent_histories, target_idx, target_future_states)
    rollout = out["rollout"][0]                     # [t_rev, 4] -- (x, y, heading, v), collision-instant -> earliest
    forward_traj = torch.flip(rollout, dims=[0])     # reverse -> earliest -> collision, for natural playback

    xs, ys = forward_traj[:, 0].tolist(), forward_traj[:, 1].tolist()
    headings = forward_traj[:, 2].tolist()
    pad = max(VEHICLE_LENGTH, 2.0)

    hw, hl = VEHICLE_WIDTH / 2.0, VEHICLE_LENGTH / 2.0
    local_corners = torch.tensor([[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]])

    def corners_at(x, y, theta):
        cos_t, sin_t = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        rot_x = local_corners[:, 0] * cos_t - local_corners[:, 1] * sin_t
        rot_y = local_corners[:, 0] * sin_t + local_corners[:, 1] * cos_t
        return torch.stack([rot_x + x, rot_y + y], dim=-1)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_xlabel("x (meters, scene frame)")
    ax.set_ylabel("y (meters, scene frame)")
    weights_note = "trained checkpoint" if args.checkpoint else "untrained/random weights"
    ax.set_title(f"COSTER -- time-reversed rollout, played forward to the collision\n({weights_note})", fontsize=9.5)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)

    trail_line, = ax.plot([], [], "-o", color=BLUE, linewidth=2, markersize=3, zorder=4, label="lead-up trajectory")
    collision_marker = ax.scatter([xs[-1]], [ys[-1]], s=0, color=ORANGE, marker="*", zorder=6, label="collision instant")
    poly_order = [0, 1, 3, 2]
    vehicle_polygon = Polygon([[0, 0]] * 4, closed=True, facecolor=BLUE, edgecolor="#0b0b0b", linewidth=1.2, zorder=5)
    ax.add_patch(vehicle_polygon)
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    n_frames = len(xs)

    def update(frame_idx):
        corners = corners_at(xs[frame_idx], ys[frame_idx], headings[frame_idx]).numpy()
        vehicle_polygon.set_xy(corners[poly_order])
        trail_line.set_data(xs[: frame_idx + 1], ys[: frame_idx + 1])
        is_collision_frame = frame_idx == n_frames - 1
        collision_marker.set_sizes([220 if is_collision_frame else 0])
        vehicle_polygon.set_facecolor(ORANGE if is_collision_frame else BLUE)
        step_text.set_text(f"step {frame_idx + 1}/{n_frames}" + ("  ** COLLISION **" if is_collision_frame else ""))
        return vehicle_polygon, trail_line, collision_marker, step_text

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out} ({n_frames} frames)")


if __name__ == "__main__":
    main()
