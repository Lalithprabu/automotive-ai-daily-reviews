"""
Animated top-down simulation of a Planning Expert predicted trajectory:
a generic vehicle footprint moves through the sampled waypoints, leaving a
trail. Saved as an animated GIF -- GitHub renders GIFs inline in READMEs,
so this is meant to be embedded there directly (no video codec needed).

Note: the paper does not specify a vehicle footprint size (the Planning
Expert predicts a point trajectory, not a footprint) -- a generic sedan-size
box (4.5m x 1.9m) is used here purely for visualization.

Run with: python simulate.py --config tiny_qwen_cfg.yaml
"""
import argparse

import torch
import yaml

from src.data.trajectory_dataset import TrajectoryDataset, collate_kv_cache_batch
from src.models.planning_expert import PlanningExpert
from src.sampling.euler_sampler import euler_sample

VEHICLE_LENGTH = 4.5   # generic sedan, for illustration only -- not from the paper
VEHICLE_WIDTH = 1.9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/planning_expert_base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="trajectory_simulation.gif")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon

    device = torch.device("cpu")
    model = PlanningExpert(**cfg["model"]).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = TrajectoryDataset(
        model_dim=cfg["model"]["model_dim"], cond_dim=cfg["model"]["cond_dim"],
        n_caches=cfg["model"]["n_caches"], n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["data"]["n_kv_heads"], l_ctx=cfg["data"]["l_ctx"],
        n_waypoints=cfg["model"]["n_waypoints"], synthetic=cfg["data"]["synthetic"],
        synthetic_length=1,
    )
    batch = collate_kv_cache_batch([dataset[0]])
    kv_cache = batch["kv_cache"]
    instruction_embed = batch["instruction_embed"].to(device)
    ego_state_embed = batch["ego_state_embed"].to(device)

    trajectory = euler_sample(
        model, kv_cache, instruction_embed, ego_state_embed,
        n_waypoints=cfg["model"]["n_waypoints"], n_steps=cfg["sampling"]["n_steps"],
        device=device,
    )[0].detach()   # [T, 3] -- (x, y, heading)

    xs, ys = trajectory[:, 0].tolist(), trajectory[:, 1].tolist()
    headings = trajectory[:, 2].tolist()
    pad = max(VEHICLE_LENGTH, 2.0)

    hw, hl = VEHICLE_WIDTH / 2.0, VEHICLE_LENGTH / 2.0
    local_corners = torch.tensor([[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]])

    def corners_at(x, y, theta):
        cos_t, sin_t = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        rot_x = local_corners[:, 0] * cos_t - local_corners[:, 1] * sin_t
        rot_y = local_corners[:, 0] * sin_t + local_corners[:, 1] * cos_t
        return torch.stack([rot_x + x, rot_y + y], dim=-1)   # [4, 2]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_xlabel("x (meters, ego frame)")
    ax.set_ylabel("y (meters, ego frame)")
    weights_note = "trained checkpoint" if args.checkpoint else "untrained/random weights"
    ax.set_title(f"Planning Expert predicted trajectory -- simulation\n"
                 f"({weights_note}; generic {VEHICLE_LENGTH}m x {VEHICLE_WIDTH}m box, not from the paper)",
                 fontsize=9.5)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)

    trail_line, = ax.plot([], [], "-o", color="#2a78d6", linewidth=2, markersize=4, zorder=4)
    poly_order = [0, 1, 3, 2]   # FL, FR, RR, RL winding
    vehicle_polygon = Polygon([[0, 0]] * 4, closed=True, facecolor="#eb6834",
                               edgecolor="#0b0b0b", linewidth=1.2, zorder=5)
    ax.add_patch(vehicle_polygon)
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)

    def update(frame_idx):
        corners = corners_at(xs[frame_idx], ys[frame_idx], headings[frame_idx]).numpy()
        vehicle_polygon.set_xy(corners[poly_order])
        trail_line.set_data(xs[: frame_idx + 1], ys[: frame_idx + 1])
        step_text.set_text(f"waypoint {frame_idx + 1}/{len(xs)}")
        return vehicle_polygon, trail_line, step_text

    anim = FuncAnimation(fig, update, frames=len(xs), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out} ({len(xs)} frames)")


if __name__ == "__main__":
    main()
