"""
Animated top-down simulation of PART running over a sequence of radar
frames: a handful of moving point clusters (simulated vehicles/pedestrians,
each with its own ground-truth velocity) plus background clutter drift
across the scene, and PART's top-scoring queries are drawn each frame as a
marker + velocity arrow at their predicted surface point. Saved as an
animated GIF -- GitHub renders GIFs inline in READMEs.

Note: with untrained/random weights, existence scores are close to random,
so "top-K detections" mostly illustrates the visualization pipeline (radar
points -> DAQI query seeds -> per-query existence/point/velocity heads)
rather than real detection quality -- see the README for the caveat this
script also prints.

Run with: python simulate.py --config tiny_part_cfg.yaml
"""
import argparse

import torch
import yaml

from src.models.part_model import PARTConfig, PARTModel

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK_MUTED = "#898781"


def simulate_radar_frames(n_frames: int, n_points: int, n_clusters: int = 3, seed: int = 0):
    """
    Builds a sequence of synthetic radar point clouds: n_clusters moving
    "objects" (each a small blob of points sharing one ground-truth
    velocity) plus uniform background clutter, drifting frame to frame.

    Returns: list of length n_frames, each a [n_points, 5] tensor
             (x, y, z, doppler_v, rcs).
    """
    g = torch.Generator().manual_seed(seed)
    cluster_centers = torch.rand(n_clusters, 2, generator=g) * 40 - 20         # [-20, 20] meters
    cluster_velocities = torch.randn(n_clusters, 2, generator=g) * 2.0          # m/s

    frames = []
    n_per_cluster = 8
    n_clutter = n_points - n_clusters * n_per_cluster
    for t in range(n_frames):
        pts = []
        for c in range(n_clusters):
            center = cluster_centers[c] + cluster_velocities[c] * t * 0.5
            local = torch.randn(n_per_cluster, 2, generator=g) * 0.6 + center
            radial_v = (cluster_velocities[c] * (local - center).sign()).norm(dim=-1) * 0 \
                + cluster_velocities[c].norm()
            z = torch.zeros(n_per_cluster, 1)
            doppler = radial_v.unsqueeze(-1)
            rcs = torch.rand(n_per_cluster, 1, generator=g) * 2 + 1.0
            pts.append(torch.cat([local, z, doppler, rcs], dim=-1))

        clutter_xy = torch.rand(n_clutter, 2, generator=g) * 40 - 20
        clutter = torch.cat([
            clutter_xy, torch.zeros(n_clutter, 1),
            torch.randn(n_clutter, 1, generator=g) * 0.3,
            torch.rand(n_clutter, 1, generator=g) * 0.5,
        ], dim=-1)
        pts.append(clutter)
        frames.append(torch.cat(pts, dim=0))
    return frames, cluster_centers, cluster_velocities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tiny_part_cfg.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="trajectory_simulation.gif")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    device = torch.device("cpu")
    part_cfg = PARTConfig(**cfg["model"])
    model = PARTModel(part_cfg).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    n_frames = cfg["data"]["n_frames"]
    n_points = cfg["data"]["n_points"]
    frames, cluster_centers, _ = simulate_radar_frames(n_frames, n_points)

    per_frame_points, per_frame_dets = [], []
    with torch.no_grad():
        for pts in frames:
            radar_points = pts.unsqueeze(0)                      # [1, P, 5]
            point_mask = torch.ones(1, radar_points.shape[1], dtype=torch.bool)
            out = model(radar_points, point_mask)
            scores = out["existence_logits"].squeeze(-1).squeeze(0)   # [Q]
            k = min(args.top_k, scores.shape[0])
            topk = scores.topk(k).indices
            det_points = out["surface_points"].squeeze(0)[topk][:, :2]     # [k, 2] (dx, dy) offsets -> treat as xy
            det_vel = out["velocities"].squeeze(0)[topk]                    # [k, 2]
            per_frame_points.append(pts.numpy())
            per_frame_dets.append((det_points.numpy(), det_vel.numpy()))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    pad = 5
    ax.set_xlim(-20 - pad, 20 + pad)
    ax.set_ylim(-20 - pad, 20 + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    weights_note = "trained checkpoint" if args.checkpoint else "untrained/random weights -- illustrates the pipeline, not real detection quality"
    ax.set_title(f"PART -- radar frames + top-{args.top_k} detections\n({weights_note})", fontsize=9.5)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)

    scatter = ax.scatter([], [], s=8, c=INK_MUTED, alpha=0.6, zorder=2, label="radar returns")
    det_scatter = ax.scatter([], [], s=60, c=ORANGE, edgecolors="#0b0b0b", zorder=4, label="top-k detections")
    quivers = []
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    def update(frame_idx):
        nonlocal quivers
        pts = per_frame_points[frame_idx]
        det_xy, det_vel = per_frame_dets[frame_idx]

        scatter.set_offsets(pts[:, :2])
        det_scatter.set_offsets(det_xy)

        for q in quivers:
            q.remove()
        quivers = []
        if len(det_xy) > 0:
            q = ax.quiver(det_xy[:, 0], det_xy[:, 1], det_vel[:, 0], det_vel[:, 1],
                           color=BLUE, scale=30, width=0.006, zorder=5)
            quivers.append(q)

        step_text.set_text(f"frame {frame_idx + 1}/{len(per_frame_points)}")
        return [scatter, det_scatter, step_text] + quivers

    anim = FuncAnimation(fig, update, frames=len(per_frame_points), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out} ({len(per_frame_points)} frames)")


if __name__ == "__main__":
    main()
