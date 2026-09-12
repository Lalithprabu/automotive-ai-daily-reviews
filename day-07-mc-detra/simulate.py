"""
Animated top-down simulation of MC-DeTra's output for one detected actor:
the t=0 detection box is drawn first, then all K forecast modes' position
trails grow simultaneously frame by frame out to the T-step horizon --
illustrating the shared [Object x Time x Mode] query volume's detection
(t=0) vs. forecast (t>0) read-out described in the paper. Saved as an
animated GIF -- GitHub renders GIFs inline in READMEs.

Run with: python simulate.py --config tiny_mcdetra_cfg.yaml
"""
import argparse

import torch
import yaml
from matplotlib.patches import Polygon

from src.models.mc_detra import MCDeTraConfig, MCDeTra

MODE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tiny_mcdetra_cfg.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="trajectory_simulation.gif")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--object-idx", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    torch.manual_seed(0)
    device = torch.device("cpu")

    mc_cfg = MCDeTraConfig(**cfg["model"])
    model = MCDeTra(mc_cfg).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    lidar_bev = torch.randn(1, mc_cfg.lidar_channels, cfg["sim"]["lidar_h"], cfg["sim"]["lidar_w"])
    map_tokens = torch.randn(1, cfg["sim"]["n_map_tokens"], mc_cfg.map_feat_dim)

    with torch.no_grad():
        out = model(lidar_bev, map_tokens)

    n = args.object_idx
    box = out["detection_boxes"][0, n].tolist()             # (x, y, length, width, heading)
    positions = out["positions"][0, n]                        # [T, K, 2]
    T, K, _ = positions.shape
    x0, y0, length, width, heading = box

    hw, hl = abs(width) / 2.0 + 0.5, abs(length) / 2.0 + 0.5   # keep a sane minimum footprint for visualization
    local_corners = torch.tensor([[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]])
    cos_t, sin_t = torch.cos(torch.tensor(heading)), torch.sin(torch.tensor(heading))
    rot_x = local_corners[:, 0] * cos_t - local_corners[:, 1] * sin_t
    rot_y = local_corners[:, 0] * sin_t + local_corners[:, 1] * cos_t
    box_corners = torch.stack([rot_x + x0, rot_y + y0], dim=-1).numpy()
    poly_order = [0, 1, 3, 2]

    all_x = positions[..., 0].flatten().tolist()
    all_y = positions[..., 1].flatten().tolist()
    pad = max(abs(length), 2.0) + 1.0

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_xlabel("x (meters, BEV frame)")
    ax.set_ylabel("y (meters, BEV frame)")
    weights_note = "trained checkpoint" if args.checkpoint else "untrained/random weights"
    ax.set_title(f"MC-DeTra -- object {n}: detection (t=0) + {K}-mode forecast\n({weights_note})", fontsize=9.5)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)

    det_polygon = Polygon(box_corners[poly_order], closed=True, facecolor="none",
                           edgecolor="#0b0b0b", linewidth=1.8, zorder=5, label="detection (t=0)")
    ax.add_patch(det_polygon)

    mode_lines = []
    for k in range(K):
        color = MODE_COLORS[k % len(MODE_COLORS)]
        line, = ax.plot([], [], "-o", color=color, linewidth=2, markersize=3, zorder=4, label=f"forecast mode {k}")
        mode_lines.append(line)

    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    pos_np = positions.numpy()  # [T, K, 2]

    def update(frame_idx):
        for k, line in enumerate(mode_lines):
            xs = pos_np[: frame_idx + 1, k, 0]
            ys = pos_np[: frame_idx + 1, k, 1]
            line.set_data(xs, ys)
        step_text.set_text(f"t={frame_idx} / {T - 1}" + ("  (detection)" if frame_idx == 0 else "  (forecast)"))
        return mode_lines + [det_polygon, step_text]

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out} ({T} frames, {K} modes)")


if __name__ == "__main__":
    main()
