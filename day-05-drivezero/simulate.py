"""
Animated top-down simulation of DriveZero's deployed path: at each step,
DriveVFM fuses (synthetic, frozen-VFM-shaped) per-source feature tokens into
unified driving tokens, `value_guided_action_search` samples K candidate
(steering, acceleration) actions from the camera-only student and picks the
teacher-value-scored best one, and a simple bicycle-model integrator turns
that action into the next ego pose -- repeated for n_steps to build a full
driven trajectory, then played back frame by frame. Saved as an animated
GIF -- GitHub renders GIFs inline in READMEs.

Run with: python simulate.py --config tiny_drivezero_cfg.yaml
"""
import argparse
import math

import torch
import yaml
from matplotlib.patches import Polygon

from src.models.drivezero import DriveVFMConfig, DriveVFM, PrivilegedTeacherPolicy, CameraOnlyStudentPolicy, value_guided_action_search

BLUE = "#2a78d6"
ORANGE = "#eb6834"
VEHICLE_LENGTH, VEHICLE_WIDTH = 4.5, 1.9


def bicycle_step(x, y, theta, v, steering, accel, dt, wheelbase=2.7):
    v = max(0.0, v + accel * dt)
    x = x + v * math.cos(theta) * dt
    y = y + v * math.sin(theta) * dt
    theta = theta + (v / wheelbase) * math.tan(steering) * dt
    return x, y, theta, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tiny_drivezero_cfg.yaml")
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

    vfm_cfg = DriveVFMConfig(**cfg["drive_vfm"])
    drive_vfm = DriveVFM(vfm_cfg).to(device)
    teacher = PrivilegedTeacherPolicy(vfm_cfg.model_dim, cfg["teacher"]["privileged_dim"],
                                       cfg["teacher"]["action_dim"], hidden=cfg["teacher"]["hidden"]).to(device)
    student = CameraOnlyStudentPolicy(vfm_cfg.model_dim, cfg["student"]["goal_dim"],
                                       cfg["teacher"]["action_dim"], hidden=cfg["student"]["hidden"]).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        drive_vfm.load_state_dict(state["drive_vfm"])
        teacher.load_state_dict(state["teacher"])
        student.load_state_dict(state["student"])
    drive_vfm.eval(); teacher.eval(); student.eval()

    n_steps = cfg["sim"]["n_steps"]
    n_tok = cfg["sim"]["n_tokens_per_source"]
    dt = cfg["sim"]["dt"]
    k = cfg["sim"]["search_k"]

    x, y, theta, v = 0.0, 0.0, 0.0, 3.0
    xs, ys, headings = [x], [y], [theta]

    goal_embed = torch.randn(1, cfg["student"]["goal_dim"])
    with torch.no_grad():
        for step in range(n_steps):
            features = {name: torch.randn(1, n_tok, dim) for name, dim in vfm_cfg.source_dims.items()}
            unified_tokens = drive_vfm(features)
            privileged_estimate = torch.randn(1, cfg["teacher"]["privileged_dim"])

            action = value_guided_action_search(student, teacher, unified_tokens, goal_embed,
                                                 privileged_estimate, k=k)[0]  # [action_dim] -> (steering, accel)
            steering, accel = float(action[0]) * 0.3, float(action[1])  # scale steering to a plausible radian range

            x, y, theta, v = bicycle_step(x, y, theta, v, steering, accel, dt)
            xs.append(x); ys.append(y); headings.append(theta)

    pad = max(VEHICLE_LENGTH, 3.0)
    hw, hl = VEHICLE_WIDTH / 2.0, VEHICLE_LENGTH / 2.0
    local_corners = torch.tensor([[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]])

    def corners_at(px, py, ptheta):
        cos_t, sin_t = math.cos(ptheta), math.sin(ptheta)
        rot_x = local_corners[:, 0] * cos_t - local_corners[:, 1] * sin_t
        rot_y = local_corners[:, 0] * sin_t + local_corners[:, 1] * cos_t
        return torch.stack([rot_x + px, rot_y + py], dim=-1)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("y (meters)")
    weights_note = "trained checkpoint" if args.checkpoint else "untrained/random weights -- expect an erratic path"
    ax.set_title(f"DriveZero -- value-guided action search, driven rollout\n({weights_note})", fontsize=9.5)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)

    trail_line, = ax.plot([], [], "-o", color=BLUE, linewidth=2, markersize=3, zorder=4)
    poly_order = [0, 1, 3, 2]
    vehicle_polygon = Polygon([[0, 0]] * 4, closed=True, facecolor=ORANGE, edgecolor="#0b0b0b", linewidth=1.2, zorder=5)
    ax.add_patch(vehicle_polygon)
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)

    def update(frame_idx):
        corners = corners_at(xs[frame_idx], ys[frame_idx], headings[frame_idx]).numpy()
        vehicle_polygon.set_xy(corners[poly_order])
        trail_line.set_data(xs[: frame_idx + 1], ys[: frame_idx + 1])
        step_text.set_text(f"step {frame_idx + 1}/{len(xs)}")
        return vehicle_polygon, trail_line, step_text

    anim = FuncAnimation(fig, update, frames=len(xs), interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out} ({len(xs)} frames)")


if __name__ == "__main__":
    main()
