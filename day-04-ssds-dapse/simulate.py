"""
Animated top-down simulation contrasting SSDS + DAPSE's two roles from the
SAME frozen denoiser: the ego vehicle's PLANNER-role trajectory (comfort
energy) vs. an other agent's SCENARIO-GENERATOR-role trajectory
(adversarial-proximity energy, pulled toward a near-miss with the ego) are
sampled by repeatedly applying `dapse_guided_step` across outer diffusion
timesteps, then played back frame by frame. Saved as an animated GIF --
GitHub renders GIFs inline in READMEs.

Run with: python simulate.py --config tiny_ssds_cfg.yaml
"""
import argparse

import torch
import yaml

from src.models.ssds_dapse import (
    SSDSConfig, SSDSDenoiser, dapse_guided_step, comfort_energy, adversarial_proximity_energy,
)

BLUE = "#2a78d6"
ORANGE = "#eb6834"


def run_dapse_rollout(model, ctx_tokens, nav_embed, cfg, energy_fn, n_outer_steps, r_t, beta, eta, n_inner_steps):
    x_t = torch.randn(ctx_tokens.shape[0], N_AGENTS, N_HORIZON, 3)
    for step in range(n_outer_steps, 0, -1):
        t_index = torch.full((ctx_tokens.shape[0],), step * (1000 // n_outer_steps))
        x_t = dapse_guided_step(x_t, model, ctx_tokens, t_index, nav_embed,
                                 r_t=r_t, energy_fn=energy_fn, beta=beta, eta=eta,
                                 n_inner_steps=n_inner_steps)
    return x_t  # [B, A, H, 3]


def main():
    global N_AGENTS, N_HORIZON
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tiny_ssds_cfg.yaml")
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
    N_AGENTS = cfg["data"]["n_agents"]
    N_HORIZON = cfg["data"]["n_horizon"]
    C = cfg["data"]["n_context"]

    ssds_cfg = SSDSConfig(**cfg["model"], max_traj_tokens=N_AGENTS * N_HORIZON)
    model = SSDSDenoiser(ssds_cfg).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    ctx_tokens = torch.randn(1, C, ssds_cfg.ctx_feat_dim)
    nav_embed = torch.randn(1, ssds_cfg.cond_dim)
    dp = cfg["dapse"]

    planner_traj = run_dapse_rollout(model, ctx_tokens, nav_embed, ssds_cfg, comfort_energy,
                                      dp["n_outer_steps"], dp["r_t"], dp["beta"], dp["eta"], dp["n_inner_steps"])
    scenario_traj = run_dapse_rollout(model, ctx_tokens, nav_embed, ssds_cfg, adversarial_proximity_energy,
                                       dp["n_outer_steps"], dp["r_t"], dp["beta"], dp["eta"], dp["n_inner_steps"])

    ego_planner = planner_traj[0, 0].detach()      # [H, 3] -- ego under PLANNER role
    ego_scenario = scenario_traj[0, 1].detach()    # [H, 3] -- a non-ego agent under SCENARIO-GENERATOR role

    xs_p, ys_p = ego_planner[:, 0].tolist(), ego_planner[:, 1].tolist()
    xs_s, ys_s = ego_scenario[:, 0].tolist(), ego_scenario[:, 1].tolist()

    all_x = xs_p + xs_s
    all_y = ys_p + ys_s
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y), 0.5)
    pad = span * 0.6 + 0.5   # pad scaled to the actual trajectory extent, not a fixed vehicle length --
                              # with untrained weights the predicted span can be much smaller than a real car

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_xlabel("x (meters, scene frame)")
    ax.set_ylabel("y (meters, scene frame)")
    weights_note = "trained checkpoint" if args.checkpoint else "untrained/random weights"
    ax.set_title(f"SSDS + DAPSE -- same frozen model, two roles\n({weights_note})", fontsize=9.5)
    ax.grid(True, color="#e1e0d9", linewidth=0.8)

    planner_trail, = ax.plot([], [], "-o", color=BLUE, linewidth=2, markersize=4,
                              zorder=4, label="PLANNER role (comfort_energy)")
    scenario_trail, = ax.plot([], [], "-o", color=ORANGE, linewidth=2, markersize=4,
                               zorder=4, label="SCENARIO-GENERATOR role (adversarial_proximity_energy)")
    planner_head = ax.scatter([], [], s=90, color=BLUE, edgecolors="#0b0b0b", zorder=5)
    scenario_head = ax.scatter([], [], s=90, color=ORANGE, edgecolors="#0b0b0b", zorder=5)
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    n_frames = len(xs_p)

    def update(frame_idx):
        planner_trail.set_data(xs_p[: frame_idx + 1], ys_p[: frame_idx + 1])
        scenario_trail.set_data(xs_s[: frame_idx + 1], ys_s[: frame_idx + 1])
        planner_head.set_offsets([[xs_p[frame_idx], ys_p[frame_idx]]])
        scenario_head.set_offsets([[xs_s[frame_idx], ys_s[frame_idx]]])
        step_text.set_text(f"step {frame_idx + 1}/{n_frames}")
        return planner_trail, scenario_trail, planner_head, scenario_head, step_text

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out} ({n_frames} frames)")


if __name__ == "__main__":
    main()
