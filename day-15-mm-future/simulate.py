"""Render a synthetic "MM-Future in action" simulation GIF.

For one held-out bimodal driving scene (ego must swerve left or right
around a blocker agent — a real fork in outcomes, not a straight line),
this steps through the model's flow-matching ODE integration and animates:

  Panel A (left)  — top-down BEV "sensor feed": history + ground-truth
                    future agent motion, with all M candidate trajectories
                    converging from noise toward the ground truth, colored
                    by the scorer's live confidence, and the top-ranked
                    proposal highlighted.
  Panel B (right) — ground-truth vs model-decoded future occupancy
                    ("input vs ground truth vs prediction" overlay).
  Panel C (bottom)— telemetry: the proposal scorer's confidence in the
                    eventual winning mode over the course of generation, and
                    running ADE-to-ground-truth of the top-ranked proposal.

Usage:
    python simulate.py --config config.yaml --checkpoint checkpoints/mm_future.pt
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.animation import PillowWriter

from src.data import MMFutureDataset, SyntheticDrivingScenes
from src.model import MMFuture

# Project-standard categorical palette (see AUTOMOTIVE AI TECHNICAL REFERENCE
# MATRIX, section 5).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#898781"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ckpt_path = args.checkpoint or cfg["train"]["checkpoint_path"]
    out_path = args.out or cfg["simulate"]["out_path"]

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MMFuture(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.action_prior.means = ckpt["action_prior_means"]
    model.action_prior.stds = ckpt["action_prior_stds"]
    model.action_prior.num_clusters = ckpt["action_prior_k"]
    model.eval()

    d = ckpt["cfg"]["data"]
    rng = np.random.default_rng(2026)
    gen = SyntheticDrivingScenes(d["grid_size"], d["num_agents"], d["history_frames"],
                                  d["future_frames"], rng)
    scene = gen.sample()

    val_ds = MMFutureDataset(1, d["grid_size"], d["num_agents"], d["history_frames"],
                              d["future_frames"], d["camera_crop"], seed=999)
    val_ds.scenes = [scene]
    sample = val_ds[0]

    history_cameras = sample["history_cameras"].unsqueeze(0)
    ego_last_pos = sample["ego_history"][-1].unsqueeze(0)
    gt_trajectory = sample["gt_trajectory"].numpy()
    ego_history = sample["ego_history"].numpy()
    frames_np = sample["frames"].numpy()

    ode_steps = ckpt["cfg"]["model"]["flow_ode_steps"]
    result = model.generate(history_cameras, ego_last_pos, device, ode_steps=ode_steps)
    trace = result["trace"][:, 0].detach().numpy()          # [steps+1, M, Ta, 2]
    score_trace = torch.softmax(result["score_trace"][:, 0], dim=-1).detach().numpy()  # [steps+1, M]
    final_scores = result["scores"][0].detach().numpy()
    winner = int(final_scores.argmax())
    bev_pred = result["bev_pred"][0, winner].detach().numpy()  # [chunks, out, out]

    # Oracle reference (hindsight-only, not used by the model): which proposal
    # actually ends up closest to ground truth. Shown alongside the scorer's
    # real pick because this reconstruction's proposal scorer under-performs
    # its own flow-matching generator (see README "Honest results" — final
    # validation top-1 ranking accuracy is ~30%), so the two frequently
    # diverge. Hiding that would misrepresent what the model actually does.
    final_ade_all = np.linalg.norm(trace[-1] - gt_trajectory[None], axis=-1).mean(axis=-1)
    oracle = int(final_ade_all.argmin())

    hist_frames = d["history_frames"]
    fpc = d["frames_per_chunk"]
    future_frames_np = frames_np[hist_frames:]
    gt_bev = future_frames_np.max(axis=1)  # [Tf, g, g] combine agent + ego channel
    n_chunks = bev_pred.shape[0]
    gt_bev_chunks = gt_bev.reshape(n_chunks, fpc, gt_bev.shape[-2], gt_bev.shape[-1]).max(axis=1)

    num_steps = trace.shape[0]
    num_proposals = trace.shape[1]

    fig = plt.figure(figsize=(11, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.1], width_ratios=[1.3, 1, 1])
    ax_bev = fig.add_subplot(gs[0, 0])
    ax_gtocc = fig.add_subplot(gs[0, 1])
    ax_predocc = fig.add_subplot(gs[0, 2])
    ax_telemetry = fig.add_subplot(gs[1, :])
    ax_telemetry2 = ax_telemetry.twinx()  # created once; cleared (not recreated) each frame

    g = d["grid_size"]

    def draw_frame(step_idx):
        for ax in (ax_bev, ax_gtocc, ax_predocc, ax_telemetry, ax_telemetry2):
            ax.clear()
            ax.set_facecolor(SURFACE)

        # --- Panel A: BEV with converging proposals ---
        background = frames_np[hist_frames - 1, 0]
        ax_bev.imshow(background, cmap="Greys", origin="lower", alpha=0.35, extent=(0, g, 0, g))
        ax_bev.plot(ego_history[:, 0], ego_history[:, 1], color=INK2, lw=2, marker="o", ms=3, label="Ego history")
        ax_bev.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], color=AQUA, lw=2.5, ls="--", label="Ground truth future")

        probs = score_trace[step_idx]
        for p in range(num_proposals):
            if p in (winner, oracle):
                continue
            traj = trace[step_idx, p]
            alpha = 0.25 + 0.45 * probs[p]
            ax_bev.plot(traj[:, 0], traj[:, 1], color=BLUE, lw=1.1, alpha=min(alpha, 0.85))
        if oracle != winner:
            traj_o = trace[step_idx, oracle]
            ax_bev.plot(traj_o[:, 0], traj_o[:, 1], color=YELLOW, lw=2.2, ls=":",
                        label="Best available proposal (oracle, not selected)")
        traj_w = trace[step_idx, winner]
        ax_bev.plot(traj_w[:, 0], traj_w[:, 1], color=ORANGE, lw=3.0, alpha=0.95,
                    label="Top-ranked proposal (scorer's pick)")
        ax_bev.scatter([ego_history[-1, 0]], [ego_history[-1, 1]], color=INK, s=40, zorder=5)
        t_frac = step_idx / (num_steps - 1)
        ax_bev.set_title(f"MM-Future generation — flow step {step_idx}/{num_steps-1}  (t={t_frac:.2f})",
                          color=INK, fontsize=10, loc="left")
        ax_bev.set_xlim(0, g)
        ax_bev.set_ylim(0, g)
        ax_bev.set_aspect("equal")
        ax_bev.legend(loc="upper left", fontsize=6.5, framealpha=0.85)
        ax_bev.set_xticks([]); ax_bev.set_yticks([])

        # --- Panel B: ground-truth future occupancy ---
        ax_gtocc.imshow(gt_bev_chunks[-1], cmap="Greys", origin="lower", vmin=0, vmax=1)
        ax_gtocc.set_title("Ground-truth future\n(occupancy, last chunk)", color=INK, fontsize=9)
        ax_gtocc.set_xticks([]); ax_gtocc.set_yticks([])

        # --- Panel C: predicted future occupancy (decoded from scene tokens) ---
        blend = step_idx / (num_steps - 1)
        shown = bev_pred[-1] * blend + 0.0 * (1 - blend)
        cmap_pred = matplotlib.colors.LinearSegmentedColormap.from_list("pred", [SURFACE, ORANGE])
        ax_predocc.imshow(shown, cmap=cmap_pred, origin="lower", vmin=0, vmax=1)
        ax_predocc.set_title("Model-decoded future\n(scene tokens → occupancy)", color=INK, fontsize=9)
        ax_predocc.set_xticks([]); ax_predocc.set_yticks([])

        # --- Panel D: telemetry ---
        steps_axis = np.arange(step_idx + 1)
        winner_conf = score_trace[: step_idx + 1, winner]
        ax_telemetry.plot(steps_axis, winner_conf, color=ORANGE, lw=2, marker="o", ms=3,
                           label="Scorer confidence (top-ranked mode)")
        ade_over_time = np.linalg.norm(trace[: step_idx + 1, winner] - gt_trajectory[None], axis=-1).mean(axis=-1)
        ax_telemetry2.plot(steps_axis, ade_over_time, color=BLUE, lw=2, marker="s", ms=3, label="Top-1 ADE to GT (m)")
        ax_telemetry.set_xlim(0, num_steps - 1)
        ax_telemetry.set_ylim(0, 1.02)
        ax_telemetry.set_xlabel("ODE integration step", color=INK2, fontsize=8)
        ax_telemetry.set_ylabel("confidence", color=ORANGE, fontsize=8)
        ax_telemetry2.set_ylabel("ADE (grid units)", color=BLUE, fontsize=8)
        ax_telemetry.tick_params(colors=INK2, labelsize=7)
        ax_telemetry2.tick_params(colors=INK2, labelsize=7)
        lines1, labels1 = ax_telemetry.get_legend_handles_labels()
        lines2, labels2 = ax_telemetry2.get_legend_handles_labels()
        ax_telemetry.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6.5)
        ax_telemetry.set_title("Telemetry: proposal ranking confidence & trajectory error during generation",
                                color=INK, fontsize=9, loc="left")

        fig.suptitle("MM-Future — Multi-Mode Joint World-Action Modeling (synthetic reconstruction demo)",
                     color=INK, fontsize=12, fontweight="bold")

    writer = PillowWriter(fps=4)
    with writer.saving(fig, out_path, dpi=110):
        for step_idx in range(num_steps):
            draw_frame(step_idx)
            writer.grab_frame()
        for _ in range(4):  # hold on the final, fully-converged frame
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved simulation GIF to {out_path}")
    print(f"Ground truth mode: {'go-left' if scene['go_left'] else 'go-right'}")
    print(f"Scorer's top-ranked proposal: #{winner} (softmax confidence "
          f"{torch.softmax(torch.tensor(final_scores), dim=0)[winner]:.3f}) | ADE to GT: "
          f"{final_ade_all[winner]:.3f} grid units")
    print(f"Oracle best-available proposal: #{oracle} | ADE to GT: {final_ade_all[oracle]:.3f} grid units")
    if oracle != winner:
        print("NOTE: the scorer did not pick the best available proposal in this scenario — "
              "see README 'Honest results' for why (final validation top-1 ranking accuracy ~30%).")


if __name__ == "__main__":
    main()
