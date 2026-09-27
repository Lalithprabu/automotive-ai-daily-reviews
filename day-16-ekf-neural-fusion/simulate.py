"""
simulate.py -- renders an animated GIF from a trained checkpoint showing:
    - past trajectory (input, solid black)
    - 4 EKF candidate futures (dashed, distinct colors)
    - neural-only prediction (blue)
    - fused prediction (red)
    - ground truth future (green)
    - an animated marker at the agent's current position for gt/neural/fused
    - a live telemetry sub-panel: running ADE-so-far for neural-only vs fused

Usage: python simulate.py --config config.yaml
"""

from __future__ import annotations

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import yaml

from src.dataset import TrajectoryDataset, REGIMES
from src.ekf_motion_models import build_ekf_bank, run_ekf_bank
from src.neural_predictor import NeuralTrajectoryPredictor
from src.fusion_network import LateFusionNetwork
from src.utils import set_seed


EKF_COLORS = ["#888888", "#8e44ad", "#a0522d", "#00bcd4"]  # CV, CA, CTRV, CTRA
EKF_NAMES = ["CV", "CA", "CTRV", "CTRA"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def running_ade(pred_so_far: np.ndarray, gt_so_far: np.ndarray) -> float:
    if len(pred_so_far) == 0:
        return 0.0
    diff = pred_so_far - gt_so_far
    return float(np.linalg.norm(diff, axis=-1).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    d = cfg["data"]
    set_seed(d["seed"])

    device = torch.device("cpu")
    ckpt_path = cfg["train"]["checkpoint_path"]
    print(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    np_cfg = cfg["neural_predictor"]
    predictor = NeuralTrajectoryPredictor(
        hidden_size=np_cfg["hidden_size"], latent_size=np_cfg["latent_size"],
        encoder_layers=np_cfg["encoder_layers"], decoder_layers=np_cfg["decoder_layers"],
    )
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    predictor.eval()

    f_cfg = cfg["fusion"]
    fusion = LateFusionNetwork(
        future_len=d["future_len"], num_ekf=f_cfg["num_candidates"] - 1,
        embed_dim=f_cfg["embed_dim"], attn_hidden=f_cfg["attn_hidden"], residual_hidden=f_cfg["residual_hidden"],
    )
    fusion.load_state_dict(ckpt["fusion_state_dict"])
    fusion.eval()

    # Build the same test split used in train.py (seed+2) so scene_index is
    # reproducible, and pick a "turning" regime scene if possible since that
    # is the most visually informative regime for this paper's claim.
    test_ds = TrajectoryDataset(d["num_test"], d["history_len"], d["future_len"], d["dt"], d["noise_std"], d["seed"] + 2)

    scene_index = cfg["simulate"]["scene_index"]
    # prefer a turning-regime scene near scene_index for a more informative GIF
    turning_indices = [i for i, s in enumerate(test_ds.scenes) if s.regime == "turning"]
    if turning_indices:
        scene_index = turning_indices[scene_index % len(turning_indices)]

    scene = test_ds.scenes[scene_index]
    print(f"Simulating scene #{scene_index} (regime = '{scene.regime}')")

    history_np = scene.history            # [T_h, 2]
    future_np = scene.future              # [T_f, 2] ground truth
    history_t = torch.tensor(history_np, dtype=torch.float32).unsqueeze(0)  # [1, T_h, 2]

    with torch.no_grad():
        neural_pred, _ = predictor(history_t, future=None, future_len=d["future_len"], sample_z=False)
    neural_pred_np = neural_pred.squeeze(0).numpy()   # [T_f, 2]

    ekf_models = build_ekf_bank(cfg["ekf"])
    ekf_cand, ekf_unc = run_ekf_bank(ekf_models, history_np, d["dt"], d["future_len"])  # [4,T_f,2], [4,T_f]

    with torch.no_grad():
        fused, weights = fusion(
            neural_pred,
            torch.tensor(ekf_cand, dtype=torch.float32).unsqueeze(0),
            torch.tensor(ekf_unc, dtype=torch.float32).unsqueeze(0),
        )
    fused_np = fused.squeeze(0).numpy()               # [T_f, 2]

    T_f = d["future_len"]

    # ---- figure layout: trajectory overlay (left) + telemetry (right) ----
    fig, (ax_traj, ax_tel) = plt.subplots(1, 2, figsize=(12, 5.5))

    all_x = np.concatenate([history_np[:, 0], future_np[:, 0], neural_pred_np[:, 0], fused_np[:, 0], ekf_cand[:, :, 0].flatten()])
    all_y = np.concatenate([history_np[:, 1], future_np[:, 1], neural_pred_np[:, 1], fused_np[:, 1], ekf_cand[:, :, 1].flatten()])
    pad = 3.0
    ax_traj.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax_traj.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax_traj.set_aspect("equal")
    ax_traj.set_title(f"Trajectory fusion -- scene #{scene_index} ({scene.regime})")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")

    # static: history (input)
    ax_traj.plot(history_np[:, 0], history_np[:, 1], "o-", color="black", label="History (input)", markersize=4)

    # static: EKF candidate dashed lines (full, drawn once)
    for k in range(ekf_cand.shape[0]):
        ax_traj.plot(ekf_cand[k, :, 0], ekf_cand[k, :, 1], "--", color=EKF_COLORS[k], linewidth=1.3,
                     label=f"EKF {EKF_NAMES[k]}", alpha=0.8)

    # animated lines (grow frame by frame)
    gt_line, = ax_traj.plot([], [], "-", color="#2ecc71", linewidth=2, label="Ground truth")
    neural_line, = ax_traj.plot([], [], "-", color="#3498db", linewidth=2, label="Neural-only")
    fused_line, = ax_traj.plot([], [], "-", color="#e74c3c", linewidth=2.4, label="Fused (ours)")

    gt_marker, = ax_traj.plot([], [], "s", color="#2ecc71", markersize=9, markeredgecolor="black")
    neural_marker, = ax_traj.plot([], [], "o", color="#3498db", markersize=8, markeredgecolor="black")
    fused_marker, = ax_traj.plot([], [], "D", color="#e74c3c", markersize=8, markeredgecolor="black")

    ax_traj.legend(loc="upper left", fontsize=7, ncol=2)

    # ---- telemetry sub-panel: running ADE-so-far ----
    ax_tel.set_xlim(0, T_f)
    max_err_guess = max(
        float(np.linalg.norm(neural_pred_np - future_np, axis=-1).max()),
        float(np.linalg.norm(fused_np - future_np, axis=-1).max()),
    )
    ax_tel.set_ylim(0, max_err_guess * 1.3 + 1e-3)
    ax_tel.set_title("Running ADE-so-far")
    ax_tel.set_xlabel("future step (0.5s each)")
    ax_tel.set_ylabel("ADE (m)")
    ax_tel.grid(alpha=0.3)

    neural_ade_line, = ax_tel.plot([], [], "-o", color="#3498db", label="Neural-only", markersize=4)
    fused_ade_line, = ax_tel.plot([], [], "-o", color="#e74c3c", label="Fused (ours)", markersize=4)
    ax_tel.legend(loc="upper left", fontsize=8)

    def init():
        for ln in (gt_line, neural_line, fused_line, gt_marker, neural_marker, fused_marker,
                   neural_ade_line, fused_ade_line):
            ln.set_data([], [])
        return (gt_line, neural_line, fused_line, gt_marker, neural_marker, fused_marker,
                neural_ade_line, fused_ade_line)

    def update(frame):
        t = frame + 1  # reveal 1..T_f points
        gt_line.set_data(future_np[:t, 0], future_np[:t, 1])
        neural_line.set_data(neural_pred_np[:t, 0], neural_pred_np[:t, 1])
        fused_line.set_data(fused_np[:t, 0], fused_np[:t, 1])

        gt_marker.set_data([future_np[t - 1, 0]], [future_np[t - 1, 1]])
        neural_marker.set_data([neural_pred_np[t - 1, 0]], [neural_pred_np[t - 1, 1]])
        fused_marker.set_data([fused_np[t - 1, 0]], [fused_np[t - 1, 1]])

        steps = np.arange(1, t + 1)
        n_ade = [running_ade(neural_pred_np[:s], future_np[:s]) for s in steps]
        f_ade = [running_ade(fused_np[:s], future_np[:s]) for s in steps]
        neural_ade_line.set_data(steps, n_ade)
        fused_ade_line.set_data(steps, f_ade)

        return (gt_line, neural_line, fused_line, gt_marker, neural_marker, fused_marker,
                neural_ade_line, fused_ade_line)

    fps = cfg["simulate"]["fps"]
    anim = animation.FuncAnimation(fig, update, frames=T_f, init_func=init, blit=True, interval=1000 / fps)

    out_path = cfg["simulate"]["out_path"]
    t0 = time.time()
    writer = animation.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    elapsed = time.time() - t0
    plt.close(fig)

    print(f"Saved simulation GIF to {out_path} (render time: {elapsed:.2f}s, {T_f} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
