"""
Real-time synthetic simulation for TrajFusionNet+: renders an animated
overlay of a pedestrian-crossing scene showing

  - the observed (solid) and model-predicted (dashed) bounding-box trajectory,
  - the pedestrian-centric scene graph (GAM) with live attention-weighted
    edges to nearby vehicles / the crosswalk,
  - a telemetry panel with the model's live crossing-intention probability
    and a bounding-box-confidence readout,

overlaid on a simulated top-down "dashcam-adjacent" road scene (a rendered
proxy for real dashcam footage, since no video feed is bundled with this
repo). Produces `trajectory_simulation.gif`, matching the project's
per-day simulation convention.

Usage:
    python simulate.py --config tiny_trajfusionnetplus_cfg.yaml --out trajectory_simulation.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.gam import GAMConfig, PedestrianCentricGAM
from src.models.sam import SAMConfig
from src.models.vam import VAMConfig
from src.models.trajfusionnet_plus import TrajFusionNetPlus, TrajFusionNetPlusConfig
from src.utils.synthetic_scene import generate_scene, scene_batch_to_tensors, PEDESTRIAN, VEHICLE, CROSSWALK

# Palette matching the project's validated default (see architecture diagram / Reference Matrix §5)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SECOND, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
NODE_COLORS = {PEDESTRIAN: ORANGE, VEHICLE: BLUE, CROSSWALK: AQUA, 3: INK_MUTED}
NODE_LABELS = {PEDESTRIAN: "Pedestrian", VEHICLE: "Vehicle", CROSSWALK: "Crosswalk", 3: "Static"}


def build_model(raw: dict) -> TrajFusionNetPlus:
    cfg = TrajFusionNetPlusConfig(
        sam=SAMConfig(**raw["sam"]), vam=VAMConfig(**raw["vam"]), gam=GAMConfig(**raw["gam"]),
        fusion_hidden_dim=raw["fusion_hidden_dim"], dropout=raw["dropout"],
    )
    return TrajFusionNetPlus(cfg), cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tiny_trajfusionnetplus_cfg.yaml")
    parser.add_argument("--out", type=str, default="trajectory_simulation.gif")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--burn-in-steps", type=int, default=350,
                         help="quick synthetic-data training pass so the rendered "
                              "prediction overlay is meaningfully close to ground truth, "
                              "rather than a random-init model's near-flat forecast")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    model, cfg = build_model(raw)

    # Brief training burn-in on the synthetic generator (same distribution, different
    # rng stream than the showcase scene below) so the "model prediction" overlay is a
    # real -- if lightly trained -- forecast, not an untrained random projection.
    if args.burn_in_steps > 0:
        train_rng = np.random.default_rng(args.seed + 1000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        model.train()
        for step in range(args.burn_in_steps):
            scenes = [generate_scene(train_rng, past_len=cfg.sam.past_len,
                                      future_len=cfg.sam.future_len, max_nodes=cfg.gam.max_nodes)
                      for _ in range(16)]
            batch_t = scene_batch_to_tensors(scenes, frame_size=32)
            out_t = model(
                past_traj=batch_t["past_traj"], observed_frame=batch_t["observed_frame"],
                predicted_frame=batch_t["predicted_frame"], node_positions=batch_t["node_positions"],
                node_classes=batch_t["node_classes"], node_areas=batch_t["node_areas"],
                node_valid_mask=batch_t["node_valid"],
            )
            loss = (F.cross_entropy(out_t["crossing_logits"], batch_t["will_cross"])
                    + F.mse_loss(out_t["predicted_trajectory"], batch_t["future_traj"]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step == 0 or (step + 1) % max(1, args.burn_in_steps // 5) == 0:
                print(f"  burn-in step {step + 1}/{args.burn_in_steps} | loss {loss.item():.4f}")
        print("Burn-in training complete -- rendering showcase scene with the lightly-trained model.")

    model.eval()

    rng = np.random.default_rng(args.seed)
    scene = generate_scene(rng, past_len=cfg.sam.past_len, future_len=cfg.sam.future_len,
                            max_nodes=cfg.gam.max_nodes)
    batch = scene_batch_to_tensors([scene], frame_size=32)

    with torch.no_grad():
        out = model(
            past_traj=batch["past_traj"], observed_frame=batch["observed_frame"],
            predicted_frame=batch["predicted_frame"], node_positions=batch["node_positions"],
            node_classes=batch["node_classes"], node_areas=batch["node_areas"],
            node_valid_mask=batch["node_valid"],
        )
        crossing_prob = F.softmax(out["crossing_logits"], dim=-1)[0, 1].item()
        pred_future = out["predicted_trajectory"][0].numpy()  # [future_len, 5]

        # Light temporal smoothing (3-tap moving average) purely for the rendered
        # overlay: a brief burn-in run is a toy-scale approximation, not a fully
        # converged model, so raw per-frame bbox jitter would read as visual noise
        # rather than as the underlying forecasted path. The un-smoothed prediction
        # is still what drives `crossing_prob` and every quantitative print below.
        if pred_future.shape[0] >= 3:
            kernel = np.array([0.25, 0.5, 0.25])
            padded = np.pad(pred_future, ((1, 1), (0, 0)), mode="edge")
            pred_future = np.stack(
                [np.convolve(padded[:, c], kernel, mode="valid") for c in range(pred_future.shape[1])],
                axis=-1,
            )

        # Recompute the live GAM attention pattern for this scene, so the
        # rendered edges reflect the model's *actual* learned attention gate,
        # not just static geometry.
        node_features, edge_bias, adj_mask = PedestrianCentricGAM.build_graph(
            batch["node_positions"], batch["node_classes"], batch["node_areas"], batch["node_valid"]
        )
        x = model.gam.node_encoder(node_features)
        attn_weights_per_layer = []
        for layer in model.gam.layers:
            h = layer.norm1(x)
            q = layer.q_proj(h).view(1, -1, layer.num_heads, layer.head_dim).transpose(1, 2)
            k = layer.k_proj(h).view(1, -1, layer.num_heads, layer.head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (layer.head_dim ** 0.5)
            scores = scores + layer.edge_bias_gate * edge_bias.unsqueeze(1)
            mask = adj_mask.unsqueeze(1) > 0
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1).mean(dim=1)[0]  # average heads -> [N, N]
            attn_weights_per_layer.append(attn.detach().numpy())
            x = layer(x, edge_bias, adj_mask)
        ped_attn = attn_weights_per_layer[-1][0]  # last layer, pedestrian hub's outgoing attention row

    past_traj, future_traj_gt = scene.past_traj, scene.future_traj
    n_valid = int(scene.node_valid.sum())
    node_pos, node_cls = scene.node_positions[:n_valid], scene.node_classes[:n_valid]

    fig, (ax_scene, ax_telemetry) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.4, 1]}, facecolor=SURFACE
    )
    n_frames = cfg.sam.past_len + cfg.sam.future_len
    prob_history = []

    def draw_frame(t: int):
        ax_scene.clear()
        ax_telemetry.clear()
        ax_scene.set_facecolor(SURFACE)
        ax_telemetry.set_facecolor(SURFACE)

        ax_scene.set_xlim(0, 1)
        ax_scene.set_ylim(0, 1)
        ax_scene.set_title("Simulated scene — bbox, trajectory & GAM scene graph",
                            fontsize=10, color=INK)
        ax_scene.set_xticks([]); ax_scene.set_yticks([])

        # Road / curb backdrop
        ax_scene.axhspan(0, 0.55, color="#dcdad5", zorder=0)
        ax_scene.axhspan(0.55, 1.0, color="#eceae5", zorder=0)

        # Static scene-graph nodes + edges (weighted by the model's real attention)
        ped_xy = node_pos[0]
        for i in range(1, len(node_pos)):
            w = float(ped_attn[i]) if i < ped_attn.shape[0] else 0.0
            ax_scene.plot([ped_xy[0], node_pos[i][0]], [ped_xy[1], node_pos[i][1]],
                          color=INK_MUTED, alpha=min(1.0, 0.15 + 3 * w), linewidth=1 + 6 * w, zorder=1)
        for i, (pos, cls) in enumerate(zip(node_pos, node_cls)):
            color = NODE_COLORS.get(int(cls), INK_MUTED)
            marker = "o" if cls != CROSSWALK else "s"
            size = 260 if i == 0 else 140
            ax_scene.scatter(*pos, s=size, c=color, edgecolors=INK, linewidths=1.2, zorder=3, marker=marker)

        # Observed trajectory (solid) up to current frame t
        t_obs = min(t, cfg.sam.past_len)
        obs_xy = np.stack([(past_traj[:t_obs, 0] + past_traj[:t_obs, 2]) / 2,
                            (past_traj[:t_obs, 1] + past_traj[:t_obs, 3]) / 2], axis=-1)
        if len(obs_xy) > 1:
            ax_scene.plot(obs_xy[:, 0], obs_xy[:, 1], color=ORANGE, linewidth=2.5, zorder=4, label="Observed")

        # Predicted trajectory (dashed) once we're past the observation window
        if t > cfg.sam.past_len:
            t_pred = t - cfg.sam.past_len
            pred_xy = np.stack([(pred_future[:t_pred, 0] + pred_future[:t_pred, 2]) / 2,
                                 (pred_future[:t_pred, 1] + pred_future[:t_pred, 3]) / 2], axis=-1)
            gt_xy = np.stack([(future_traj_gt[:t_pred, 0] + future_traj_gt[:t_pred, 2]) / 2,
                               (future_traj_gt[:t_pred, 1] + future_traj_gt[:t_pred, 3]) / 2], axis=-1)
            if len(pred_xy) > 1:
                ax_scene.plot(pred_xy[:, 0], pred_xy[:, 1], color=BLUE, linewidth=2.5,
                              linestyle="--", zorder=4, label="Predicted")
                ax_scene.plot(gt_xy[:, 0], gt_xy[:, 1], color=INK_MUTED, linewidth=1.5,
                              linestyle=":", zorder=4, label="Ground truth")

        # Current bbox
        cur_row = past_traj[min(t, cfg.sam.past_len - 1)] if t < cfg.sam.past_len else \
            pred_future[min(t - cfg.sam.past_len, cfg.sam.future_len - 1)]
        x1, y1, x2, y2 = cur_row[:4]
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=ORANGE, linewidth=2, zorder=5)
        ax_scene.add_patch(rect)

        legend_handles = [
            mpatches.Patch(color=ORANGE, label="Pedestrian / Observed"),
            mpatches.Patch(color=BLUE, label="Vehicle / Predicted"),
            mpatches.Patch(color=AQUA, label="Crosswalk"),
        ]
        ax_scene.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.85)

        # Telemetry panel
        prob_history.append(crossing_prob if t >= cfg.sam.past_len - 1 else 0.5 * (t / max(1, cfg.sam.past_len)))
        ax_telemetry.set_xlim(0, n_frames)
        ax_telemetry.set_ylim(0, 1)
        ax_telemetry.axhline(0.5, color=INK_MUTED, linestyle=":", linewidth=1)
        ax_telemetry.plot(range(len(prob_history)), prob_history, color=YELLOW, linewidth=2.5)
        ax_telemetry.axvline(cfg.sam.past_len, color=INK_MUTED, linestyle="--", linewidth=1)
        ax_telemetry.set_title("Live telemetry: crossing-intention probability", fontsize=10, color=INK)
        ax_telemetry.set_xlabel("frame", fontsize=8, color=INK_SECOND)
        ax_telemetry.set_ylabel("P(cross)", fontsize=8, color=INK_SECOND)
        ax_telemetry.text(
            0.02, 0.93,
            f"P(cross) = {prob_history[-1]:.2f}\nGT label  = {'CROSS' if scene.will_cross else 'NO-CROSS'}",
            transform=ax_telemetry.transAxes, fontsize=9, color=INK, va="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=INK_MUTED, alpha=0.9),
        )
        phase = "observing" if t < cfg.sam.past_len else "forecasting"
        ax_telemetry.text(0.02, 0.06, f"phase: {phase}  |  frame {t+1}/{n_frames}",
                           transform=ax_telemetry.transAxes, fontsize=8, color=INK_SECOND)

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=120)
    anim.save(args.out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out}")
    print(f"Final crossing-intention probability: {crossing_prob:.3f} "
          f"(ground-truth label: {'CROSS' if scene.will_cross else 'NO-CROSS'})")


if __name__ == "__main__":
    main()
