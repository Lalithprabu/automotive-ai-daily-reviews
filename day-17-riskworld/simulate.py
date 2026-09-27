"""Renders an animated GIF from a REAL trained RiskWorld checkpoint:
  - BEV scene input
  - Ground-truth occupancy
  - Persistence-baseline occupancy (naive: last observed frame held static)
  - RiskWorld's flow-warped + residual-corrected forecast occupancy
  - The nominal ego trajectory overlaid, and the replacement candidate if the
    selective-replacement gate fires
  - A live telemetry sub-panel: running risk score / collision-score
    correction accumulated as the horizon unfolds.

Usage:
    python simulate.py --checkpoint checkpoints/riskworld.pt --out outputs/riskworld_simulation.gif
"""
from __future__ import annotations

import argparse
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.collision_score import CollisionScoreModule
from src.dataset import SceneConfig, generate_scene
from src.planning import plan_step
from src.riskworld_model import RiskWorld
from src.utils import load_config


def find_demo_scene(scene_cfg: SceneConfig, model: RiskWorld, collision_scorer: CollisionScoreModule,
                     planning_cfg: dict, max_tries: int = 300, base_seed: int = 9000):
    """Searches a handful of seeds for a hazard scene where the selective
    replacement gate actually fires, so the rendered GIF demonstrates the
    full mechanism. Falls back to the first hazard scene found if none
    trigger replacement within max_tries."""
    fallback = None
    for i in range(max_tries):
        rng = np.random.default_rng(base_seed + i)
        scene = generate_scene(scene_cfg, rng)
        if not bool(scene["has_hazard"]):
            continue
        batch = {k: torch.from_numpy(v).unsqueeze(0).float() if v.dtype != np.bool_ else torch.from_numpy(v).unsqueeze(0)
                  for k, v in scene.items()}
        with torch.no_grad():
            out = model(
                bev_grid=batch["bev_grid"], agent_history=batch["agent_history"],
                agent_last_pos=batch["agent_last_pos"], agent_mask=batch["agent_mask"],
                prev_occupancy=batch["prev_occupancy"],
            )
            plan = plan_step(
                out, batch["ego_nominal_traj"], collision_scorer,
                planning_cfg["num_candidates"], planning_cfg["tau_risk"], planning_cfg["tau_dev"],
            )
        if fallback is None:
            fallback = (scene, batch, out, plan)
        if bool(plan["selection"].replaced[0]):
            return scene, batch, out, plan
    return fallback


def render_gif(cfg: dict, checkpoint_path: str, out_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = RiskWorld(ckpt["cfg"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data_cfg = cfg["data"]
    scene_cfg = SceneConfig(
        grid_size=data_cfg["grid_size"], history_len=data_cfg["history_len"], horizon=data_cfg["horizon"],
        max_agents=data_cfg["max_agents"], cross_traffic_prob=1.0,  # force hazard scenes for the demo search
    )
    collision_scorer = CollisionScoreModule(grid_size=data_cfg["grid_size"])

    scene, batch, out, plan = find_demo_scene(scene_cfg, model, collision_scorer, cfg["planning"])
    horizon = data_cfg["horizon"]
    G = data_cfg["grid_size"]

    nominal_traj = batch["ego_nominal_traj"][0].numpy()          # (horizon, 2)
    replaced = bool(plan["selection"].replaced[0])
    sel_idx = int(plan["selection"].selected_index[0])
    selected_traj = plan["candidates"][0, sel_idx].numpy() if replaced else None

    forecast_occ = out["forecast_occupancy"][0, :, 0].numpy()      # (horizon, G, G)
    persistence_occ = out["persistence_occupancy"][0, :, 0].numpy()  # (horizon, G, G)
    gt_occ = scene["future_occupancy_gt"][:, 0]                       # (horizon, G, G)
    bev_grid = scene["bev_grid"]                                        # (4, G, G)
    risk_field = out["risk_field"][0, 0].numpy()                          # (G, G)

    # Per-step telemetry: sample forecast/persistence/risk at the nominal
    # trajectory's waypoint for each step, to show a running risk curve.
    running_forecast, running_persistence, running_correction = [], [], []
    with torch.no_grad():
        for t in range(horizon):
            pos_t = batch["ego_nominal_traj"][:, t:t + 1, :]  # (1, 1, 2)
            occ_f = collision_scorer.sample_at_positions(out["forecast_occupancy"][:, t], pos_t)[0, 0].item()
            occ_p = collision_scorer.sample_at_positions(out["persistence_occupancy"][:, t], pos_t)[0, 0].item()
            risk_v = collision_scorer.sample_at_positions(out["risk_field"], pos_t)[0, 0].item()
            f_score = occ_f + risk_v
            p_score = occ_p + risk_v
            running_forecast.append(f_score)
            running_persistence.append(p_score)
            running_correction.append(max(0.0, f_score - p_score))

    frames = []
    for t in range(horizon):
        fig, axes = plt.subplots(2, 4, figsize=(14, 7), gridspec_kw={"height_ratios": [3, 1.4]})
        titles = ["BEV scene (road/lane/obstacle/agents)", "Ground-truth occupancy",
                  "Persistence baseline (naive, static)", "RiskWorld forecast (flow+residual)"]
        panels = [
            np.clip(bev_grid[0] * 0.3 + bev_grid[1] * 0.5 + bev_grid[2] * 0.8 + bev_grid[3], 0, 1),
            gt_occ[t], persistence_occ[t], forecast_occ[t],
        ]
        for j, (ax, panel, title) in enumerate(zip(axes[0], panels, titles)):
            ax.imshow(panel, cmap="inferno", vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            # Trajectory overlay, drawn up to the current frame's step.
            ax.plot(nominal_traj[: t + 1, 0], nominal_traj[: t + 1, 1], "o--",
                     color="deepskyblue" if not (replaced and j == 3) else "lightgray",
                     linewidth=2, markersize=4, label="nominal")
            if replaced and j == 3:
                ax.plot(selected_traj[: t + 1, 0], selected_traj[: t + 1, 1], "o-",
                         color="lime", linewidth=2, markersize=4, label="replacement (selected)")
            if j == 0:
                ax.legend(fontsize=6, loc="upper left")
            ax.set_xlim(0, G - 1); ax.set_ylim(G - 1, 0)

        # Telemetry sub-panel spans the bottom row.
        for ax in axes[1]:
            ax.remove()
        tele_ax = fig.add_subplot(2, 1, 2)
        steps = list(range(1, horizon + 1))
        tele_ax.plot(steps[: t + 1], running_forecast[: t + 1], "o-", color="crimson", label="forecast risk score")
        tele_ax.plot(steps[: t + 1], running_persistence[: t + 1], "o--", color="gray", label="persistence risk score")
        tele_ax.bar(steps[: t + 1], running_correction[: t + 1], alpha=0.3, color="orange",
                     label="collision-score correction (>=0)")
        tele_ax.axhline(cfg["planning"]["tau_risk"], color="black", linestyle=":", linewidth=1, label="tau_risk")
        tele_ax.set_xlim(0.5, horizon + 0.5)
        tele_ax.set_ylim(0, max(1.5, max(running_forecast + running_persistence) * 1.2))
        tele_ax.set_xlabel("planning step")
        tele_ax.set_ylabel("risk score (sampled along nominal path)")
        tele_ax.legend(fontsize=7, loc="upper left")
        status = "REPLACED (risk exceeded tau_risk & a safe alternative qualified)" if replaced else \
            "nominal kept (risk below tau_risk, or no qualifying alternative)"
        fig.suptitle(f"RiskWorld reconstruction -- planning step {t + 1}/{horizon}  |  decision: {status}",
                      fontsize=10)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=900, loop=0)
    print(f"saved {len(frames)}-frame GIF to {out_path}")
    return out_path, len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = args.checkpoint or cfg["paths"]["checkpoint"]
    out_path = args.out or cfg["paths"]["gif_out"]
    render_gif(cfg, checkpoint_path, out_path)


if __name__ == "__main__":
    main()
