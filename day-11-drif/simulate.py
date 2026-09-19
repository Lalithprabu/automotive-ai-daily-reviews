"""Render a synced 3-panel top-down GIF: INPUT scene / OLD WAY (fixed-shape
potential field) / DRiF PREDICTION (trained model's learned risk field +
risk-adjusted planned path), across the scripted ego-cyclist crossing
scenario.

A fresh model is trained for `--steps` steps before rendering so the
prediction panel reflects a real, lightly-trained model rather than a
random initialization.

Usage:
    python simulate.py --config config.yaml --steps 400
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from src.models.drif_model import DRiFModel
from src.models.risk_head import sample_risk_at_points
from src.utils.batching import collate_scenes
from src.utils.synthetic_scene import LANE_HALF_WIDTH_M, SceneConfig, SyntheticSceneGenerator

# Palette (matches series-wide visual palette)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def build_model(cfg: dict) -> DRiFModel:
    bev, heads = cfg["bev"], cfg["heads"]
    return DRiFModel(
        in_channels=bev["in_channels"], base_channels=bev["base_channels"],
        bottleneck_channels=bev["bottleneck_channels"], attn_heads=bev["attn_heads"],
        static_map_classes=heads["static_map_classes"], planning_horizon=heads["planning_horizon"],
        planning_dim=heads["planning_dim"],
    )


def build_scene_gen(cfg: dict, total_steps: int) -> SyntheticSceneGenerator:
    bev, heads, scene = cfg["bev"], cfg["heads"], cfg["scene"]
    scfg = SceneConfig(
        grid_size=bev["grid_size"], range_m=bev["range_m"], in_channels=bev["in_channels"],
        num_risk_points=scene["num_risk_points"], num_pairs=scene["num_pairs"],
        planning_horizon=heads["planning_horizon"], total_steps=total_steps, seed=scene["seed"],
    )
    return SyntheticSceneGenerator(scfg)


def train_briefly(model, scene_gen, cfg, steps, rng):
    loss_cfg = cfg["train"]
    lc = cfg["loss"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["simulate"]["lr"], weight_decay=loss_cfg["weight_decay"])
    model.train()
    for step in range(1, steps + 1):
        scenes = scene_gen.generate_batch(cfg["simulate"]["batch_size"], rng)
        batch = collate_scenes(scenes)
        outputs = model(batch["bev_grid"])
        losses = model.compute_losses(
            batch, outputs, w_rank=lc["w_rank"], w_map=lc["w_map"], w_plan=lc["w_plan"],
            w_tv=lc["w_tv"], rank_margin=lc["rank_margin"],
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"[pretrain] step {step:4d}/{steps} | total {losses['total'].item():.4f}")
    model.eval()


def draw_agents(ax, agents, hist, half_range):
    ego, lead, cyc = agents["ego"], agents["lead"], agents["cyclist"]

    def box(ax, center, w, h, color, label):
        rect = Rectangle((center[1] - w / 2, center[0] - h / 2), w, h, angle=0,
                          facecolor=color, edgecolor=INK, linewidth=0.8, alpha=0.9, zorder=5)
        ax.add_patch(rect)
        ax.text(center[1], center[0] + h / 2 + 1.2, label, ha="center", va="bottom",
                 fontsize=7, color=INK_SECONDARY)

    for key, color in (("ego", BLUE), ("lead", ORANGE), ("cyclist", AQUA)):
        h = hist[key]
        ax.plot(h[:, 1], h[:, 0], linestyle=":", color=color, linewidth=1.3, alpha=0.8, zorder=3)

    box(ax, ego, 1.8, 4.2, BLUE, "EGO")
    box(ax, lead, 1.8, 4.2, ORANGE, "LEAD")
    box(ax, cyc, 0.7, 1.6, AQUA, "CYCLIST")

    lane = LANE_HALF_WIDTH_M
    ax.axvline(-lane, color=INK_MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(lane, color=INK_MUTED, linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlim(-half_range, half_range)
    ax.set_ylim(-half_range, half_range)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)


def render_frame(scene, risk_map_pred, planned_path, half_range, t, timesteps):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=SURFACE)

    titles = ["INPUT", "OLD WAY (fixed-shape potential field)", "DRiF PREDICTION"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10, color=INK, fontweight="bold")

    # Panel 1: INPUT scene
    draw_agents(axes[0], scene["agents"], scene["history"], half_range)

    # Panel 2: OLD WAY -- fixed-shape potential field baseline
    axes[1].imshow(
        scene["classical_map"], origin="lower", cmap="inferno",
        extent=[-half_range, half_range, -half_range, half_range], aspect="equal",
    )
    draw_agents(axes[1], scene["agents"], scene["history"], half_range)

    # Panel 3: DRiF PREDICTION -- learned risk field + risk-adjusted path
    axes[2].imshow(
        risk_map_pred, origin="lower", cmap="inferno",
        extent=[-half_range, half_range, -half_range, half_range], aspect="equal",
    )
    axes[2].plot(planned_path[:, 1], planned_path[:, 0], color=YELLOW, linewidth=2.2,
                 marker="o", markersize=3, zorder=6, label="planned path")
    draw_agents(axes[2], scene["agents"], scene["history"], half_range)
    axes[2].legend(loc="lower right", fontsize=6, framealpha=0.7)

    mean_risk = float(risk_map_pred.mean())
    fig.suptitle(
        f"t = {t+1:02d}/{timesteps}   |   mean predicted risk = {mean_risk:.3f}",
        fontsize=10, color=INK_SECONDARY,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = Image.fromarray(buf[..., :3].copy())
    plt.close(fig)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulate"]
    steps = args.steps if args.steps is not None else sim_cfg["steps"]
    timesteps = sim_cfg["timesteps"]

    torch.manual_seed(cfg["scene"]["seed"] + 1)
    rng = np.random.default_rng(cfg["scene"]["seed"] + 1)

    model = build_model(cfg)
    scene_gen = build_scene_gen(cfg, total_steps=timesteps)

    print(f"Pretraining model for {steps} steps before rendering simulation...")
    train_briefly(model, scene_gen, cfg, steps, rng)

    half_range = cfg["bev"]["range_m"] / 2
    frames = []
    eval_rng = np.random.default_rng(cfg["scene"]["seed"] + 999)

    with torch.no_grad():
        for t in range(timesteps):
            scene = scene_gen.generate(t=t, rng=eval_rng)
            batch = collate_scenes([scene])
            outputs = model(batch["bev_grid"])
            risk_map_pred = outputs["risk_map"][0].numpy()
            planned_path = outputs["waypoints"][0].numpy()

            frame = render_frame(scene, risk_map_pred, planned_path, half_range, t, timesteps)
            frames.append(frame)
            print(f"rendered frame {t+1}/{timesteps} | mean risk {risk_map_pred.mean():.4f}")

    out_path = sim_cfg["out_gif"]
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / sim_cfg["fps"]), loop=0,
    )
    print(f"Saved simulation GIF to {out_path}")


if __name__ == "__main__":
    main()
