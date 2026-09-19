"""Train DRiFModel on synthetic ego/lead-vehicle/cyclist scenes.

Usage:
    python train.py --config config.yaml --steps 250
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from src.models.drif_model import DRiFModel
from src.utils.batching import collate_scenes
from src.utils.synthetic_scene import SceneConfig, SyntheticSceneGenerator


def build_model(cfg: dict) -> DRiFModel:
    bev = cfg["bev"]
    heads = cfg["heads"]
    return DRiFModel(
        in_channels=bev["in_channels"],
        base_channels=bev["base_channels"],
        bottleneck_channels=bev["bottleneck_channels"],
        attn_heads=bev["attn_heads"],
        static_map_classes=heads["static_map_classes"],
        planning_horizon=heads["planning_horizon"],
        planning_dim=heads["planning_dim"],
    )


def build_scene_gen(cfg: dict) -> SyntheticSceneGenerator:
    bev = cfg["bev"]
    heads = cfg["heads"]
    scene = cfg["scene"]
    scfg = SceneConfig(
        grid_size=bev["grid_size"],
        range_m=bev["range_m"],
        in_channels=bev["in_channels"],
        num_risk_points=scene["num_risk_points"],
        num_pairs=scene["num_pairs"],
        planning_horizon=heads["planning_horizon"],
        seed=scene["seed"],
    )
    return SyntheticSceneGenerator(scfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    steps = args.steps if args.steps is not None else cfg["train"]["steps"]
    batch_size = cfg["train"]["batch_size"]
    lr = cfg["train"]["lr"]
    weight_decay = cfg["train"]["weight_decay"]
    log_every = cfg["train"]["log_every"]
    loss_cfg = cfg["loss"]

    torch.manual_seed(cfg["scene"]["seed"])
    rng = np.random.default_rng(cfg["scene"]["seed"])

    model = build_model(cfg)
    scene_gen = build_scene_gen(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    history = []
    for step in range(1, steps + 1):
        scenes = scene_gen.generate_batch(batch_size, rng)
        batch = collate_scenes(scenes)

        outputs = model(batch["bev_grid"])
        losses = model.compute_losses(
            batch,
            outputs,
            w_rank=loss_cfg["w_rank"],
            w_map=loss_cfg["w_map"],
            w_plan=loss_cfg["w_plan"],
            w_tv=loss_cfg["w_tv"],
            rank_margin=loss_cfg["rank_margin"],
        )

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        history.append({k: v.item() for k, v in losses.items()})

        if step == 1 or step % log_every == 0 or step == steps:
            print(
                f"step {step:4d}/{steps} | total {losses['total'].item():.4f} "
                f"| rank {losses['rank'].item():.4f} | map {losses['map'].item():.4f} "
                f"| plan {losses['plan'].item():.4f} | tv {losses['tv'].item():.4f}"
            )

    first_total = history[0]["total"]
    last_total = history[-1]["total"]
    print(f"\nTotal loss: {first_total:.4f} -> {last_total:.4f}")

    torch.save(model.state_dict(), "drif_model.pt")
    print("Saved model weights to drif_model.pt")


if __name__ == "__main__":
    main()
