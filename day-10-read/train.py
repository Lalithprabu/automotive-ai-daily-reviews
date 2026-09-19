"""Training script for the READ risk field reconstruction.

Trains a READModel on synthetic scenes with the risk_ranking_loss
(agent-proximal probes vs. background probes + background regularizer),
logging loss at a fixed interval. Real gradient steps, real synthetic
data -- no numbers here are precomputed or copied from elsewhere.
"""

import argparse

import numpy as np
import torch
import yaml

from src.models.read_model import READModel, risk_ranking_loss
from src.utils.synthetic_scene import (
    generate_synthetic_scene,
    sample_agent_proximal_probes,
    sample_background_probes,
    scene_to_tensors,
)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict) -> READModel:
    m = cfg["model"]
    return READModel(
        bev_channels=m["bev_channels"],
        agent_features=m["agent_features"],
        hidden_dim=m["hidden_dim"],
        pooled_size=m["pooled_size"],
        num_scene_layers=m["num_scene_layers"],
        num_scene_heads=m["num_scene_heads"],
        num_freqs=m["num_freqs"],
        num_cross_layers=m["num_cross_layers"],
        num_risk_heads=m["num_risk_heads"],
    )


def train(config_path: str):
    cfg = load_config(config_path)
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    model = build_model(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scene_cfg = cfg["scene"]
    loss_cfg = cfg["loss"]

    num_steps = cfg["training"]["num_steps"]
    log_every = cfg["training"]["log_every"]

    losses = []
    for step in range(1, num_steps + 1):
        # Fresh synthetic scene + random timestep each step -> the model
        # sees varied agent configurations rather than overfitting one.
        seed = cfg["training"]["seed"] * 10_000 + step
        scene = generate_synthetic_scene(
            num_timesteps=scene_cfg["num_timesteps"], dt=scene_cfg["dt"], seed=seed
        )
        timestep = np.random.randint(0, scene_cfg["num_timesteps"])
        bev_grid, agent_history = scene_to_tensors(scene, timestep)

        agent_probes = sample_agent_proximal_probes(
            scene, timestep, num_probes=loss_cfg["num_agent_probes"], seed=seed
        )
        bg_probes = sample_background_probes(
            num_probes=loss_cfg["num_background_probes"],
            t_max=scene_cfg["num_timesteps"] * scene_cfg["dt"],
            seed=seed,
        )
        agent_probe_xyt = torch.from_numpy(agent_probes).unsqueeze(0).float()
        bg_probe_xyt = torch.from_numpy(bg_probes).unsqueeze(0).float()

        optimizer.zero_grad()
        loss = risk_ranking_loss(
            model, bev_grid, agent_history, agent_probe_xyt, bg_probe_xyt,
            margin=loss_cfg["margin"], background_weight=loss_cfg["background_weight"],
        )
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step == 1 or step % log_every == 0 or step == num_steps:
            print(f"step {step:4d}/{num_steps}  risk_ranking_loss = {loss.item():.4f}")

    print(f"\nTraining complete. First loss = {losses[0]:.4f}  ->  Last loss = {losses[-1]:.4f}")

    ckpt_path = "read_model_trained.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
