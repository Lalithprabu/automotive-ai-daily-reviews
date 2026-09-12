"""
Minimal inference script: loads a checkpoint (or a fresh model, for
smoke-testing) and runs the 10-step Euler sampler to produce a trajectory
from a single (synthetic, by default) example.
"""
import argparse

import torch
import yaml

from src.data.trajectory_dataset import TrajectoryDataset, collate_kv_cache_batch
from src.models.planning_expert import PlanningExpert
from src.sampling.euler_sampler import euler_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/planning_expert_base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlanningExpert(**cfg["model"]).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = TrajectoryDataset(
        model_dim=cfg["model"]["model_dim"],
        cond_dim=cfg["model"]["cond_dim"],
        n_caches=cfg["model"]["n_caches"],
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["data"]["n_kv_heads"],
        l_ctx=cfg["data"]["l_ctx"],
        n_waypoints=cfg["model"]["n_waypoints"],
        synthetic=cfg["data"]["synthetic"],
        synthetic_length=1,
    )
    batch = collate_kv_cache_batch([dataset[0]])

    kv_cache = batch["kv_cache"]
    kv_cache.keys = [k.to(device) for k in kv_cache.keys]
    kv_cache.values = [v.to(device) for v in kv_cache.values]
    instruction_embed = batch["instruction_embed"].to(device)
    ego_state_embed = batch["ego_state_embed"].to(device)

    trajectory = euler_sample(
        model, kv_cache, instruction_embed, ego_state_embed,
        n_waypoints=cfg["model"]["n_waypoints"], n_steps=cfg["sampling"]["n_steps"],
        device=device,
    )
    print("predicted trajectory shape:", trajectory.shape)
    print(trajectory[0])


if __name__ == "__main__":
    main()
