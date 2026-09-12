"""
Minimal training loop for the Planning Expert.

Runs out of the box against the synthetic dataset (config data.synthetic:
true) so you can verify the pipeline end-to-end before wiring up real
pre-extracted VLM KV caches. Swap `synthetic: false` and implement
TrajectoryDataset._load_sample once your manifest is ready.
"""
import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.trajectory_dataset import TrajectoryDataset, collate_kv_cache_batch
from src.losses.flow_matching import FlowMatchingObjective
from src.models.planning_expert import PlanningExpert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/planning_expert_base.yaml")
    parser.add_argument("--steps", type=int, default=None,
                         help="Override: run only this many optimizer steps (smoke-testing).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PlanningExpert(**cfg["model"]).to(device)
    objective = FlowMatchingObjective(**cfg["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    dataset = TrajectoryDataset(
        model_dim=cfg["model"]["model_dim"],
        cond_dim=cfg["model"]["cond_dim"],
        n_caches=cfg["model"]["n_caches"],
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["data"]["n_kv_heads"],
        l_ctx=cfg["data"]["l_ctx"],
        n_waypoints=cfg["model"]["n_waypoints"],
        synthetic=cfg["data"]["synthetic"],
    )
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,
        collate_fn=collate_kv_cache_batch,
    )

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for batch in loader:
            kv_cache = batch["kv_cache"]
            kv_cache.keys = [k.to(device) for k in kv_cache.keys]
            kv_cache.values = [v.to(device) for v in kv_cache.values]

            x1_gt = batch["trajectory"].to(device)
            instruction_embed = batch["instruction_embed"].to(device)
            ego_state_embed = batch["ego_state_embed"].to(device)

            loss = objective(model, x1_gt, kv_cache, instruction_embed, ego_state_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 10 == 0 or (args.steps and step >= args.steps):
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            if args.steps and step >= args.steps:
                print("reached --steps limit, stopping (smoke test mode)")
                return


if __name__ == "__main__":
    main()
