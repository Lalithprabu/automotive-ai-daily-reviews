"""
Smoke-test training loop for COSTER: a real synthetic-data training loop
(CVAE reconstruction + collision-time CE + contact-pose L1 + beta*KL), then
one inference-mode generation pass sampling entirely from the learned
conditional prior (no ground truth).

Usage:
    python train.py --steps 20
"""
import argparse

import torch

from src.models.coster import COSTERConfig, COSTERModel, coster_loss


def synthetic_batch(cfg: COSTERConfig, batch_size: int, n_agents: int, n_polylines: int, n_points: int):
    polylines = torch.randn(batch_size, n_polylines, n_points, cfg.map_point_dim)
    point_mask = torch.ones(batch_size, n_polylines, n_points, dtype=torch.bool)

    agent_histories = torch.randn(batch_size, n_agents, cfg.n_history_steps, cfg.agent_state_dim)
    target_idx = torch.zeros(batch_size, dtype=torch.long)  # always agent 0 for simplicity

    target_future_states = torch.randn(batch_size, cfg.n_collision_time_bins, cfg.state_dim)
    gt_lead_up_traj = torch.randn(batch_size, cfg.t_rev, cfg.state_dim)
    gt_collision_time_bin = torch.randint(0, cfg.n_collision_time_bins, (batch_size,))
    gt_contact_pose_offset = torch.randn(batch_size, 3)

    return (polylines, point_mask, agent_histories, target_idx, target_future_states,
            gt_lead_up_traj, gt_collision_time_bin, gt_contact_pose_offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = COSTERConfig(map_point_dim=7, agent_state_dim=4, hidden_dim=32, latent_dim=8,
                        state_dim=4, n_history_steps=10, t_rev=12, n_collision_time_bins=8, n_heads=4)
    model = COSTERModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step in range(args.steps):
        batch = synthetic_batch(cfg, batch_size=6, n_agents=5, n_polylines=10, n_points=9)
        (polylines, point_mask, agent_histories, target_idx, target_future_states,
         gt_lead_up_traj, gt_collision_time_bin, gt_contact_pose_offset) = batch

        outputs = model(polylines, point_mask, agent_histories, target_idx,
                         target_future_states, gt_lead_up_traj=gt_lead_up_traj)
        loss, logs = coster_loss(outputs, gt_lead_up_traj, gt_collision_time_bin,
                                  gt_contact_pose_offset, beta=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step:03d}  " + "  ".join(f"{k}={v:.4f}" for k, v in logs.items()))

    # ---- inference-mode generation: no ground truth, samples from the prior ----
    model.eval()
    with torch.no_grad():
        batch = synthetic_batch(cfg, batch_size=3, n_agents=5, n_polylines=10, n_points=9)
        polylines, point_mask, agent_histories, target_idx, target_future_states = batch[:5]
        outputs = model(polylines, point_mask, agent_histories, target_idx, target_future_states)
    print("inference rollout shape:", tuple(outputs["rollout"].shape))
    print(f"reached --steps limit ({args.steps}), stopping (smoke test mode)")


if __name__ == "__main__":
    main()
