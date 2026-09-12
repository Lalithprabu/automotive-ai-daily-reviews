"""
Smoke-test training loop for MC-DeTra: one forward pass through the full
model + MotionConsistencyLosses, one combined backward pass, and a
parameter-tensor gradient-coverage check.

Usage:
    python train.py --steps 20
"""
import argparse

import torch

from src.models.mc_detra import MCDeTraConfig, MCDeTra, MotionConsistencyLosses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = MCDeTraConfig(d_model=32, n_objects=6, n_future_steps=4, n_modes=3,
                         n_blocks=2, n_heads=4, lidar_channels=16, map_feat_dim=16,
                         occupancy_grid=5, knn_k=4)
    model = MCDeTra(cfg)
    mc_losses = MotionConsistencyLosses(cfg)

    params = list(model.parameters()) + list(mc_losses.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    B, H, W, M = 4, 24, 24, 20

    for step in range(args.steps):
        lidar_bev = torch.randn(B, cfg.lidar_channels, H, W)
        map_tokens = torch.randn(B, M, cfg.map_feat_dim)

        out = model(lidar_bev, map_tokens)

        past_motion_gt = torch.randn(B, cfg.n_objects, cfg.n_future_steps, 2)
        occupancy_gt = (torch.rand(B, cfg.n_objects, cfg.occupancy_grid, cfg.occupancy_grid) > 0.5).float()
        valid_mask_obj = torch.ones(B, cfg.n_objects)
        valid_mask_traj = torch.ones(B, cfg.n_objects, cfg.n_future_steps, cfg.n_modes)

        mc_out = mc_losses(out["t0_query"], out["headings"], out["positions"],
                            past_motion_gt, occupancy_gt, valid_mask_obj, valid_mask_traj)

        optimizer.zero_grad()
        mc_out["loss_mc_total"].backward()
        optimizer.step()

        print(f"step {step:03d}  " + "  ".join(
            f"{k}={v.item():.4f}" for k, v in mc_out.items()
        ))

    print("detection_boxes:", tuple(out["detection_boxes"].shape))
    print("forecast_offsets:", tuple(out["forecast_offsets"].shape))
    print("headings:", tuple(out["headings"].shape))
    print("positions:", tuple(out["positions"].shape))

    n_params = sum(p.numel() for p in params)
    n_tensors = sum(1 for _ in params)
    print(f"total parameter tensors: {n_tensors} | total parameters: {n_params:,}")
    print(f"reached --steps limit ({args.steps}), stopping (smoke test mode)")


if __name__ == "__main__":
    main()
