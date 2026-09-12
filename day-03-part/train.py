"""
Smoke-test training loop for PART.

Usage:
    python train.py --steps 20
"""
import argparse

import torch

from src.models.part_model import (
    PARTConfig, PARTModel, UncertaintyAwareSupervision, part_loss,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    B, P = 4, 512
    cfg = PARTConfig(model_dim=64, point_feat_dim=32, n_heads=4, n_layers=3, n_queries=16)
    model = PARTModel(cfg)
    uas = UncertaintyAwareSupervision()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(args.steps):
        radar_points = torch.randn(B, P, 5)
        radar_points[..., 4] = radar_points[..., 4].abs()  # RCS-like channel non-negative
        point_mask = torch.ones(B, P, dtype=torch.bool)
        point_mask[:, -50:] = False  # simulate padding

        out = model(radar_points, point_mask)

        matched_gt_mask = torch.rand(B, cfg.n_queries) > 0.7
        existence_targets = uas(matched_gt_mask)
        surface_point_targets = torch.randn(B, cfg.n_queries, 3)
        velocity_targets = torch.randn(B, cfg.n_queries, 2)

        loss, logs = part_loss(out, existence_targets, surface_point_targets,
                                velocity_targets, matched_gt_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step:03d}  " + "  ".join(f"{k}={v:.4f}" for k, v in logs.items()))

    print(f"reached --steps limit ({args.steps}), stopping (smoke test mode)")


if __name__ == "__main__":
    main()
