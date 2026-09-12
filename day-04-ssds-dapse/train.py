"""
Smoke-test training + DAPSE sampling script for SSDS.

Usage:
    python train.py --steps 20
"""
import argparse

import torch

from src.models.ssds_dapse import (
    SSDSConfig, SSDSDenoiser, dapse_guided_step, comfort_energy, adversarial_proximity_energy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    B, A, H, C = 2, 6, 16, 24
    cfg = SSDSConfig(
        traj_feat_dim=3, ctx_feat_dim=8, model_dim=64, cond_dim=64,
        n_heads=4, n_dual_blocks=2, n_single_blocks=2,
    )
    model = SSDSDenoiser(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ctx_tokens = torch.randn(B, C, cfg.ctx_feat_dim)
    nav_embed = torch.randn(B, cfg.cond_dim)
    x1_gt = torch.randn(B, A, H, 3)

    for step in range(args.steps):
        timesteps = torch.randint(0, 1000, (B,))
        noise = torch.randn_like(x1_gt)
        t_frac = (timesteps.float() / 1000).view(B, 1, 1, 1)
        x_t = (1 - t_frac) * x1_gt + t_frac * noise

        x0_pred = model(x_t, ctx_tokens, timesteps, nav_embed)
        loss = torch.nn.functional.mse_loss(x0_pred, x1_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step:03d}  loss {loss.item():.4f}")

    x_t = torch.randn(B, A, H, 3)
    timesteps = torch.full((B,), 500)

    x0_planner = dapse_guided_step(
        x_t, model, ctx_tokens, timesteps, nav_embed,
        r_t=0.5, energy_fn=comfort_energy, beta=0.1, eta=0.02, n_inner_steps=3,
    )
    print("DAPSE planner-role x0:", tuple(x0_planner.shape))

    x0_scenario = dapse_guided_step(
        x_t, model, ctx_tokens, timesteps, nav_embed,
        r_t=0.5, energy_fn=adversarial_proximity_energy, beta=0.5, eta=0.02, n_inner_steps=3,
    )
    print("DAPSE scenario-generator-role x0:", tuple(x0_scenario.shape))
    print(f"reached --steps limit ({args.steps}), stopping (smoke test mode)")


if __name__ == "__main__":
    main()
