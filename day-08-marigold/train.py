"""Synthetic smoke-test training loop for the Marigold reconstruction.
Mirrors the paper's actual training regime in spirit: train ONLY on
synthetic image/depth pairs (here, random tensors stand in for rendered
datasets like Hypersim + Virtual KITTI), predicting the added noise at a
random diffusion timestep. Run: python train.py --steps 20
"""
import argparse
import torch
from src.models.marigold import MarigoldConfig, MarigoldDepthModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    cfg = MarigoldConfig(base_channels=16, time_embed_dim=32, num_inference_steps=4)
    model = MarigoldDepthModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters (smoke-scale reference config): {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, H, W = 4, 32, 32

    for step in range(args.steps):
        # Synthetic stand-in for a rendered (image, depth) pair, as Marigold
        # trains exclusively on synthetic data with GT depth latents.
        image = torch.rand(B, 3, H, W) * 2 - 1
        depth_latent_gt = torch.randn(B, cfg.latent_channels, H // 4, W // 4)

        loss = model.forward_train_step(image, depth_latent_gt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {step:02d}  loss {loss.item():.4f}")

    print("\n--- inference smoke test (no ground truth, sampling from noise) ---")
    model.eval()
    test_image = torch.rand(1, 3, H, W) * 2 - 1
    depth = model.predict_depth(test_image, num_ensemble=5)
    print(f"Ensembled affine-invariant depth map shape: {tuple(depth.shape)}")
    print("reached --steps limit, stopping (smoke test mode)")


if __name__ == "__main__":
    main()
