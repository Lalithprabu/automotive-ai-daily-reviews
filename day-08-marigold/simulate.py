"""
Animated visualization of Marigold's actual few-step "trailing DDIM"
denoising process, run on THIS repo's own reconstructed model (random or
loaded weights) -- every frame is a real forward pass through
`MarigoldDepthModel.predict_depth_trajectory`, not a hand-drawn
illustration. Saved as an animated GIF -- GitHub renders GIFs inline in
READMEs.

Run with: python simulate.py --config tiny_marigold_cfg.yaml
"""
import argparse

import torch
import yaml

from src.models.marigold import MarigoldConfig, MarigoldDepthModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tiny_marigold_cfg.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="trajectory_simulation.gif")
    parser.add_argument("--fps", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    torch.manual_seed(0)
    device = torch.device("cpu")

    m_cfg = MarigoldConfig(**cfg["model"])
    model = MarigoldDepthModel(m_cfg).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    H, W = cfg["sim"]["image_h"], cfg["sim"]["image_w"]
    # Synthetic stand-in for a real camera frame (a soft radial gradient so
    # there is at least some spatial structure for the conv stem to see).
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    radial = torch.sqrt(xx**2 + yy**2)
    image = torch.stack([1 - radial, 0.5 - 0.3 * radial, radial], dim=0).unsqueeze(0)
    image = image.clamp(-1, 1)  # [1, 3, H, W]

    trajectory = model.predict_depth_trajectory(image)  # list of [1,1,H,W], real few-step denoise
    n_steps = len(trajectory)
    weights_note = "trained checkpoint" if args.checkpoint else "untrained/random weights"

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4), dpi=120)
    fig.patch.set_facecolor("#fcfcfb")

    img_np = ((image[0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
    axes[0].imshow(img_np)
    axes[0].set_title("Input image", fontsize=10)
    axes[0].axis("off")

    depth_im = axes[1].imshow(trajectory[0][0, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
    axes[1].axis("off")
    title = axes[1].set_title("", fontsize=10)
    fig.suptitle(f"Marigold few-step trailing-DDIM denoising ({weights_note})", fontsize=10.5)

    def update(frame_idx):
        depth_im.set_data(trajectory[frame_idx][0, 0].numpy())
        label = "step 0 (pure noise)" if frame_idx == 0 else f"step {frame_idx} / {n_steps - 1}"
        title.set_text(f"Depth latent decode -- {label}")
        return [depth_im, title]

    anim = FuncAnimation(fig, update, frames=n_steps, interval=1000 / args.fps, blit=False)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved simulation GIF to {args.out} ({n_steps} steps, {weights_note})")


if __name__ == "__main__":
    main()
