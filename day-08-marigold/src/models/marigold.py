"""
Marigold — Repurposing a pretrained Latent-Diffusion image generator (Stable
Diffusion) as a monocular depth estimator, by (1) freezing/reusing its VAE
image latent space, (2) fine-tuning the denoising U-Net to predict a DEPTH
latent conditioned on an IMAGE latent via channel-concatenation, and
(3) sampling with a "trailing" DDIM schedule so 1-4 denoising steps suffice
instead of the usual 10-50.

Faithful, reduced-scale PyTorch reconstruction of the published mechanism
(Ke et al., CVPR 2024 arXiv:2312.02145; T-PAMI 2025 arXiv:2505.09358) — NOT
the authors' original ~866M-parameter Stable-Diffusion-based checkpoint.
This file reproduces the *architectural idea* (latent concatenation
conditioning + few-step trailing-DDIM depth denoising + test-time ensembling)
at a size that trains/tests in seconds on CPU, for the GitHub series.

Key facts this file is grounded in:
- Depth is predicted as an AFFINE-INVARIANT (scale+shift ambiguous) latent,
  not metric depth. Downstream metric alignment is a separate step.
- Conditioning mechanism: concatenate the (frozen) image latent with the
  noisy depth latent along the channel dimension before the U-Net -- no
  cross-attention/ControlNet branch is needed.
- Multiple independent noise seeds can be denoised and their depth maps
  averaged/median-aligned into one ensembled prediction for lower variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MarigoldConfig:
    latent_channels: int = 4          # VAE latent channel count (SD-style: 4)
    base_channels: int = 64           # U-Net stem width (reduced from SD's 320)
    time_embed_dim: int = 128
    num_train_timesteps: int = 1000   # full DDPM noise schedule length
    num_inference_steps: int = 4      # Marigold's few-step "trailing" regime


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding for the diffusion
    timestep t, projected through a small MLP (as in SD's U-Net)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] integer/float timestep indices -> [B, dim*4] embedding
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)      # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        return self.mlp(emb)                                     # [B, dim*4]


class ResBlock(nn.Module):
    """A timestep-conditioned residual conv block (the U-Net's basic unit)."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x:      [B, in_ch,  H, W]
        # t_emb:  [B, time_dim]
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))       # [B, out_ch, H, W]
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)      # broadcast add
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))       # [B, out_ch, H, W]
        return h + self.skip(x)                                        # [B, out_ch, H, W]


class MarigoldUNet(nn.Module):
    """Reduced-scale stand-in for Marigold's fine-tuned SD denoising U-Net.

    Core innovation lives entirely in `forward`'s first line: the noisy
    DEPTH latent and the (clean, frozen) IMAGE latent are concatenated on
    the channel axis. This lets the U-Net attend to full RGB scene context
    at every denoising step without any new cross-attention weights --
    the pretrained U-Net's own self-attention layers do the rest, which is
    why only the first conv layer needs its input channels changed when
    repurposing a frozen text-to-image model.
    """

    def __init__(self, cfg: MarigoldConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)
        time_dim = cfg.time_embed_dim * 4

        # Concatenated input: [noisy_depth_latent ; image_latent] on channel dim.
        in_ch = cfg.latent_channels * 2
        c = cfg.base_channels

        self.stem = nn.Conv2d(in_ch, c, 3, padding=1)
        self.down1 = ResBlock(c, c, time_dim)
        self.down2 = ResBlock(c, c * 2, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid = ResBlock(c * 2, c * 2, time_dim)
        self.up2 = ResBlock(c * 2, c, time_dim)
        self.up1 = ResBlock(c, c, time_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.out_norm = nn.GroupNorm(8, c)
        self.out_conv = nn.Conv2d(c, cfg.latent_channels, 3, padding=1)

    def forward(
        self,
        noisy_depth_latent: torch.Tensor,  # [B, 4, H, W]
        image_latent: torch.Tensor,        # [B, 4, H, W]  (frozen VAE encoding)
        t: torch.Tensor,                   # [B]           diffusion timestep
    ) -> torch.Tensor:
        x = torch.cat([noisy_depth_latent, image_latent], dim=1)  # [B, 8, H, W]  <- the trick
        t_emb = self.time_embed(t)                                # [B, time_dim]

        h = self.stem(x)                                          # [B, c,   H,   W]
        h1 = self.down1(h, t_emb)                                 # [B, c,   H,   W]
        h2 = self.down2(self.pool(h1), t_emb)                     # [B, 2c,  H/2, W/2]
        m = self.mid(h2, t_emb)                                   # [B, 2c,  H/2, W/2]
        u2 = self.up2(m, t_emb)                                   # [B, c,   H/2, W/2]
        u1 = self.up1(self.upsample(u2) + h1, t_emb)              # [B, c,   H,   W]
        eps_pred = self.out_conv(torch.nn.functional.silu(self.out_norm(u1)))
        return eps_pred                                           # [B, 4,   H,   W]  predicted noise


class TrailingDDIMScheduler:
    """The mechanism that lets Marigold drop from 10-50 DDIM steps to 1-4:
    instead of spacing inference timesteps evenly from 0, the *last*
    training timestep (num_train_timesteps - 1, the noisiest) is always
    included and steps trail backward from there. Concretely this means
    the model spends its few available steps exactly where the signal is
    hardest to recover, rather than wasting steps near t=0 where the
    depth latent is nearly clean anyway.
    """

    def __init__(self, cfg: MarigoldConfig):
        self.cfg = cfg
        betas = torch.linspace(1e-4, 2e-2, cfg.num_train_timesteps)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0)  # [T] cumulative product

    def trailing_timesteps(self) -> torch.Tensor:
        T = self.cfg.num_train_timesteps
        n = self.cfg.num_inference_steps
        # e.g. T=1000, n=4 -> [999, 749, 499, 249] (trailing from the last step)
        step = T // n
        return torch.arange(T - 1, -1, -step)[:n]

    @torch.no_grad()
    def denoise(self, unet: MarigoldUNet, image_latent: torch.Tensor) -> torch.Tensor:
        """Runs the full few-step reverse process starting from pure noise.
        image_latent: [B, 4, H, W] -> returns depth_latent: [B, 4, H, W]
        """
        B = image_latent.shape[0]
        device = image_latent.device
        alpha_bar = self.alpha_bar.to(device)

        depth_latent = torch.randn_like(image_latent)             # x_T ~ N(0, I)
        timesteps = self.trailing_timesteps().to(device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            eps_pred = unet(depth_latent, image_latent, t_batch)  # [B, 4, H, W]

            a_t = alpha_bar[t].clamp(min=1e-5)
            x0_pred = (depth_latent - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)                     # normalized depth latent

            if i + 1 < len(timesteps):
                t_next = timesteps[i + 1]
                a_next = alpha_bar[t_next]
                depth_latent = torch.sqrt(a_next) * x0_pred + torch.sqrt(1 - a_next) * eps_pred
            else:
                depth_latent = x0_pred
        return depth_latent                                       # [B, 4, H, W]

    @torch.no_grad()
    def denoise_with_intermediates(
        self, unet: MarigoldUNet, image_latent: torch.Tensor
    ) -> list[torch.Tensor]:
        """Same reverse process as `denoise`, but returns the depth latent
        after EVERY step (including the initial noise as step 0) so callers
        can animate the trajectory from noise to final prediction.
        Returns a list of length num_inference_steps + 1, each [B, 4, H, W].
        """
        B = image_latent.shape[0]
        device = image_latent.device
        alpha_bar = self.alpha_bar.to(device)

        depth_latent = torch.randn_like(image_latent)
        timesteps = self.trailing_timesteps().to(device)
        trajectory = [depth_latent.clone()]

        for i, t in enumerate(timesteps):
            t_batch = t.expand(B)
            eps_pred = unet(depth_latent, image_latent, t_batch)

            a_t = alpha_bar[t].clamp(min=1e-5)
            x0_pred = (depth_latent - torch.sqrt(1 - a_t) * eps_pred) / torch.sqrt(a_t)
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i + 1 < len(timesteps):
                t_next = timesteps[i + 1]
                a_next = alpha_bar[t_next]
                depth_latent = torch.sqrt(a_next) * x0_pred + torch.sqrt(1 - a_next) * eps_pred
            else:
                depth_latent = x0_pred
            trajectory.append(depth_latent.clone())
        return trajectory                                          # list of [B, 4, H, W]


def ensemble_depth_predictions(latents: list[torch.Tensor]) -> torch.Tensor:
    """Test-time ensembling: run the stochastic sampler N times from
    independent noise seeds and align+average the results. Here (post
    decode, single-channel depth maps assumed) we do a simple affine
    (scale+shift) alignment to the first sample before averaging, since
    each sample is only affine-invariant-consistent with the others.
    preds: list of [B, 1, H, W] depth maps -> [B, 1, H, W] ensembled depth
    """
    ref = latents[0]
    aligned = [ref]
    for pred in latents[1:]:
        # Least-squares affine fit: pred_aligned = a * pred + b  ~=  ref
        pred_flat = pred.flatten(1)
        ref_flat = ref.flatten(1)
        p_mean, r_mean = pred_flat.mean(-1, keepdim=True), ref_flat.mean(-1, keepdim=True)
        cov = ((pred_flat - p_mean) * (ref_flat - r_mean)).mean(-1, keepdim=True)
        var = ((pred_flat - p_mean) ** 2).mean(-1, keepdim=True).clamp(min=1e-6)
        a = cov / var
        b = r_mean - a * p_mean
        aligned.append((a.unsqueeze(-1).unsqueeze(-1) * pred + b.unsqueeze(-1).unsqueeze(-1)))
    return torch.stack(aligned, dim=0).mean(dim=0)                 # [B, 1, H, W]


class MarigoldDepthModel(nn.Module):
    """End-to-end wrapper: frozen-VAE-latent stand-in encoder -> denoising
    U-Net -> frozen-VAE-latent stand-in decoder. The real Marigold reuses
    Stable Diffusion's actual pretrained VAE (frozen); here the encoder/
    decoder are lightweight conv stand-ins so the whole pipeline is
    trainable and testable without shipping an 866M-parameter backbone.
    """

    def __init__(self, cfg: MarigoldConfig | None = None):
        super().__init__()
        self.cfg = cfg or MarigoldConfig()
        c = self.cfg.latent_channels

        # Stand-ins for the frozen SD VAE encoder/decoder (8x spatial downsample in the
        # real model; kept at 4x here to stay lightweight for the smoke tests below).
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, c, 3, stride=2, padding=1),
        )
        self.depth_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c, 32, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 1, 3, padding=1),
        )

        self.unet = MarigoldUNet(self.cfg)
        self.scheduler = TrailingDDIMScheduler(self.cfg)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, 3, H, W] in [-1, 1] -> image_latent: [B, 4, H/4, W/4]
        return self.image_encoder(image)

    def forward_train_step(
        self, image: torch.Tensor, depth_latent_gt: torch.Tensor
    ) -> torch.Tensor:
        """One training step: add noise to the GT depth latent at a random
        timestep, ask the U-Net to predict that noise, return the MSE loss.
        image:          [B, 3, H, W]
        depth_latent_gt:[B, 4, H/4, W/4]  (GT depth VAE-encoded, affine-normalized)
        """
        B = image.shape[0]
        device = image.device
        image_latent = self.encode_image(image)                         # [B, 4, H/4, W/4]

        t = torch.randint(0, self.cfg.num_train_timesteps, (B,), device=device)
        alpha_bar = self.scheduler.alpha_bar.to(device)[t].view(B, 1, 1, 1)
        noise = torch.randn_like(depth_latent_gt)
        noisy_depth_latent = torch.sqrt(alpha_bar) * depth_latent_gt + torch.sqrt(1 - alpha_bar) * noise

        eps_pred = self.unet(noisy_depth_latent, image_latent, t)       # [B, 4, H/4, W/4]
        return torch.nn.functional.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def predict_depth(self, image: torch.Tensor, num_ensemble: int = 1) -> torch.Tensor:
        """Inference: few-step trailing-DDIM denoise -> decode -> (optionally)
        ensemble across independent noise seeds.
        image: [B, 3, H, W] -> depth: [B, 1, H, W] (affine-invariant, NOT metric)
        """
        image_latent = self.encode_image(image)                         # [B, 4, H/4, W/4]
        depth_maps = []
        for _ in range(num_ensemble):
            depth_latent = self.scheduler.denoise(self.unet, image_latent)  # [B, 4, H/4, W/4]
            depth_maps.append(self.depth_decoder(depth_latent))             # [B, 1, H,   W]
        if num_ensemble == 1:
            return depth_maps[0]
        return ensemble_depth_predictions(depth_maps)                    # [B, 1, H, W]

    @torch.no_grad()
    def predict_depth_trajectory(self, image: torch.Tensor) -> list[torch.Tensor]:
        """Real few-step denoising trajectory, decoded to depth-map space at
        every step -- this is what `simulate.py` animates. Not an
        illustration: every frame is this model's own forward pass.
        image: [B, 3, H, W] -> list of [B, 1, H, W], length num_inference_steps + 1
        """
        image_latent = self.encode_image(image)
        latent_trajectory = self.scheduler.denoise_with_intermediates(self.unet, image_latent)
        return [self.depth_decoder(lat) for lat in latent_trajectory]
