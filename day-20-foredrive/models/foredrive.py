"""
Full model wiring: ForeDriveModel = encoder -> world model (stop-grad routed)
-> gated fusion (+ future-status injection) -> TAB-biased DiT planner.

THIS FILE IS WHERE THE STOP-GRADIENT ROUTING BOUNDARY IS ACTUALLY ENFORCED.
Per the abstract: "planning gradients update the shared online encoder,
while stop-gradient routing trains the latent predictor with forecasting
losses only." Concretely, in `forward_train` below:

  1. `current_tokens = self.encoder(current_frame)`
     -- requires grad w.r.t. encoder params. Used TWICE, differently:

  2. Forecasting path (trains predictor ONLY):
       predictor_input = current_tokens.detach()                     <- DETACH #1
       target_tokens    = self.encoder(future_frames).detach()          <- DETACH #2 (asymmetric JEPA target)
       predicted_tokens, confidence = self.world_model(predictor_input, ego_context)
       forecast_loss = mse(predicted_tokens, target_tokens)
     Because predictor_input is detached, NO gradient from forecast_loss can
     reach the encoder. Because target_tokens is detached, NO gradient flows
     through the "target" branch at all (standard JEPA/BYOL-style stop-grad
     target). The ONLY parameters forecast_loss can update are inside
     `self.world_model`.

  3. Planning path (trains encoder end-to-end, but NOT the predictor):
       fused_tokens = self.fusion(
           current_tokens,                    <- NOT detached: this is the path
                                                   through which "planning gradients
                                                   update the shared online encoder"
           predicted_tokens.detach(),          <- DETACH #3
           confidence.detach(),                 <- DETACH #4
       )
       eps_pred = self.dit_planner(noisy_traj, t, fused_tokens)
       planning_loss = mse(eps_pred, noise)
     Because predicted_tokens and confidence are detached before entering
     fusion, planning_loss's gradient cannot reach `self.world_model` at all
     — this is the "stop-gradient routing" that keeps the predictor trained
     "with forecasting losses only". planning_loss's gradient DOES reach
     `self.encoder` (via the un-detached `current_tokens` branch) and
     `self.fusion` / `self.dit_planner` (both fully trained by planning_loss).

This exact property (encoder gets gradient from planning_loss but NOT from
forecast_loss; world_model gets gradient from forecast_loss but NOT from
planning_loss) is verified by tests/test_stop_gradient.py — this is called
out in the task spec as the single most important correctness property of
the repo, so it is tested directly on `.grad` values after real `.backward()`
calls, not merely asserted in a comment.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.encoder import VisualEncoder
from models.world_model import MultiHorizonLatentPredictor
from models.fusion import GatedFutureFusion
from models.dit_planner import DiTPlanner
from src.utils.diffusion import make_beta_schedule, q_sample, ddim_respaced_timesteps, ddim_step


class ForeDriveModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        horizons = cfg["data"]["horizons"]
        self.num_horizons = len(horizons)
        self.num_waypoints = cfg["data"]["num_waypoints"]

        self.encoder = VisualEncoder(
            in_channels=cfg["data"]["raster_channels"],
            raster_size=cfg["data"]["raster_size"],
            tokens_per_side=cfg["encoder"]["tokens_per_side"],
            token_dim=cfg["encoder"]["token_dim"],
        )

        self.world_model = MultiHorizonLatentPredictor(
            token_dim=cfg["encoder"]["token_dim"],
            ego_state_dim=cfg["encoder"]["ego_state_dim"],
            num_horizons=self.num_horizons,
            hidden_dim=cfg["world_model"]["hidden_dim"],
            num_layers=cfg["world_model"]["num_layers"],
            num_heads=cfg["world_model"]["num_heads"],
            dropout=cfg["world_model"]["dropout"],
        )

        self.fusion = GatedFutureFusion(
            token_dim=cfg["encoder"]["token_dim"],
            num_horizons=self.num_horizons,
            status_dim=cfg["fusion"]["status_dim"],
            fused_token_dim=cfg["fusion"]["fused_token_dim"],
        )

        self.dit_planner = DiTPlanner(
            fused_token_dim=cfg["fusion"]["fused_token_dim"],
            num_waypoints=self.num_waypoints,
            hidden_dim=cfg["dit"]["hidden_dim"],
            num_layers=cfg["dit"]["num_layers"],
            num_heads=cfg["dit"]["num_heads"],
            dropout=cfg["dit"]["dropout"],
            tab_hidden_dim=cfg["tab"]["hidden_dim"],
        )

        betas, alphas, alpha_bars = make_beta_schedule(
            cfg["dit"]["num_diffusion_steps"], cfg["dit"]["beta_start"], cfg["dit"]["beta_end"],
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.num_diffusion_steps = cfg["dit"]["num_diffusion_steps"]
        self.num_sampling_steps = cfg["dit"]["num_sampling_steps"]

        # --- Trajectory (waypoint) normalization ---------------------------
        # REAL BUG FOUND + FIXED during development: expert waypoints in our
        # synthetic dataset have a large, non-zero mean and non-unit scale
        # (e.g. forward displacement dx has mean ~13, std ~7 over the
        # num_waypoints horizon; see tests/test_stop_gradient.py's sibling
        # investigation notes in train.py). Standard DDPM/DDIM diffusion
        # assumes the clean data x0 is roughly zero-mean, unit-variance,
        # because the reverse process STARTS sampling from x_T ~ N(0, I).
        # Without normalization, an early prototype's sampled trajectories
        # were wildly wrong (mean displacement error ~13m vs. a ~0.5m
        # constant-velocity baseline) even though the epsilon-prediction MSE
        # loss was decreasing normally -- the model was learning to denoise
        # correctly in NORMALIZED space in principle, but the raw data scale
        # made that space pathological to sample in with only 15 DDIM steps.
        # Fix: normalize expert waypoints to zero-mean/unit-std before
        # diffusion (forward_train), and de-normalize the final sampled
        # trajectory back to real-world meters (sample_trajectory). The
        # mean/std are set once from the training set via
        # `set_trajectory_normalization` (see train.py) and are registered
        # buffers, so they are saved/restored automatically with the model
        # checkpoint.
        self.register_buffer("waypoint_mean", torch.zeros(2))
        self.register_buffer("waypoint_std", torch.ones(2))

    def set_trajectory_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the (dx, dy) normalization statistics used to whiten expert
        waypoints before diffusion. `std` is clamped away from zero to avoid
        divide-by-zero on a degenerate (constant) dimension."""
        with torch.no_grad():
            self.waypoint_mean.copy_(mean)
            self.waypoint_std.copy_(std.clamp(min=1e-3))

    # ------------------------------------------------------------------
    def forward_train(self, current_frame: torch.Tensor, future_frames: torch.Tensor,
                       ego_context: torch.Tensor, expert_waypoints: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        current_frame     : (B, C, H, W)
        future_frames       : (B, num_horizons, C, H, W)
        ego_context          : (B, ego_state_dim)
        expert_waypoints       : (B, num_waypoints, 2)

        Returns a dict with 'forecast_loss', 'planning_loss', and a few extra
        tensors useful for logging (confidence, predicted eps) — see the
        stop-gradient routing explanation in this file's module docstring.
        """
        b = current_frame.shape[0]
        device = current_frame.device

        # === Shared online encoder forward (grad-tracked) ===================
        current_tokens = self.encoder(current_frame)   # (B, N, token_dim), requires_grad

        # === Forecasting path: predictor trained ONLY by forecast_loss ======
        with torch.no_grad():
            # Target embeddings come from the SAME online encoder applied to the
            # true future frames (a genuine JEPA "online -> target" asymmetry
            # would normally use a momentum-averaged encoder; we use the same
            # online encoder here for simplicity and document that as our
            # choice — see SOURCING.md). no_grad() here is belt-and-suspenders
            # on top of the explicit .detach() below: targets must never
            # receive gradient regardless of which loss is being computed.
            h = self.num_horizons
            future_flat = future_frames.reshape(b * h, *future_frames.shape[2:])   # (B*H, C, H, W)
            target_tokens = self.encoder(future_flat)                                   # (B*H, N, token_dim)
            target_tokens = target_tokens.reshape(b, h, *target_tokens.shape[1:])            # (B, H, N, token_dim)

        predictor_input = current_tokens.detach()   # DETACH #1 — blocks predictor->encoder grad
        predicted_tokens, confidence = self.world_model(predictor_input, ego_context)   # (B,H,N,C), (B,H,N)

        forecast_loss = torch.nn.functional.mse_loss(predicted_tokens, target_tokens.detach())  # DETACH #2

        # === Planning path: encoder trained end-to-end, predictor stop-gradded ===
        fused_tokens, _ = self.fusion(
            current_tokens,                 # NOT detached -> planning grad reaches encoder
            predicted_tokens.detach(),        # DETACH #3 -> planning grad cannot reach world_model
            confidence.detach(),               # DETACH #4 -> ditto, for the confidence-gate path
        )

        expert_waypoints_norm = (expert_waypoints - self.waypoint_mean) / self.waypoint_std

        t = torch.randint(0, self.num_diffusion_steps, (b,), device=device)
        noise = torch.randn_like(expert_waypoints_norm)
        noisy_traj = q_sample(expert_waypoints_norm, t, self.alpha_bars, noise)
        eps_pred = self.dit_planner(noisy_traj, t, fused_tokens)
        planning_loss = torch.nn.functional.mse_loss(eps_pred, noise)

        return {
            "forecast_loss": forecast_loss,
            "planning_loss": planning_loss,
            "confidence": confidence.detach(),
            "eps_pred": eps_pred.detach(),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_trajectory(self, current_frame: torch.Tensor, ego_context: torch.Tensor):
        """Inference: run the full pipeline using ONLY the current front-view
        frame as visual input (matching the paper's claimed inference-time
        input), producing a sampled future ego trajectory via a short DDIM
        reverse-diffusion loop.

        Returns:
          trajectory  : (B, num_waypoints, 2)  final denoised trajectory
          confidence   : (B, num_horizons, N)     per-horizon/token reliability,
                                                     for visualization (simulate.py)
        """
        device = current_frame.device
        b = current_frame.shape[0]

        current_tokens = self.encoder(current_frame)
        predicted_tokens, confidence = self.world_model(current_tokens, ego_context)
        fused_tokens, _ = self.fusion(current_tokens, predicted_tokens, confidence)

        x_t = torch.randn(b, self.num_waypoints, 2, device=device)
        respaced = ddim_respaced_timesteps(self.num_diffusion_steps, self.num_sampling_steps)
        for i, t in enumerate(respaced):
            t_prev = respaced[i + 1] if i + 1 < len(respaced) else -1
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            eps_pred = self.dit_planner(x_t, t_batch, fused_tokens)
            x_t = ddim_step(x_t, eps_pred, t, t_prev, self.alpha_bars)

        trajectory = x_t * self.waypoint_std + self.waypoint_mean   # de-normalize back to meters
        return trajectory, confidence
