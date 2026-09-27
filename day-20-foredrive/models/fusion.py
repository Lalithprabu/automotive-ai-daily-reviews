"""
Gated visual fusion + future-status injection.

Per the abstract: "Because predicted futures have varying reliability across
horizons ... we use gated visual fusion, future-status injection, and
Trajectory-Adaptive Bias (TAB) to inject future latents as guidance without
overriding the current observation." This module implements the first two
of those three mechanisms (TAB lives in models/tab.py, since it is applied
inside the DiT planner's attention, not here).

Two ideas combined:

1. GATED VISUAL FUSION: each predicted future token is scaled by a learned
   per-token, per-horizon confidence/reliability scalar in [0, 1] (produced
   by the world model's confidence head — see models/world_model.py). A
   horizon the model is unsure about (e.g. a far-future prediction, or one
   near an unresolved multi-agent interaction) contributes less to the
   fused representation than a confident one. This directly implements
   "inject future latents as guidance without overriding the current
   observation": the current-frame tokens are NEVER gated/down-weighted,
   only the predicted future tokens are.

2. FUTURE-STATUS INJECTION: alongside the gated latent, we inject a small
   per-horizon "status" embedding built from (a) a learned embedding of the
   horizon index and (b) the scalar confidence value itself, so the DiT
   planner's conditioning tokens carry explicit metadata about *which*
   future horizon a token came from and *how reliable* it currently is —
   not just an implicitly-gated vector.

RECONSTRUCTION DISCLOSURE: the exact fusion architecture is not specified at
the abstract level we have access to. This module is our own reconstruction,
documented as such (see SOURCING.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFutureFusion(nn.Module):
    """Fuses current-frame tokens with H sets of confidence-gated, status-tagged
    predicted future tokens into one conditioning token sequence for the DiT
    planner.

    Shapes:
      current_tokens   : (B, N, token_dim)
      predicted_tokens  : (B, H, N, token_dim)  -- caller controls whether this
                                                     carries gradient into the
                                                     predictor (see foredrive.py:
                                                     detached for the planning
                                                     path, per stop-gradient
                                                     routing)
      confidence         : (B, H, N)              in [0, 1], from the world model
      returns              : (B, N * (1 + H), fused_token_dim) fused conditioning
                              tokens, plus the raw per-horizon confidence (for
                              logging / visualization in simulate.py)
    """

    def __init__(self, token_dim: int, num_horizons: int, status_dim: int, fused_token_dim: int):
        super().__init__()
        self.num_horizons = num_horizons
        self.fused_token_dim = fused_token_dim

        self.current_proj = nn.Linear(token_dim, fused_token_dim)
        self.future_proj = nn.Linear(token_dim, fused_token_dim)

        # Learned embedding per horizon index (0..H-1), used both for the
        # "current" tokens (which get a fixed dedicated marker, index -> row 0
        # of a separate embedding so it can never collide with a real future
        # horizon) and for each future horizon's status vector.
        self.horizon_id_embed = nn.Embedding(num_horizons, status_dim // 2)
        self.current_marker = nn.Parameter(torch.randn(1, status_dim // 2) * 0.02)

        # Status MLP: [horizon-id embedding (status_dim//2) ; confidence scalar
        # broadcast to status_dim//2] -> status_dim. Keeping confidence as a
        # broadcast block (not a single scalar concatenated raw) gives the
        # tiny MLP more capacity to re-scale it meaningfully.
        self.status_mlp = nn.Sequential(
            nn.Linear(status_dim, status_dim),
            nn.GELU(),
        )
        self.status_dim = status_dim

        # Combine [gated future projection ; status embedding] -> fused_token_dim
        self.future_combine = nn.Linear(fused_token_dim + status_dim, fused_token_dim)
        self.current_combine = nn.Linear(fused_token_dim + status_dim, fused_token_dim)

    def forward(self, current_tokens: torch.Tensor, predicted_tokens: torch.Tensor,
                confidence: torch.Tensor):
        b, n, _ = current_tokens.shape
        h = self.num_horizons
        half = self.status_dim // 2

        # --- current tokens: tagged with a fixed "not-a-future-horizon" marker ---
        cur = self.current_proj(current_tokens)                     # (B, N, fused)
        cur_status_id = self.current_marker.expand(b, half)             # (B, half)
        cur_status_conf = torch.ones(b, half, device=current_tokens.device)  # confidence=1 for "now"
        cur_status = self.status_mlp(torch.cat([cur_status_id, cur_status_conf], dim=-1))  # (B, status_dim)
        cur_status = cur_status.unsqueeze(1).expand(b, n, self.status_dim)      # (B, N, status_dim)
        cur_fused = self.current_combine(torch.cat([cur, cur_status], dim=-1))     # (B, N, fused)

        # --- future tokens: gated by confidence, tagged with per-horizon status ---
        # GATING (mechanism 1): scale each predicted token by its own scalar
        # confidence before it ever reaches the planner, so low-confidence
        # (unreliable) future predictions contribute a near-zero vector rather
        # than actively misleading the planner.
        gated = predicted_tokens * confidence.unsqueeze(-1)                # (B, H, N, token_dim)
        fut = self.future_proj(gated)                                          # (B, H, N, fused)

        horizon_ids = torch.arange(h, device=current_tokens.device)              # (H,)
        horizon_id_emb = self.horizon_id_embed(horizon_ids)                        # (H, half)
        horizon_id_emb = horizon_id_emb.view(1, h, 1, half).expand(b, h, n, half)     # (B, H, N, half)

        # FUTURE-STATUS INJECTION (mechanism 2): concat horizon-identity embedding
        # with the (broadcast) scalar confidence, so the planner sees not just a
        # gated vector but an explicit "this is horizon h, currently trusted at
        # level c" tag.
        conf_broadcast = confidence.unsqueeze(-1).expand(b, h, n, half)              # (B, H, N, half)
        status = self.status_mlp(torch.cat([horizon_id_emb, conf_broadcast], dim=-1))    # (B, H, N, status_dim)

        fut_fused = self.future_combine(torch.cat([fut, status], dim=-1))                  # (B, H, N, fused)
        fut_fused = fut_fused.reshape(b, h * n, self.fused_token_dim)                          # (B, H*N, fused)

        fused_tokens = torch.cat([cur_fused, fut_fused], dim=1)   # (B, N*(1+H), fused_token_dim)
        return fused_tokens, confidence
