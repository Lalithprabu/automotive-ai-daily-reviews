"""
JEPA-style multi-horizon latent world model / latent predictor.

Per the abstract: "The planner consumes multi-horizon latent future
representations learned with a JEPA-style world model ... while
stop-gradient routing trains the latent predictor with forecasting losses
only." This module is the PREDICTOR half of that JEPA setup: given the
current frame's visual tokens (a DETACHED copy — see models/foredrive.py for
exactly where the stop-gradient boundary is enforced) plus an ego-motion
context vector, it predicts latent token embeddings for H future horizons.

This is pure latent-to-latent prediction (no pixel decoding / reconstruction
anywhere), which is the defining feature of the JEPA framing: the predictor
never has to reconstruct raw pixels, only match the target encoder's own
future embeddings.

RECONSTRUCTION DISCLOSURE: the exact predictor architecture is not specified
in the abstract. We use a small Transformer encoder shared across horizons
(a horizon embedding distinguishes which future step is being predicted, and
an ego-context vector is broadcast-added as conditioning), rather than H
independent small heads. This is documented here as our design choice — the
paper may use something else entirely (e.g. per-horizon GRUs, or a single
autoregressive rollout). We chose a shared Transformer because (a) it lets
all horizons share representational capacity while still specializing via
the horizon embedding, and (b) it keeps parameter count small, which matters
for a CPU-trainable demo repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiHorizonLatentPredictor(nn.Module):
    """Predicts future visual-token embeddings for H horizons from the current
    tokens + an ego-motion context vector.

    Shapes:
      current_tokens : (B, N, token_dim)   -- MUST be a detached copy; this
                                                module does not enforce that
                                                itself (see foredrive.py).
      ego_context     : (B, ego_state_dim)
      returns          : (B, H, N, token_dim)  predicted future token embeddings
    """

    def __init__(self, token_dim: int, ego_state_dim: int, num_horizons: int,
                 hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.token_dim = token_dim
        self.num_horizons = num_horizons

        # Project raw token_dim tokens into the predictor's working hidden_dim.
        self.in_proj = nn.Linear(token_dim, hidden_dim)

        # Ego context (vx, vy, accel_proxy, speed) -> hidden_dim conditioning vector,
        # broadcast-added to every token so the predictor knows how the ego is
        # currently moving (a static image alone under-determines the future).
        self.ego_mlp = nn.Sequential(
            nn.Linear(ego_state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # One learned embedding per future horizon (h=1..H), added to every token
        # before the shared Transformer so a single forward pass over the batch
        # dimension "B*H" can specialize its prediction per horizon.
        self.horizon_embed = nn.Parameter(torch.randn(num_horizons, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_proj = nn.Linear(hidden_dim, token_dim)

        # Small confidence head: per-horizon, per-token scalar in [0, 1] used
        # downstream by the fusion module's "future-status injection" gating.
        # It lives here (not in fusion.py) because it reads the predictor's own
        # hidden representation before the final projection back to token_dim.
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, current_tokens: torch.Tensor, ego_context: torch.Tensor):
        """
        current_tokens: (B, N, token_dim)  -- caller-detached, see foredrive.py
        ego_context   : (B, ego_state_dim)
        returns:
          predicted_tokens : (B, H, N, token_dim)
          confidence        : (B, H, N)   per-horizon, per-token scalar in [0, 1]
        """
        b, n, _ = current_tokens.shape
        h = self.num_horizons

        x = self.in_proj(current_tokens)                     # (B, N, hidden)
        ego_cond = self.ego_mlp(ego_context)                    # (B, hidden)
        x = x + ego_cond.unsqueeze(1)                             # broadcast over tokens -> (B, N, hidden)

        # Expand to (B, H, N, hidden), add the per-horizon embedding, then fold
        # (B, H) into one batch dimension so the Transformer processes all
        # horizons in a single shared forward pass.
        x = x.unsqueeze(1).expand(b, h, n, x.shape[-1]).clone()  # (B, H, N, hidden)
        x = x + self.horizon_embed.view(1, h, 1, -1)               # broadcast horizon embed
        x = x.reshape(b * h, n, -1)                                  # (B*H, N, hidden)

        x = self.transformer(x)                                        # (B*H, N, hidden)

        predicted_tokens = self.out_proj(x).reshape(b, h, n, self.token_dim)  # (B, H, N, token_dim)
        confidence = torch.sigmoid(self.confidence_head(x)).reshape(b, h, n)     # (B, H, N)

        return predicted_tokens, confidence
