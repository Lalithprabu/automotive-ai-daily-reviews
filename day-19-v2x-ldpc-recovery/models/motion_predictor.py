"""
models/motion_predictor.py

ProbabilisticMotionPredictor: a GRU encoder-decoder that consumes a short
history of a vehicle's own kinematic state (as it would appear in its own
BSM stream) and predicts a *full predictive distribution* -- not a point
estimate -- over each of the next `future_len` BSM timesteps.

This is the "probabilistic motion prediction" component the paper
(arXiv:2609.25609) describes only abstractly ("a probabilistic motion
prediction ... keeps a full predictive probability distribution over vehicle
state"). The paper does not specify an architecture. DISCLOSED CHOICE: this
repo uses a single-layer GRU encoder-decoder with a residual pre/post
MLP block and LayerNorm around the GRU stages (a small Transformer would
also fit the paper's description; GRU was chosen here for speed/simplicity
on CPU for a daily demo repo).

Output: per future timestep, a diagonal Gaussian (mean + log-variance) over
the 5 continuous BSM fields [x, y, speed, heading, accel], in both
normalized (training/loss) space and physical (bit-probability layer) units.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.bsm import DEFAULT_FIELD_SPECS, FIELD_ORDER

LOGVAR_MIN = -8.0   # clamp bounds keep sigma in a numerically sane range
LOGVAR_MAX = 6.0


class ResidualMLPBlock(nn.Module):
    """Linear -> GELU -> Linear, added back to the input (residual), then LayerNorm."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Dim]  (or [Batch, Time, Dim], Linear broadcasts over leading dims)
        h = self.fc1(x)          # [..., Dim*expansion]
        h = self.act(h)
        h = self.fc2(h)          # [..., Dim]
        h = self.drop(h)
        return self.norm(x + h)  # residual + LayerNorm, shape unchanged: [..., Dim]


class ProbabilisticMotionPredictor(nn.Module):
    def __init__(self, num_fields: int = 5, hidden_dim: int = 64,
                 future_len: int = 5, dropout: float = 0.1):
        super().__init__()
        assert num_fields == len(FIELD_ORDER)
        self.num_fields = num_fields
        self.hidden_dim = hidden_dim
        self.future_len = future_len

        # Per-field (center, scale) normalization derived from the BSM quantizer ranges
        # in src/bsm.py, so the network operates on roughly zero-mean, unit-scale inputs.
        lows = torch.tensor([DEFAULT_FIELD_SPECS[f][0] for f in FIELD_ORDER], dtype=torch.float32)
        highs = torch.tensor([DEFAULT_FIELD_SPECS[f][1] for f in FIELD_ORDER], dtype=torch.float32)
        self.register_buffer("field_center", (lows + highs) / 2.0)   # [F]
        self.register_buffer("field_scale", (highs - lows) / 2.0)    # [F]

        # ---- Encoder ----
        self.input_proj = nn.Linear(num_fields, hidden_dim)
        self.encoder_pre_block = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.encoder_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.encoder_out_norm = nn.LayerNorm(hidden_dim)

        # ---- Decoder (autoregressive GRUCell) ----
        self.step_embed = nn.Embedding(future_len, hidden_dim)  # learned positional embedding per future step
        self.decoder_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.decoder_post_block = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.mu_head = nn.Linear(hidden_dim, num_fields)
        self.logvar_head = nn.Linear(hidden_dim, num_fields)

    def normalize(self, x_phys: torch.Tensor) -> torch.Tensor:
        # x_phys: [..., F] physical units -> normalized ~[-1, 1]
        return (x_phys - self.field_center) / self.field_scale

    def denormalize_mu(self, mu_norm: torch.Tensor) -> torch.Tensor:
        return mu_norm * self.field_scale + self.field_center

    def denormalize_sigma(self, sigma_norm: torch.Tensor) -> torch.Tensor:
        return sigma_norm * self.field_scale

    def encode(self, history_phys: torch.Tensor) -> torch.Tensor:
        """
        history_phys: [Batch, T_hist, F] physical-unit kinematic history.
        Returns final encoder hidden state: [Batch, hidden_dim]
        """
        x = self.normalize(history_phys)                 # [B, T_hist, F]
        h = self.input_proj(x)                            # [B, T_hist, hidden_dim]
        h = self.encoder_pre_block(h)                      # [B, T_hist, hidden_dim] (residual + LN)
        _, h_n = self.encoder_gru(h)                        # h_n: [1, B, hidden_dim]
        h_n = h_n.squeeze(0)                                # [B, hidden_dim]
        return self.encoder_out_norm(h_n)                   # [B, hidden_dim]

    def forward(self, history_phys: torch.Tensor):
        """
        history_phys: [Batch, T_hist, F] physical-unit kinematic history
                      (fields ordered as src.bsm.FIELD_ORDER = [x, y, speed, heading, accel]).

        Returns dict of:
          mu_norm, logvar_norm: [Batch, T_fut, F]   -- normalized space, used for the training loss
          mu_phys, sigma_phys:  [Batch, T_fut, F]   -- physical units, used by BSMBitProbabilityLayer
        """
        B = history_phys.shape[0]
        hidden = self.encode(history_phys)          # [B, hidden_dim]

        # Autoregressive decoding: feed the previous step's predicted mean (normalized)
        # as the next GRUCell input, plus a learned embedding of the future step index.
        prev_mu_norm = self.normalize(history_phys[:, -1, :])   # [B, F], last observed state as the seed input

        mus, logvars = [], []
        for t in range(self.future_len):
            step_emb = self.step_embed(torch.full((B,), t, dtype=torch.long,
                                                    device=history_phys.device))  # [B, hidden_dim]
            step_input = self.input_proj(prev_mu_norm) + step_emb                # [B, hidden_dim]
            hidden = self.decoder_cell(step_input, hidden)                        # [B, hidden_dim]
            hidden = self.decoder_post_block(hidden)                              # [B, hidden_dim] residual+LN

            mu_t = self.mu_head(hidden)                                            # [B, F] normalized-space mean
            logvar_t = self.logvar_head(hidden).clamp(LOGVAR_MIN, LOGVAR_MAX)       # [B, F] normalized-space log-var

            mus.append(mu_t)
            logvars.append(logvar_t)
            prev_mu_norm = mu_t.detach()  # autoregress on the predicted mean (detach: no BPTT through the feed-back path)

        mu_norm = torch.stack(mus, dim=1)         # [B, T_fut, F]
        logvar_norm = torch.stack(logvars, dim=1)  # [B, T_fut, F]

        mu_phys = self.denormalize_mu(mu_norm)                       # [B, T_fut, F]
        sigma_phys = self.denormalize_sigma(torch.exp(0.5 * logvar_norm))  # [B, T_fut, F]

        return {
            "mu_norm": mu_norm,
            "logvar_norm": logvar_norm,
            "mu_phys": mu_phys,
            "sigma_phys": sigma_phys,
        }


def gaussian_nll_loss(mu_norm: torch.Tensor, logvar_norm: torch.Tensor,
                       target_norm: torch.Tensor) -> torch.Tensor:
    """
    Diagonal-Gaussian negative log-likelihood, averaged over batch/time/field.
    mu_norm, logvar_norm, target_norm: [Batch, T_fut, F], all normalized space.
    """
    var = torch.exp(logvar_norm)
    nll = 0.5 * (logvar_norm + (target_norm - mu_norm) ** 2 / var + torch.log(torch.tensor(2 * torch.pi)))
    return nll.mean()
