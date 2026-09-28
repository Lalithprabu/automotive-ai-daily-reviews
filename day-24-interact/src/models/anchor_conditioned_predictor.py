"""
AnchorConditionedPredictor -- the centerpiece module of this reconstruction.

This is this project's OWN reconstruction of the paper's "prediction" stage (see
SOURCING.md for exactly what is paper-sourced vs. reconstructed). The paper states
the key property qualitatively: surrounding agents respond to the ego's *intent*
rather than its exact path, so a single reactive prediction stays valid across an
entire family of plans sharing that intent. This module is built to have exactly
that property and nothing more is assumed about its internals from the paper.

Design (all reconstruction choices):
  - A GRU encoder reads the joint history of ego + other agent.
  - The K-dim (here 2-dim) anchor/intent vector is projected through a small MLP
    into FiLM parameters (gamma, beta) that affinely modulate the encoder's summary
    state. This is the "novel conditioning mechanism" analogue mentioned in the
    LinkedIn template -- a cheap, differentiable way for one scalar intent vector to
    reshape the entire downstream prediction.
  - A lightweight cross-attention layer lets the autoregressive decoder look back at
    every encoded history timestep (not just the final hidden state) while unrolling
    the future.
  - A GRUCell decoder unrolls `future_len` steps, each producing the other agent's
    predicted (lat, spd) at that timestep.

Critical property (unit-tested in tests/test_predictor.py): calling forward() with
the SAME history but DIFFERENT `cond` vectors must produce genuinely different
predictions -- otherwise "anchor conditioning" would be a no-op and the whole
efficiency argument (query once per anchor, not once per candidate) would be moot.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AnchorConditionedPredictor(nn.Module):
    def __init__(self, state_dim: int = 4, cond_dim: int = 2, hidden_dim: int = 32,
                 future_len: int = 10, future_dim: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.future_len = future_len
        self.future_dim = future_dim

        # --- Encoder: joint history of ego + other agent ---
        # Input at each past timestep: concat(ego_state[state_dim], other_state[state_dim])
        self.encoder = nn.GRU(input_size=state_dim * 2, hidden_size=hidden_dim, batch_first=True)

        # --- Conditioning branch: anchor/intent vector -> FiLM (gamma, beta) ---
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),   # -> [gamma | beta], each hidden_dim wide
        )

        # --- Cross-attention: decoder query attends over ALL encoded history steps ---
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # --- Autoregressive decoder ---
        self.decoder_cell = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.output_head = nn.Linear(hidden_dim, future_dim)

    def forward(self, ego_hist: torch.Tensor, other_hist: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ego_hist:   [B, H, state_dim]   past ego (lat, spd, dlat, dspd)
            other_hist: [B, H, state_dim]   past other-agent state, same layout
            cond:       [B, cond_dim]       ego's committed intent/anchor vector
                                             (target_lat_offset, target_speed_delta).
                                             THIS is the single query variable that
                                             changes between "once per anchor" (new
                                             way) and "once per candidate" (naive
                                             old way B) usage -- the module itself
                                             does not know or care which regime it
                                             is being used in.

        Returns:
            pred_future: [B, future_len, future_dim]  predicted (lat, spd) of the
                         OTHER agent at each future timestep, conditioned on `cond`.
        """
        assert ego_hist.dim() == 3 and other_hist.dim() == 3, "expected [B,H,state_dim] histories"
        B, H, _ = ego_hist.shape

        enc_in = torch.cat([ego_hist, other_hist], dim=-1)        # [B, H, 2*state_dim]
        enc_out, h_n = self.encoder(enc_in)                        # enc_out: [B,H,hidden]  h_n: [1,B,hidden]
        h_n = h_n.squeeze(0)                                       # [B, hidden]

        # --- FiLM conditioning: this is where `cond` genuinely changes the prediction ---
        gamma_beta = self.cond_mlp(cond)                           # [B, 2*hidden]
        gamma, beta = gamma_beta.chunk(2, dim=-1)                  # each [B, hidden]
        h_cond = h_n * (1.0 + gamma) + beta                        # [B, hidden]

        keys = self.key_proj(enc_out)                              # [B, H, hidden]
        values = self.value_proj(enc_out)                          # [B, H, hidden]

        dec_hidden = h_cond                                        # [B, hidden]  decoder init state
        outputs = []
        for _t in range(self.future_len):
            query = self.query_proj(dec_hidden).unsqueeze(1)                       # [B, 1, hidden]
            attn_logits = torch.bmm(query, keys.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # [B,1,H]
            attn_weights = torch.softmax(attn_logits, dim=-1)                       # [B,1,H]
            context = torch.bmm(attn_weights, values).squeeze(1)                    # [B, hidden]

            dec_hidden = self.decoder_cell(context, dec_hidden)                     # [B, hidden]
            step_out = self.output_head(dec_hidden)                                 # [B, future_dim]
            outputs.append(step_out)

        pred_future = torch.stack(outputs, dim=1)                  # [B, future_len, future_dim]
        return pred_future
