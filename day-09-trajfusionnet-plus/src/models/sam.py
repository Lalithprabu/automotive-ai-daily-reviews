"""
Sequence Attention Module (SAM) — carried over from the base TrajFusionNet
(arXiv:2508.19866, verified from its full text this cycle) into
TrajFusionNet+. Two stacked transformers:

  1. A trajectory-prediction encoder-decoder that forecasts the pedestrian's
     future bounding-box trajectory from the observed past.
  2. A classification transformer that reads the concatenated
     [past || predicted] trajectory (with a 0/1 past-vs-future token) and
     produces the sequence branch's contribution to late fusion.

Verified dims (from the base paper's full text): d_model=128, past window
15 frames (0.5s @ 30fps) x 5 channels (bbox corners + speed), forecast
horizon 60 frames (2s). TrajFusionNet+'s own paper text was not reachable
this cycle (see gam.py's sourcing note) — SAM is assumed unchanged from the
base model per the abstract, which describes GAM as the addition.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SAMConfig:
    past_len: int = 15          # observed frames
    future_len: int = 60        # forecast horizon (frames)
    traj_channels: int = 5      # [x1, y1, x2, y2, speed]
    d_model: int = 128
    traj_enc_layers: int = 8
    traj_dec_layers: int = 8
    traj_heads: int = 4
    traj_ffn_dim: int = 512
    cls_layers: int = 6
    cls_heads: int = 8           # base paper reports 12; 128 isn't divisible by 12, so this
                                  # repo uses 8 (128/8=16) to keep d_model consistent across
                                  # branches while staying close to the documented value
    cls_ffn_dim: int = 1024
    proj_dim: int = 40
    dropout: float = 0.1


class TrajectoryTransformer(nn.Module):
    """Non-autoregressive encoder-decoder: encodes the past window once,
    then decodes all `future_len` future steps in a single forward pass
    (learned positional queries), rather than one step at a time."""

    def __init__(self, cfg: SAMConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.traj_channels, cfg.d_model)
        self.pos_embed_past = nn.Parameter(torch.randn(1, cfg.past_len, cfg.d_model) * 0.02)
        self.future_queries = nn.Parameter(torch.randn(1, cfg.future_len, cfg.d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.traj_heads, dim_feedforward=cfg.traj_ffn_dim,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.traj_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.traj_heads, dim_feedforward=cfg.traj_ffn_dim,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.traj_dec_layers)
        self.output_head = nn.Linear(cfg.d_model, cfg.traj_channels)

    def forward(self, past_traj: torch.Tensor) -> torch.Tensor:
        # past_traj: [B, past_len, 5]  ->  pred_future: [B, future_len, 5]
        B = past_traj.shape[0]
        src = self.input_proj(past_traj) + self.pos_embed_past
        memory = self.encoder(src)                                    # [B, past_len, d_model]
        tgt = self.future_queries.expand(B, -1, -1)                   # [B, future_len, d_model]
        decoded = self.decoder(tgt, memory)                           # [B, future_len, d_model]
        return self.output_head(decoded)                              # [B, future_len, 5]


class SequenceAttentionModule(nn.Module):
    def __init__(self, cfg: SAMConfig | None = None):
        super().__init__()
        self.cfg = cfg or SAMConfig()
        self.trajectory_transformer = TrajectoryTransformer(self.cfg)

        total_len = self.cfg.past_len + self.cfg.future_len
        self.cls_input_proj = nn.Linear(self.cfg.traj_channels + 1, self.cfg.d_model)  # +1: past/future flag
        self.cls_pos_embed = nn.Parameter(torch.randn(1, total_len, self.cfg.d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg.d_model) * 0.02)

        cls_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model, nhead=self.cfg.cls_heads, dim_feedforward=self.cfg.cls_ffn_dim,
            dropout=self.cfg.dropout, batch_first=True, norm_first=True,
        )
        self.cls_encoder = nn.TransformerEncoder(cls_layer, num_layers=self.cfg.cls_layers)
        self.proj = nn.Linear(self.cfg.d_model, self.cfg.proj_dim)

    def forward(self, past_traj: torch.Tensor):
        # past_traj: [B, past_len, 5]
        B = past_traj.shape[0]
        pred_future = self.trajectory_transformer(past_traj)         # [B, future_len, 5]

        past_flag = torch.zeros(B, self.cfg.past_len, 1, device=past_traj.device)
        future_flag = torch.ones(B, self.cfg.future_len, 1, device=past_traj.device)
        seq = torch.cat([
            torch.cat([past_traj, past_flag], dim=-1),
            torch.cat([pred_future, future_flag], dim=-1),
        ], dim=1)                                                    # [B, past+future, 6]

        x = self.cls_input_proj(seq) + self.cls_pos_embed
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)                            # [B, 1+total_len, d_model]
        encoded = self.cls_encoder(x)
        cls_out = encoded[:, 0, :]                                    # [B, d_model]

        return self.proj(cls_out), pred_future                        # [B, proj_dim], [B, future_len, 5]
