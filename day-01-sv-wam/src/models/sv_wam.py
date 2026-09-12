"""
SVWAM -- top-level Surround-View World-Action Model.
Reference implementation inspired by:
  "SV-WAM: An Efficient Surround-View World-Action Model for
   End-to-End Autonomous Driving" (arXiv:2609.03602, Sept 2026)

Training forward pass:
    images        -> SurroundViewTokenizer -> context tokens
    action_queries + video_queries appended
    -> N x SVWAMBlock with action-centered causal mask
    -> action_head produces the planned trajectory
    -> video_head produces predicted future latents (training signal only)

Deployment forward pass (`predict_action_only=True`):
    the video_queries / video_head are skipped entirely, so inference
    cost matches a single-camera planner even though training used all
    6 cameras + a full video-generation objective.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.tokenizer import SurroundViewTokenizer
from src.models.wam_transformer import SVWAMBlock
from src.models.heads import ActionHead, VideoHead
from src.utils.masking import build_action_centered_causal_mask


class SVWAM(nn.Module):
    def __init__(
        self,
        n_cams: int = 6,
        embed_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        n_action_steps: int = 6,     # e.g. 6 waypoints x 3 seconds @ 2 Hz
        n_video_tokens: int = 64,    # flattened future BEV/latent tokens
        action_dim: int = 3,         # (x, y, heading) per waypoint
        img_size=(224, 400),         # must match the images this model is fed
        patch_size: int = 16,
    ):
        super().__init__()
        self.tokenizer = SurroundViewTokenizer(
            n_cams=n_cams, embed_dim=embed_dim,
            img_size=tuple(img_size), patch_size=patch_size,
        )
        self.n_action_steps = n_action_steps
        self.n_video_tokens = n_video_tokens

        # Learned query tokens: the transformer "fills these in" via attention,
        # analogous to DETR object queries but for actions / future frames.
        self.action_queries = nn.Parameter(torch.zeros(1, n_action_steps, embed_dim))
        self.video_queries = nn.Parameter(torch.zeros(1, n_video_tokens, embed_dim))
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        nn.init.trunc_normal_(self.video_queries, std=0.02)

        self.blocks = nn.ModuleList(
            [SVWAMBlock(embed_dim, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        self.action_head = ActionHead(embed_dim, action_dim)
        self.video_head = VideoHead(embed_dim)

    def forward(
        self, images: torch.Tensor, predict_action_only: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # images: [B, N_cams, C, H, W]
        b = images.shape[0]
        device = images.device

        context = self.tokenizer(images)                       # [B, N*P, D]
        action_q = self.action_queries.expand(b, -1, -1)        # [B, A, D]

        if predict_action_only:
            # Deployment path: no video branch at all -> cheap inference.
            x = torch.cat([context, action_q], dim=1)           # [B, N*P+A, D]
            mask = build_action_centered_causal_mask(
                n_context=context.shape[1], n_action=action_q.shape[1],
                n_video=0, device=device,
            )
            for block in self.blocks:
                x = block(x, mask)
            x = self.final_norm(x)
            action_tokens = x[:, context.shape[1]:, :]          # [B, A, D]
            trajectory = self.action_head(action_tokens)        # [B, A, 3]
            return trajectory, None

        # Training path: jointly denoise action + future-video tokens.
        video_q = self.video_queries.expand(b, -1, -1)           # [B, V, D]
        x = torch.cat([context, action_q, video_q], dim=1)       # [B, N*P+A+V, D]
        mask = build_action_centered_causal_mask(
            n_context=context.shape[1], n_action=action_q.shape[1],
            n_video=video_q.shape[1], device=device,
        )
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)

        n_ctx, n_act = context.shape[1], action_q.shape[1]
        action_tokens = x[:, n_ctx:n_ctx + n_act, :]             # [B, A, D]
        video_tokens = x[:, n_ctx + n_act:, :]                   # [B, V, D]

        trajectory = self.action_head(action_tokens)             # [B, A, 3]
        future_latents = self.video_head(video_tokens)           # [B, V, D]
        return trajectory, future_latents
