"""MM-Tokenizer: compresses multi-camera video into compact planning-oriented
tokens ("MM-Tokens"), mirroring MM-Future's visual-encoding stage.

Paper mechanism (arXiv:2609.20377), reconstructed here at reduced scale:
  1. A shared CNN backbone extracts a patch-token grid from each camera crop.
  2. A small set of *learnable register queries*, one group per camera,
     cross-attends over that camera's patch tokens — this is the
     "register tokens... aggregate information across cameras" step.
  3. Every `frames_per_chunk` consecutive frames are concatenated and
     projected down to a fixed number of chunk tokens ("every two-frame
     temporal chunk is compressed into 64 tokens, each with dimension 256"
     in the paper; this build uses `chunk_tokens` / `d_model` from
     config.yaml, which are smaller for CPU trainability).

Shapes are annotated at every step.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _PatchBackbone(nn.Module):
    """Small conv backbone: camera crop -> patch-token grid."""

    def __init__(self, in_channels: int, patch_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, patch_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, patch_dim // 2),
            nn.GELU(),
            nn.Conv2d(patch_dim // 2, patch_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, patch_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, crop, crop] -> feat: [N, patch_dim, crop/4, crop/4]
        feat = self.net(x)
        n, c, h, w = feat.shape
        # -> patch tokens: [N, h*w, patch_dim]
        return feat.flatten(2).transpose(1, 2)


class RegisterAggregator(nn.Module):
    """Learnable register queries cross-attend over one camera's patch
    tokens, producing a fixed number of tokens per camera regardless of
    the patch-grid resolution."""

    def __init__(self, patch_dim: int, d_model: int, num_registers: int, n_heads: int = 4):
        super().__init__()
        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.randn(num_registers, d_model) * 0.02)
        self.kv_proj = nn.Linear(patch_dim, d_model * 2)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [N, P, patch_dim] (N = batch * cameras * frames)
        n = patch_tokens.shape[0]
        kv = self.kv_proj(patch_tokens)                       # [N, P, 2*d_model]
        k, v = kv.chunk(2, dim=-1)                             # each [N, P, d_model]
        q = self.registers.unsqueeze(0).expand(n, -1, -1)      # [N, R, d_model]
        out, _ = self.attn(q, k, v, need_weights=False)
        return self.norm(out)                                  # [N, R, d_model]


class ChunkCompressor(nn.Module):
    """Concatenates `frames_per_chunk` frames' register tokens (across all
    cameras) and projects them to a fixed-size set of MM-Tokens per chunk."""

    def __init__(self, d_model: int, num_cameras: int, register_tokens: int,
                 frames_per_chunk: int, chunk_tokens: int):
        super().__init__()
        in_tokens = num_cameras * register_tokens * frames_per_chunk
        self.chunk_tokens = chunk_tokens
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        # Learned linear "token mixer": in_tokens -> chunk_tokens, applied
        # along the token axis (a fixed-size compression, analogous to a
        # perceiver-resampler step).
        self.token_mix = nn.Parameter(torch.randn(chunk_tokens, in_tokens) * (1.0 / in_tokens ** 0.5))
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, in_tokens, d_model]
        tokens = self.pool(tokens)
        mixed = torch.einsum("ct,btd->bcd", self.token_mix, tokens)  # [B, chunk_tokens, d_model]
        return self.out_norm(mixed)


class MMTokenizer(nn.Module):
    """Full pipeline: camera crops for `frames_per_chunk` frames -> MM-Tokens
    for one temporal chunk.

    forward() consumes a whole clip already grouped into chunks:
      cameras: [B, num_chunks, frames_per_chunk, num_cameras, C, crop, crop]
    and returns:
      mm_tokens: [B, num_chunks, chunk_tokens, d_model]
    """

    def __init__(self, in_channels: int, patch_dim: int, d_model: int,
                 num_cameras: int, register_tokens: int, frames_per_chunk: int,
                 chunk_tokens: int, n_heads: int = 4):
        super().__init__()
        self.num_cameras = num_cameras
        self.frames_per_chunk = frames_per_chunk
        self.backbone = _PatchBackbone(in_channels, patch_dim)
        self.register_agg = RegisterAggregator(patch_dim, d_model, register_tokens, n_heads)
        self.chunk_compressor = ChunkCompressor(
            d_model, num_cameras, register_tokens, frames_per_chunk, chunk_tokens
        )

    def forward(self, cameras: torch.Tensor) -> torch.Tensor:
        b, n_chunks, fpc, n_cam, c, crop, _ = cameras.shape
        assert fpc == self.frames_per_chunk and n_cam == self.num_cameras

        flat = cameras.reshape(b * n_chunks * fpc * n_cam, c, crop, crop)
        patch_tokens = self.backbone(flat)                 # [N, P, patch_dim]
        reg_tokens = self.register_agg(patch_tokens)         # [N, R, d_model]
        r = reg_tokens.shape[1]

        reg_tokens = reg_tokens.reshape(b * n_chunks, fpc * n_cam * r, -1)  # [B*chunks, in_tokens, d_model]
        chunk_tokens = self.chunk_compressor(reg_tokens)      # [B*chunks, chunk_tokens, d_model]
        d = chunk_tokens.shape[-1]
        return chunk_tokens.reshape(b, n_chunks, -1, d)        # [B, chunks, chunk_tokens, d_model]
