"""
SVWAMBlock -- one pre-norm Transformer block shared by context/action/video
tokens. Stacking these with the action-centered causal mask (see
src/utils/masking.py) is what lets SV-WAM jointly denoise action and
future-video tokens at train time.
"""
import torch
import torch.nn as nn


class SVWAMBlock(nn.Module):
    """One pre-norm Transformer block shared by context/action/video tokens."""

    def __init__(self, embed_dim: int = 512, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out                        # residual connection #1
        x = x + self.mlp(self.norm2(x))         # residual connection #2
        return x
