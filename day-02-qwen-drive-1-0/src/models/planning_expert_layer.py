"""
PlanningExpertLayer -- one Transformer block whose self-attention is
"joint": trajectory-token queries attend over
[cached VLM K/V || trajectory K/V] in a single softmax, exactly as
described in the paper ("concatenates the cached keys and values with
those of the trajectory tokens for joint attention").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ada_ln import AdaLNModulation, modulate


class PlanningExpertLayer(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        assert model_dim % n_heads == 0, "model_dim must divide evenly by n_heads"
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads

        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)

        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False)
        hidden = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, model_dim),
        )

        self.ada_ln = AdaLNModulation(cond_dim, model_dim)

    def forward(
        self,
        traj_tokens: torch.Tensor,      # [B, T, D]  T = number of trajectory tokens
        cached_k: torch.Tensor,         # [B, n_kv_heads, L_ctx, head_dim] (RoPE already applied)
        cached_v: torch.Tensor,         # [B, n_kv_heads, L_ctx, head_dim]
        cond: torch.Tensor,             # [B, cond_dim]
    ) -> torch.Tensor:
        b, t, d = traj_tokens.shape
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond)

        # ---- Joint self+cross attention ----
        h = modulate(self.norm1(traj_tokens), shift_attn, scale_attn)   # [B, T, D]

        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)   # [B, H, T, hd]
        k_traj = self.k_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, hd]
        v_traj = self.v_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, hd]

        # Repeat cached grouped-query K/V heads up to the Planning Expert's
        # own (larger) head count if needed, GQA-style, then concatenate
        # along the token axis so one softmax attends over context + trajectory.
        n_kv_heads = cached_k.shape[1]
        if n_kv_heads != self.n_heads:
            assert self.n_heads % n_kv_heads == 0, "n_heads must be a multiple of cached n_kv_heads"
            rep = self.n_heads // n_kv_heads
            cached_k = cached_k.repeat_interleave(rep, dim=1)   # [B, H, L_ctx, hd]
            cached_v = cached_v.repeat_interleave(rep, dim=1)   # [B, H, L_ctx, hd]

        k_joint = torch.cat([cached_k, k_traj], dim=2)   # [B, H, L_ctx + T, hd]
        v_joint = torch.cat([cached_v, v_traj], dim=2)   # [B, H, L_ctx + T, hd]

        # Scaled dot-product attention: trajectory queries attend over the
        # full joint key/value set (no causal mask -- the trajectory is
        # denoised jointly, not generated autoregressively).
        attn_out = F.scaled_dot_product_attention(q, k_joint, v_joint)   # [B, H, T, hd]
        attn_out = attn_out.transpose(1, 2).reshape(b, t, d)             # [B, T, D]
        attn_out = self.out_proj(attn_out)

        traj_tokens = traj_tokens + gate_attn.unsqueeze(1) * attn_out    # residual #1

        # ---- Feed-forward ----
        h2 = modulate(self.norm2(traj_tokens), shift_mlp, scale_mlp)
        traj_tokens = traj_tokens + gate_mlp.unsqueeze(1) * self.mlp(h2)  # residual #2

        return traj_tokens   # [B, T, D]
