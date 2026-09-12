"""
VLMKVCache -- container for the 8 cached (K, V) pairs pulled from 8 groups
of the frozen VLM's grouped-query attention layers.
"""
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class VLMKVCache:
    """
    Holds the 8 (key, value) pairs cached from 8 groups of the frozen VLM's
    grouped-query-attention layers. Keys already have rotary position
    embedding (RoPE) applied, matching how the paper caches them.

    Each tensor: [B, n_kv_heads, L_ctx, head_dim]
        B        = batch size
        n_kv_heads = number of grouped-query KV heads in the VLM backbone
        L_ctx    = number of cached context tokens (image + text tokens)
        head_dim = per-head channel dimension (must match the Planning
                   Expert's own head_dim so Q/K can be dot-producted)
    """
    keys: List[torch.Tensor]    # len == n_caches, each [B, n_kv_heads, L_ctx, head_dim]
    values: List[torch.Tensor]  # same shapes as keys

    def __post_init__(self):
        assert len(self.keys) == len(self.values), "keys/values count mismatch"

    def __len__(self) -> int:
        return len(self.keys)
