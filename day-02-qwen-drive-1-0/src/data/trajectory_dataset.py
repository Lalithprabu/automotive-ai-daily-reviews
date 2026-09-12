"""
TrajectoryDataset -- adapter template for training the Planning Expert.

The Planning Expert is a bolt-on module that consumes a frozen VLM's cached
key/value activations; it does NOT re-implement the VLM itself. This
dataset therefore yields *pre-extracted* caches and pooled embeddings
rather than raw images/text -- run your VLM backbone offline (or in a
separate data-prep pass) and cache its outputs, or wire in a live
extraction hook in `_load_sample` if you'd rather compute them on the fly.

Expected manifest entry (dict), one per training example:
    {
        "kv_cache_path": str,          # .pt file with {"keys": [...], "values": [...]}
        "trajectory": [[x, y, heading], ...] * n_waypoints,  # ground truth
        "instruction_embed_path": str,  # pooled language-instruction embedding, .pt
        "ego_state_embed_path": str,     # pooled ego speed/heading/history embedding, .pt
    }
"""
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.models.vlm_kv_cache import VLMKVCache


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        manifest: Optional[List[Dict]] = None,
        model_dim: int = 1024,
        cond_dim: int = 1024,
        n_caches: int = 8,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        l_ctx: int = 32,
        n_waypoints: int = 50,
        synthetic: bool = False,
        synthetic_length: int = 64,
    ):
        self.model_dim = model_dim
        self.cond_dim = cond_dim
        self.n_caches = n_caches
        self.n_kv_heads = n_kv_heads
        # GQA convention: kv heads share the SAME head_dim as the model's
        # query heads (model_dim // n_heads) -- there are just fewer of
        # them. Using model_dim // n_kv_heads here would make cached_k/v
        # incompatible with the query projection's head_dim in
        # PlanningExpertLayer's joint attention (see that file's `k_joint`
        # concatenation).
        self.head_dim = model_dim // n_heads
        self.l_ctx = l_ctx
        self.n_waypoints = n_waypoints
        self.synthetic = synthetic or manifest is None
        self.manifest = manifest or []
        self.synthetic_length = synthetic_length

    def __len__(self) -> int:
        return self.synthetic_length if self.synthetic else len(self.manifest)

    def __getitem__(self, idx: int) -> Dict:
        if self.synthetic:
            return self._synthetic_sample()
        return self._load_sample(self.manifest[idx])

    def _synthetic_sample(self) -> Dict:
        kv_cache = VLMKVCache(
            keys=[torch.randn(self.n_kv_heads, self.l_ctx, self.head_dim) for _ in range(self.n_caches)],
            values=[torch.randn(self.n_kv_heads, self.l_ctx, self.head_dim) for _ in range(self.n_caches)],
        )
        return {
            "kv_cache": kv_cache,
            "trajectory": torch.randn(self.n_waypoints, 3),
            "instruction_embed": torch.randn(self.cond_dim),
            "ego_state_embed": torch.randn(self.cond_dim),
        }

    def _load_sample(self, entry: Dict) -> Dict:
        # TODO: replace with real .pt loading once kv_cache_path / embed
        # paths point at your own pre-extracted VLM cache/embeddings.
        raise NotImplementedError(
            "Wire this up to your own VLM cache-extraction pipeline; "
            "the manifest format is documented in this file's module docstring."
        )


def collate_kv_cache_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate_fn: stacks the per-sample VLMKVCache list-of-tensors into
    batched tensors (default collate can't do this for a dataclass of lists).
    """
    n_caches = len(batch[0]["kv_cache"])
    keys = [torch.stack([b["kv_cache"].keys[i] for b in batch]) for i in range(n_caches)]
    values = [torch.stack([b["kv_cache"].values[i] for b in batch]) for i in range(n_caches)]

    return {
        "kv_cache": VLMKVCache(keys=keys, values=values),
        "trajectory": torch.stack([b["trajectory"] for b in batch]),
        "instruction_embed": torch.stack([b["instruction_embed"] for b in batch]),
        "ego_state_embed": torch.stack([b["ego_state_embed"] for b in batch]),
    }
