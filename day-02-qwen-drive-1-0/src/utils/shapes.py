"""
Small shape-assertion helpers used across the Planning Expert modules --
cheap runtime checks that turn silent broadcasting bugs into clear errors.
"""
from typing import Sequence

import torch


def assert_shape(tensor: torch.Tensor, expected: Sequence[int], name: str = "tensor") -> None:
    """
    Checks tensor.shape against `expected`, where -1 in `expected` means
    "any size accepted at this position" (like einops' wildcard).

    Example: assert_shape(traj, (-1, 50, 3), "trajectory")
    """
    actual = tuple(tensor.shape)
    if len(actual) != len(expected):
        raise ValueError(f"{name}: expected rank {len(expected)}, got shape {actual}")
    for a, e in zip(actual, expected):
        if e != -1 and a != e:
            raise ValueError(f"{name}: expected shape {expected}, got {actual}")


def assert_kv_cache_compatible(n_kv_heads: int, n_heads: int, name: str = "kv_cache") -> None:
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"{name}: model n_heads ({n_heads}) must be a multiple of "
            f"cached n_kv_heads ({n_kv_heads}) for grouped-query repeat_interleave"
        )
