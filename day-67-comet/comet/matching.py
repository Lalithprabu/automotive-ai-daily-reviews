"""Mutual-nearest-neighbour matching between ego and roadside detections."""
from __future__ import annotations

import torch


def mutual_nn_pairs(
    ego_xy: torch.Tensor,        # (B, N, 2)  ego-vehicle detections, ego frame [m]
    road_xy: torch.Tensor,       # (B, M, 2)  roadside detections, already mapped to ego frame [m]
    ego_mask: torch.Tensor,      # (B, N) bool, True = real detection (False = padding)
    road_mask: torch.Tensor,     # (B, M) bool
    gate: torch.Tensor,          # (B,) matching radius per sender [m]
) -> torch.Tensor:
    """Return a (B, N, M) boolean tensor marking *mutually nearest* pairs within ``gate``.

    A pair (i, j) is kept iff
      1. j is the closest valid roadside box to ego box i,
      2. i is the closest valid ego box to roadside box j, and
      3. their distance is below ``gate``.
    Requiring mutual agreement is what makes the offset estimate robust: one-sided nearest
    neighbours are exactly the pairs that a wrong shared shift would fabricate.
    """
    B, N, _ = ego_xy.shape
    M = road_xy.shape[1]
    d = torch.cdist(ego_xy, road_xy)                                        # (B, N, M)
    valid = ego_mask[:, :, None] & road_mask[:, None, :]                    # (B, N, M)
    d = torch.where(valid, d, torch.full_like(d, float("inf")))
    if N == 0 or M == 0:
        return torch.zeros(B, N, M, dtype=torch.bool, device=ego_xy.device)
    j_of_i = d.argmin(dim=2)                                                # (B, N)
    i_of_j = d.argmin(dim=1)                                                # (B, M)
    fwd = torch.zeros_like(valid)
    fwd.scatter_(2, j_of_i[:, :, None], True)                               # i -> its nearest j
    bwd = torch.zeros_like(valid)
    bwd.scatter_(1, i_of_j[:, None, :], True)                               # j -> its nearest i
    return fwd & bwd & valid & (d < gate[:, None, None])
