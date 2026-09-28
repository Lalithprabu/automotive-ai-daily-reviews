"""
AnchorGenerator: derives K diverse "anchor" intents from simple map/goal geometry.

Per this project's reconstruction of INTERACT's decomposition, an anchor is NOT a
full trajectory -- it is a compact (target_lat_offset, target_speed_delta) intent
vector plus a human-readable name. Downstream, the AnchorConditionedPredictor is
queried once per anchor, and the TrustRegionCEM refiner expands each anchor into a
whole *family* of concrete candidate trajectories during optimization.

This is this project's own reconstruction default (see SOURCING.md) -- the paper
describes anchors as "derived from map geometry" but does not publish an exact
parameterization, so the 5 canonical negotiation intents below (aggressive merge,
accelerate past, hold lane / delay, cautious yield, decelerate & follow) are a
reasonable, clearly-disclosed stand-in for a merge / unprotected-turn scenario.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.utils.kinematics import assertiveness

# (name, base_lat_offset, base_speed_delta) -- canonical, un-scaled anchor directions
_BASE_ANCHOR_DEFS = [
    ("aggressive_merge", 1.0, 0.8),
    ("accelerate_past", 0.3, 1.0),
    ("hold_lane_delay", 0.0, 0.0),
    ("cautious_yield", -0.3, -0.6),
    ("decelerate_follow", -1.0, -1.0),
]


@dataclass
class Anchor:
    name: str
    cond: torch.Tensor        # [2] (target_lat_offset, target_speed_delta)
    assertiveness: float      # scalar summary used only for analysis / bucketing


class AnchorGenerator:
    """Derives K anchors from simple scenario geometry.

    Args:
        anchor_defs: list of (name, base_lat_offset, base_speed_delta). Defaults to
                     the 5 canonical merge/negotiation intents above.
    """

    def __init__(self, anchor_defs=None):
        self.anchor_defs = anchor_defs if anchor_defs is not None else _BASE_ANCHOR_DEFS

    @property
    def k(self) -> int:
        return len(self.anchor_defs)

    def generate(self, gap_to_merge: float = 3.0, lane_width: float = 1.0) -> list[Anchor]:
        """Scales the canonical anchor directions by simple scenario geometry.

        Args:
            gap_to_merge: distance/time-like scalar to the merge point -- a smaller
                gap means the negotiation is more urgent, so speed intents are
                amplified. (reconstruction default, not paper-sourced)
            lane_width: scales lateral offsets (wider lane -> larger physical
                lateral excursions for the same "intent strength").

        Returns:
            list of K Anchor objects, each a distinct (name, cond, assertiveness).
        """
        # urgency_scale grows as gap shrinks, clipped to a sane range
        urgency_scale = float(torch.clamp(torch.tensor(3.0 / max(gap_to_merge, 1e-3)), 0.6, 1.4))

        anchors = []
        for name, base_lat, base_spd in self.anchor_defs:
            lat = base_lat * lane_width
            spd = base_spd * urgency_scale
            cond = torch.tensor([lat, spd], dtype=torch.float32)
            a = float(assertiveness(cond))
            anchors.append(Anchor(name=name, cond=cond, assertiveness=a))
        return anchors

    def anchor_center_bank(self, gap_to_merge: float = 3.0, lane_width: float = 1.0) -> torch.Tensor:
        """Convenience: stacks all anchors' cond vectors into a [K, 2] tensor,
        used by the trust-region penalty to find "distance to nearest anchor"."""
        anchors = self.generate(gap_to_merge=gap_to_merge, lane_width=lane_width)
        return torch.stack([a.cond for a in anchors], dim=0)   # [K, 2]
