"""
Synthetic interactive merge/negotiation scenario.

This is a deliberately small, fully synthetic simulator -- NOT real driving data --
built so that the "other agent's" future genuinely, falsifiably depends on the
ego's committed intent. This is the ground-truth structure the
AnchorConditionedPredictor must learn, and which the unconditional ("Old Way A")
baseline is fundamentally unable to capture, giving the ablation in train.py /
tests/test_regression_core_claim.py real teeth: if the conditioned model failed to
learn the dependency, its measured ADE/FDE would NOT beat the baseline, and this
project's honesty norm is to report that plainly rather than hide it.

Ground-truth reactive rule (this project's own choice, disclosed in SOURCING.md):
    a = assertiveness(ego_cond) = 0.5 * target_lat_offset + 0.5 * target_speed_delta
    other_future_speed(t) = other_spd0 - k_speed * a * ramp(t)      + noise
    other_future_lat(t)   = other_lat0 - k_lateral * a * ramp(t)^1.2 + noise
i.e. the more assertively the ego commits (positive a = merging/accelerating
through), the more the other agent decelerates and yields laterally away; the more
cautiously the ego commits (negative a = yielding/following), the more the other
agent proceeds at its own pace. This is intentionally a real, learnable but noisy
dependency -- not a trivial lookup table -- so a model that ignores `cond` entirely
(the unconditional baseline) will have systematically higher error specifically in
scenarios with |a| large (the "interactive" bucket).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.utils.kinematics import assertiveness, build_history, rollout_trajectory


def generate_episode(cond: torch.Tensor, history_len: int = 8, future_len: int = 10,
                      base_speed: float = 1.0, k_speed: float = 0.6, k_lateral: float = 0.5,
                      noise_std: float = 0.04) -> dict:
    """Generates one synthetic episode: past history (causally independent of `cond`,
    since the ego hasn't committed yet) + the other agent's REACTIVE future (which
    does depend on `cond`) + the ego's own deterministic future rollout.
    """
    a = assertiveness(cond)   # scalar

    ego_spd0 = base_speed + 0.05 * torch.randn(1).item()
    ego_lat0 = 0.0
    other_lat0 = 0.15 * torch.randn(1).item()
    other_spd0 = base_speed + 0.05 * torch.randn(1).item()

    ego_hist = build_history(ego_lat0, ego_spd0, history_len)
    other_hist = build_history(other_lat0, other_spd0, history_len)

    ego_future = rollout_trajectory(cond, future_len, base_speed=ego_spd0)   # [T, 2]

    t = torch.linspace(1.0 / future_len, 1.0, future_len)
    ramp = t
    other_future_spd = other_spd0 - k_speed * a * ramp + noise_std * torch.randn(future_len)
    other_future_lat = other_lat0 - k_lateral * a * ramp.pow(1.2) + noise_std * torch.randn(future_len)
    other_future = torch.stack([other_future_lat, other_future_spd], dim=-1)   # [T, 2]

    return dict(
        ego_hist=ego_hist, other_hist=other_hist, cond=cond.float(),
        other_future=other_future, ego_future=ego_future, assertiveness=torch.tensor(float(a)),
    )


class InteractiveMergeDataset(Dataset):
    """Eagerly generates `n_samples` synthetic episodes at construction time
    (deterministic given `seed`) and serves them as a map-style Dataset.

    cond_mode:
        "continuous" -- cond sampled uniformly in [-bound, bound]^2. Used to train
            the AnchorConditionedPredictor so it generalizes to ANY intent vector,
            not just the 5 canonical anchors (this is required for the "naive
            per-candidate re-query" baseline, which conditions on arbitrary CEM
            candidate intents, not just anchors).
        "anchors" -- cond sampled uniformly from a fixed provided list of anchor
            cond vectors (with small jitter). Used to build an anchor-bucketed
            validation set for the regression test / ADE-FDE-by-anchor report.
    """

    def __init__(self, n_samples: int, seed: int, history_len: int = 8, future_len: int = 10,
                 base_speed: float = 1.0, k_speed: float = 0.6, k_lateral: float = 0.5,
                 noise_std: float = 0.04, cond_mode: str = "continuous", cond_bound: float = 1.3,
                 anchor_conds: list[torch.Tensor] | None = None, anchor_names: list[str] | None = None):
        super().__init__()
        assert cond_mode in ("continuous", "anchors")
        gen = torch.Generator().manual_seed(seed)

        self.samples = []
        for i in range(n_samples):
            if cond_mode == "continuous":
                cond = (torch.rand(2, generator=gen) * 2 - 1) * cond_bound
                anchor_name = None
            else:
                assert anchor_conds is not None
                idx = torch.randint(0, len(anchor_conds), (1,), generator=gen).item()
                jitter = 0.02 * torch.randn(2, generator=gen)
                cond = anchor_conds[idx] + jitter
                anchor_name = anchor_names[idx] if anchor_names is not None else str(idx)

            ep = generate_episode(cond, history_len=history_len, future_len=future_len,
                                   base_speed=base_speed, k_speed=k_speed, k_lateral=k_lateral,
                                   noise_std=noise_std)
            if anchor_name is not None:
                ep["anchor_name"] = anchor_name
            self.samples.append(ep)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_episodes(batch: list[dict]) -> dict:
    """Simple collate_fn: stacks the tensor fields; leaves anchor_name (if present) as a list."""
    out = {}
    for key in ("ego_hist", "other_hist", "cond", "other_future", "ego_future", "assertiveness"):
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    if "anchor_name" in batch[0]:
        out["anchor_name"] = [b["anchor_name"] for b in batch]
    return out
