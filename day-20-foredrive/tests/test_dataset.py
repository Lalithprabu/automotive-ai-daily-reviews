"""Basic correctness / determinism checks for the synthetic dataset."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.synthetic_dataset import SyntheticDrivingDataset


def _make_ds(num_scenes=16, seed=123):
    return SyntheticDrivingDataset(
        num_scenes=num_scenes, scene_len=16, dt=0.5, raster_size=64,
        horizons=[1, 2, 3, 4], num_waypoints=6, context_t=5,
        num_agents_min=1, num_agents_max=3, seed=seed,
    )


def test_shapes_and_ranges():
    ds = _make_ds()
    item = ds[0]
    assert item["current_frame"].shape == (3, 64, 64)
    assert item["future_frames"].shape == (4, 3, 64, 64)
    assert item["ego_context"].shape == (4,)
    assert item["expert_waypoints"].shape == (6, 2)
    assert item["cv_baseline_waypoints"].shape == (6, 2)
    assert float(item["current_frame"].min()) >= 0.0
    assert float(item["current_frame"].max()) <= 1.0


def test_deterministic_given_seed():
    ds_a = _make_ds(seed=999)
    ds_b = _make_ds(seed=999)
    item_a = ds_a[3]
    item_b = ds_b[3]
    assert torch.allclose(item_a["current_frame"], item_b["current_frame"])
    assert torch.allclose(item_a["expert_waypoints"], item_b["expert_waypoints"])


def test_different_seeds_give_different_scenes():
    ds_a = _make_ds(seed=1)
    ds_b = _make_ds(seed=2)
    diffs = 0
    for i in range(8):
        if not torch.allclose(ds_a[i]["current_frame"], ds_b[i]["current_frame"]):
            diffs += 1
    assert diffs > 0


def test_future_frames_genuinely_differ_from_current():
    """Regression guard for the dash-period aliasing bug found during
    development (see synthetic_dataset.py's DASH_PERIOD_M comment): every
    scene's current frame must differ measurably from its horizon-4 future
    frame -- if this ever goes back to ~0 for some scenes, the JEPA
    forecasting target has silently become degenerate again."""
    ds = _make_ds(num_scenes=100, seed=7)
    min_diff = float("inf")
    for i in range(len(ds)):
        item = ds[i]
        diff = (item["current_frame"] - item["future_frames"][-1]).abs().mean().item()
        min_diff = min(min_diff, diff)
    assert min_diff > 1e-3, f"Found a near-degenerate (no-change) scene, min pixel diff={min_diff}"


def test_expert_trajectory_sometimes_deviates_from_constant_velocity():
    """The reactive controller must actually produce non-trivial behavior
    (braking / evasive maneuvers) often enough that imitation learning has
    real signal beyond "go straight at constant speed"."""
    ds = _make_ds(num_scenes=200, seed=11)
    deviations = []
    for i in range(len(ds)):
        item = ds[i]
        dev = (item["expert_waypoints"] - item["cv_baseline_waypoints"]).abs().mean().item()
        deviations.append(dev)
    deviations = np.array(deviations)
    frac_nontrivial = (deviations > 0.05).mean()
    assert frac_nontrivial > 0.05, (
        f"Only {frac_nontrivial:.1%} of scenes show reactive-controller "
        "deviation from constant velocity -- imitation task may be degenerate."
    )
