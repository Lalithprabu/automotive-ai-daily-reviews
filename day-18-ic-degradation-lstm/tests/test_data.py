"""
tests/test_data.py
Sanity tests for the synthetic dataset generator: correct shapes, no NaNs,
correct cell count, genuine cell-level train/val/test separation, and a
check that the charging signal actually carries usable information about
degradation (beats a trivial mean-curve baseline via a simple linear probe).
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.synthetic_battery import generate_cells, split_cells, BatteryICDataset  # noqa: E402


def small_cfg(n_cells=12):
    return {
        "data": {
            "n_cells": n_cells,
            "cycles_per_cell_min": 15,
            "cycles_per_cell_max": 25,
            "charge_len_T": 120,
            "ic_grid_M": 50,
            "voltage_grid_min": 3.0,
            "voltage_grid_max": 4.2,
            "c_rate_min": 1.0,
            "c_rate_max": 4.0,
            "train_frac_cells": 0.6,
            "val_frac_cells": 0.2,
            "noise_std_ic": 0.02,
            "noise_std_signal": 0.01,
        }
    }


def test_cell_count_matches_paper_stated_53():
    cfg = small_cfg(n_cells=53)
    cells = generate_cells(cfg, seed=1)
    assert len(cells) == 53


def test_shapes():
    cfg = small_cfg()
    cells = generate_cells(cfg, seed=1)
    for c in cells:
        n = len(c["cycles"])
        assert c["charge"].shape == (n, 120, 3)
        assert c["ic"].shape == (n, 50)
        assert c["soh"].shape == (n,)
        assert c["resistance"].shape == (n,)


def test_no_nans_or_infs():
    cfg = small_cfg()
    cells = generate_cells(cfg, seed=2)
    for c in cells:
        assert np.isfinite(c["charge"]).all()
        assert np.isfinite(c["ic"]).all()
        assert np.isfinite(c["soh"]).all()
        assert np.isfinite(c["resistance"]).all()


def test_soh_declines_with_cycle_index():
    """SOH should trend downward within a cell as cycles progress (fade)."""
    cfg = small_cfg()
    cells = generate_cells(cfg, seed=3)
    declining = 0
    for c in cells:
        if c["soh"][-1] < c["soh"][0]:
            declining += 1
    assert declining >= int(0.9 * len(cells)), "most cells should show net SOH fade over their cycle life"


def test_cell_level_split_is_disjoint_and_covers_all():
    cfg = small_cfg(n_cells=20)
    cells = generate_cells(cfg, seed=4)
    train, val, test = split_cells(cells, cfg, seed=4)
    train_ids = {c["cell_id"] for c in train}
    val_ids = {c["cell_id"] for c in val}
    test_ids = {c["cell_id"] for c in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids | val_ids | test_ids) == 20
    assert len(train) > 0 and len(val) > 0 and len(test) > 0


def test_dataset_normalization_uses_train_stats_only():
    cfg = small_cfg(n_cells=20)
    cells = generate_cells(cfg, seed=5)
    train, val, test = split_cells(cells, cfg, seed=5)
    train_ds = BatteryICDataset(train)
    val_ds = BatteryICDataset(val, mean=train_ds.mean, std=train_ds.std)
    assert np.allclose(val_ds.mean, train_ds.mean)
    assert np.allclose(val_ds.std, train_ds.std)
    # Train split itself should be ~zero-mean / unit-std after normalization.
    assert np.abs(train_ds.charge_norm.reshape(-1, 3).mean(axis=0)).max() < 0.05
    assert np.abs(train_ds.charge_norm.reshape(-1, 3).std(axis=0) - 1.0).max() < 0.05


def test_charging_signal_beats_mean_baseline_via_linear_probe():
    """
    Sanity check (the one called out in the task spec): a simple LINEAR
    PROBE from a flattened charging signal should predict SOH better than
    always predicting the training-set mean SOH. If this fails, the
    synthetic charging signal does not actually carry degradation
    information, and the whole dataset would be teaching the LSTM to fit
    noise.
    """
    cfg = small_cfg(n_cells=30)
    cfg["data"]["cycles_per_cell_min"] = 30
    cfg["data"]["cycles_per_cell_max"] = 50
    cells = generate_cells(cfg, seed=6)
    train, val, _ = split_cells(cells, cfg, seed=6)
    train_ds = BatteryICDataset(train)
    val_ds = BatteryICDataset(val, mean=train_ds.mean, std=train_ds.std)

    # Flatten (T,3) -> simple summary features: mean+std of each channel,
    # plus min/max, over the charging window. This is intentionally a
    # crude, cheap "linear probe" style featurization, not the LSTM itself.
    def featurize(charge_norm):
        mean_f = charge_norm.mean(axis=1)  # (N, 3)
        std_f = charge_norm.std(axis=1)    # (N, 3)
        return np.concatenate([mean_f, std_f], axis=1)  # (N, 6)

    X_train = featurize(train_ds.charge_norm)
    X_val = featurize(val_ds.charge_norm)
    y_train = train_ds.soh
    y_val = val_ds.soh

    # Closed-form least-squares linear regression (with bias term).
    X_train_b = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
    X_val_b = np.concatenate([X_val, np.ones((X_val.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
    pred_val = X_val_b @ coef

    probe_mse = np.mean((pred_val - y_val) ** 2)
    baseline_mse = np.mean((y_train.mean() - y_val) ** 2)

    assert probe_mse < baseline_mse, (
        f"linear probe (mse={probe_mse:.5f}) did not beat mean-SOH baseline "
        f"(mse={baseline_mse:.5f}) -- charging signal may not carry degradation info"
    )
