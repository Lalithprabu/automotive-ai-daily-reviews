"""
data/synthetic_battery.py
==========================================================================
SYNTHETIC dataset generator for the ChargeIC-LSTM reconstruction project.

DISCLOSURE (read this before trusting anything downstream):
The source paper (arXiv:2609.22843) gives NO dataset files, NO real
cell-cycling data, and NO numeric hyperparameters in the abstract we were
able to retrieve. Everything in this file is THIS PROJECT'S OWN synthetic
physics-inspired simulator, built only to exercise the ChargeIC-LSTM
architecture end-to-end. The only paper-sourced fact reused here is the
cell count (53 cells), which the abstract states explicitly
("a comprehensive dataset of 53 battery cells cycled under diverse
fast-charging protocols"). Everything else -- fade-rate ranges, resistance
growth model, IC-curve shape, charge-signal shape, noise levels, voltage
grid bounds -- is a reconstruction default, NOT taken from the paper.

Physics-inspired design (informal, not paper-verified):
  - Each cell gets a fixed fast-charge C-rate in [1C, 4C], a cell-specific
    capacity fade rate, and a cell-specific internal-resistance (R) growth
    rate. This gives genuine cell-to-cell heterogeneity so that a model
    trained on some cells must generalize to unseen cells (matching the
    abstract's claim of generalizing to "unseen battery data").
  - SOH(k) = 1 - fade_rate * k + noise, clipped to a realistic EOL floor.
  - R(k) = R0 + r_growth * k + noise (monotonic resistance growth).
  - The CHARGING signal (voltage, current, temperature over the fast-charge
    CC-then-taper window) is generated so it actually CARRIES INFORMATION
    about SOH(k) and R(k):
      * Higher R(k) -> the CC (constant-current) phase reaches the voltage
        cutoff sooner (shorter/steeper CC segment before tapering off into
        the CV-like tail) -- this is the real "IR-drop shortens CC time"
        effect fast-charging engineers rely on.
      * Temperature rise is modeled as roughly proportional to I^2 * R(k),
        smoothed by a simple thermal-mass (exponential) filter.
  - The ground-truth DISCHARGE IC curve (dQ/dV vs V) is modeled as a
    Gaussian bump on a small baseline:
      * peak height  ~ SOH(k)^p           (fades with degradation)
      * peak voltage ~ V0 - k_shift*R(k)  (shifts down as resistance grows)
      * peak width slowly broadens with R(k) (mild, common qualitative
        trend in real IC curves as internal resistance rises)
    This is a stylized stand-in for a real dQ/dV curve, good enough to
    give the network a genuine, learnable, non-trivial regression target
    correlated with the charging-phase input -- but it is NOT a real
    battery IC curve and must never be reported as one.

Units are arbitrary/normalized (not claimed to match real cell datasheets):
  - capacity in "Ah-like" units, current in "A-like" units, voltage in V,
    temperature in deg C offset from ambient.
==========================================================================
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Core per-cycle physics-inspired sample generator
# ---------------------------------------------------------------------------
def _generate_cycle_sample(
    c_rate: float,
    soh: float,
    resistance: float,
    T: int,
    M: int,
    v_grid: np.ndarray,
    rng: np.random.Generator,
    noise_std_signal: float,
    noise_std_ic: float,
    q_nom: float = 2.0,
    v_start: float = 3.0,
    v_cutoff: float = 4.2,
    t_amb: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate ONE (charging_signal, discharge_ic_curve) pair for a single
    cycle of a single cell.

    Returns
    -------
    charge_signal : np.ndarray, shape (T, 3)   columns = [voltage, current, temperature]
    ic_curve      : np.ndarray, shape (M,)      dQ/dV values on the fixed voltage grid
    """
    t_frac = np.linspace(0.0, 1.0, T)  # (T,) normalized progress through the charge window

    # --- CC-phase fraction shrinks as internal resistance grows -----------
    # Reconstruction default: higher R -> earlier IR-drop-triggered cutoff
    # -> shorter constant-current segment before the taper begins.
    k_r = 0.06  # reconstruction default sensitivity coefficient
    f_cc = np.clip(0.78 - k_r * resistance, 0.30, 0.85)  # scalar

    # --- Current profile: CC then exponential taper ("CV-like") -----------
    i_cc = c_rate * q_nom  # constant-current magnitude
    current = np.where(
        t_frac < f_cc,
        i_cc,
        i_cc * np.exp(-5.0 * (t_frac - f_cc) / max(1.0 - f_cc, 1e-3)),
    )
    current = current + rng.normal(0.0, noise_std_signal * i_cc, size=T)  # measurement noise

    # --- Voltage profile: rises through CC phase, ~plateaus in taper ------
    # Steeper (front-loaded) rise for higher R via gamma < 1.
    gamma = np.clip(1.0 - 0.05 * resistance, 0.55, 1.0)
    cc_progress = np.clip(t_frac / max(f_cc, 1e-3), 0.0, 1.0)
    voltage_cc = v_start + (v_cutoff - v_start) * np.power(cc_progress, gamma)
    voltage = np.where(t_frac < f_cc, voltage_cc, v_cutoff + rng.normal(0, 0.003, size=T))
    voltage = voltage + rng.normal(0.0, noise_std_signal * 0.5, size=T)
    voltage = np.clip(voltage, v_start - 0.05, v_cutoff + 0.05)

    # --- Temperature profile: heating ~ I^2 * R, smoothed by thermal mass -
    heat_input = (current ** 2) * resistance * 0.015  # reconstruction-default scaling
    temperature = np.empty(T)
    temp_state = t_amb
    alpha = 0.15  # thermal-mass smoothing factor (reconstruction default)
    for i in range(T):
        temp_state = temp_state + alpha * (heat_input[i] - 0.4 * (temp_state - t_amb))
        temperature[i] = temp_state
    temperature = temperature + rng.normal(0.0, noise_std_signal * 2.0, size=T)

    charge_signal = np.stack([voltage, current, temperature], axis=-1).astype(np.float32)  # (T, 3)

    # --- Ground-truth discharge IC curve (dQ/dV vs V) ----------------------
    peak_height = 3.0 * (soh ** 2.0)                       # fades with degradation (p=2, reconstruction default)
    peak_voltage = np.clip(3.85 - 0.03 * resistance, v_start + 0.1, v_cutoff - 0.1)  # shifts down with R
    peak_width = 0.05 + 0.004 * resistance                 # mild broadening with R
    baseline = 0.05 + 0.02 * (v_grid - v_grid.min())        # small linear baseline term

    ic_curve = baseline + peak_height * np.exp(
        -((v_grid - peak_voltage) ** 2) / (2.0 * peak_width ** 2)
    )
    ic_curve = ic_curve + rng.normal(0.0, noise_std_ic, size=M)
    ic_curve = np.clip(ic_curve, 0.0, None).astype(np.float32)  # dQ/dV is non-negative here

    return charge_signal, ic_curve


def generate_cells(cfg: dict, seed: int = 42) -> list[dict]:
    """
    Generate the full synthetic 53-cell dataset described in the module
    docstring.

    Returns a list of per-cell dicts:
        {
          "cell_id": int,
          "c_rate": float,
          "fade_rate": float,
          "r_growth": float,
          "cycles": np.ndarray[int]  (n_cycles,)
          "charge": np.ndarray (n_cycles, T, 3)
          "ic":     np.ndarray (n_cycles, M)
          "soh":    np.ndarray (n_cycles,)
          "resistance": np.ndarray (n_cycles,)
        }
    """
    rng = np.random.default_rng(seed)
    d = cfg["data"]
    T, M = d["charge_len_T"], d["ic_grid_M"]
    v_grid = np.linspace(d["voltage_grid_min"], d["voltage_grid_max"], M)

    cells = []
    for cell_id in range(d["n_cells"]):
        c_rate = rng.uniform(d["c_rate_min"], d["c_rate_max"])
        n_cycles = rng.integers(d["cycles_per_cell_min"], d["cycles_per_cell_max"] + 1)

        # --- Shared per-cell "quality" latent factor ------------------------
        # Reconstruction design choice: a cell's capacity-fade rate and its
        # internal-resistance-growth rate are NOT drawn independently. Both
        # are driven partly by a common per-cell manufacturing/quality
        # factor (worse cells age faster in BOTH dimensions), plus their
        # own independent noise. Without this shared factor, the charging
        # signal (which only encodes R) and the SOH target (which only
        # depends on fade_rate) would be nearly uncorrelated ACROSS cells,
        # making SOH essentially unpredictable from the charging signal --
        # confirmed as a real issue by this project's own sanity check
        # (tests/test_data.py::test_charging_signal_beats_mean_baseline_via_linear_probe
        # failed before this correlation was added; see README implementation notes).
        quality = rng.uniform(0.0, 1.0)  # 0 = worst cell, 1 = best cell
        fade_rate = 0.0028 - quality * 0.0018 + rng.normal(0.0, 0.00025)   # worse quality -> faster fade
        fade_rate = float(np.clip(fade_rate, 0.0006, 0.0032))
        r0 = rng.uniform(0.03, 0.07)                                         # base internal resistance
        r_growth = 0.0062 - quality * 0.0044 + rng.normal(0.0, 0.0005)       # worse quality -> faster R growth
        r_growth = float(np.clip(r_growth, 0.0010, 0.0075))

        cycles = np.arange(n_cycles)
        soh = 1.0 - fade_rate * cycles + rng.normal(0.0, 0.004, size=n_cycles)
        soh = np.clip(soh, 0.55, 1.02)
        resistance = r0 + r_growth * cycles + rng.normal(0.0, 0.0008, size=n_cycles)
        resistance = np.clip(resistance, 0.02, None)

        charge = np.zeros((n_cycles, T, 3), dtype=np.float32)
        ic = np.zeros((n_cycles, M), dtype=np.float32)
        for k in range(n_cycles):
            cs, icv = _generate_cycle_sample(
                c_rate=c_rate,
                soh=float(soh[k]),
                resistance=float(resistance[k]),
                T=T,
                M=M,
                v_grid=v_grid,
                rng=rng,
                noise_std_signal=d["noise_std_signal"],
                noise_std_ic=d["noise_std_ic"],
            )
            charge[k] = cs
            ic[k] = icv

        cells.append(
            {
                "cell_id": cell_id,
                "c_rate": c_rate,
                "fade_rate": fade_rate,
                "r_growth": r_growth,
                "cycles": cycles,
                "charge": charge,
                "ic": ic,
                "soh": soh.astype(np.float32),
                "resistance": resistance.astype(np.float32),
                "v_grid": v_grid.astype(np.float32),
            }
        )
    return cells


def split_cells(cells: list[dict], cfg: dict, seed: int = 42):
    """
    Split by CELL INDEX (not cycle) so val/test cells are entirely unseen
    during training -- this is what lets us honestly evaluate
    "generalization ability... on unseen battery data" per the abstract.
    """
    d = cfg["data"]
    n = len(cells)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(round(d["train_frac_cells"] * n))
    n_val = int(round(d["val_frac_cells"] * n))

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train_cells = [cells[i] for i in train_idx]
    val_cells = [cells[i] for i in val_idx]
    test_cells = [cells[i] for i in test_idx]
    return train_cells, val_cells, test_cells


class BatteryICDataset(Dataset):
    """
    torch Dataset flattening a list of per-cell dicts (as returned by
    generate_cells / split_cells) into individual (cycle) samples.

    Each item returns:
      charge : FloatTensor (T, 3)   -- normalized [voltage, current, temperature]
      ic     : FloatTensor (M,)      -- ground-truth discharge IC curve
      soh    : FloatTensor ()        -- scalar ground-truth SOH fraction
    """

    def __init__(self, cells: list[dict], mean: np.ndarray | None = None, std: np.ndarray | None = None):
        charges, ics, sohs, cell_ids, cycle_idxs = [], [], [], [], []
        for c in cells:
            n = c["charge"].shape[0]
            charges.append(c["charge"])
            ics.append(c["ic"])
            sohs.append(c["soh"])
            cell_ids.extend([c["cell_id"]] * n)
            cycle_idxs.extend(list(c["cycles"]))

        self.charge = np.concatenate(charges, axis=0)  # (N, T, 3)
        self.ic = np.concatenate(ics, axis=0)            # (N, M)
        self.soh = np.concatenate(sohs, axis=0)           # (N,)
        self.cell_ids = np.array(cell_ids)
        self.cycle_idxs = np.array(cycle_idxs)

        # --- per-feature normalization stats -------------------------------
        # If mean/std are not supplied (i.e. this IS the training split),
        # compute them from this split and store for reuse at val/test/inference.
        if mean is None or std is None:
            self.mean = self.charge.reshape(-1, 3).mean(axis=0)  # (3,)
            self.std = self.charge.reshape(-1, 3).std(axis=0) + 1e-6  # (3,) avoid div-by-0
        else:
            self.mean = mean
            self.std = std

        self.charge_norm = (self.charge - self.mean) / self.std  # (N, T, 3)

    def __len__(self):
        return self.charge_norm.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(self.charge_norm[i]).float()   # (T, 3)
        y_ic = torch.from_numpy(self.ic[i]).float()           # (M,)
        y_soh = torch.tensor(self.soh[i]).float()              # scalar
        return x, y_ic, y_soh


if __name__ == "__main__":
    # Quick smoke test / sanity print when run directly (self-contained cfg,
    # does not depend on config.yaml so this file can be run standalone).
    cfg = {
        "data": {
            "n_cells": 5,
            "cycles_per_cell_min": 10,
            "cycles_per_cell_max": 15,
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
    cells = generate_cells(cfg, seed=0)
    tr, va, te = split_cells(cells, cfg, seed=0)
    ds = BatteryICDataset(tr)
    x, y_ic, y_soh = ds[0]
    print(f"n_cells={len(cells)} train={len(tr)} val={len(va)} test={len(te)}")
    print(f"charge sample shape={tuple(x.shape)} ic shape={tuple(y_ic.shape)} soh={float(y_soh):.3f}")
    print(f"dataset size (cycles): {len(ds)}")
