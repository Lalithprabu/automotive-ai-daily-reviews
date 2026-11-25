"""Synthetic battery-aging generator: a stand-in for the BatteryLife benchmark used by arXiv 2611.19116.

The paper forecasts a cell's FULL state-of-health (SOH) trajectory, down to end of life (EOL, SOH = 0.8), from its
first charge-discharge cycles, on aging-condition-exclusive folds.  BatteryLife is not bundled here, so this module
simulates cells whose aging is driven by three textbook mechanisms (my construction, not the paper's):

    SEI growth / loss of lithium inventory (LLI)   q_sei(n) = k_sei * sqrt(n)       faster when hot, deeper DoD
    loss of active material (LAM)                  q_lam(n) = k_lam * n             faster at high C-rate, deep DoD
    lithium-plating "knee"                         q_knee(n) = k_knee * relu(n - n_k)^2
        the knee starts when accumulated LAM reaches a cell-specific threshold theta  ->  n_k = theta / k_lam
        plating is stronger when cold and at high charge C-rate

    SOH(n) = 1 - q_sei - q_lam - q_knee + small noise          lifetime L = first n with SOH <= 0.8

Aging condition = (temperature T, charge C-rate, depth of discharge DoD): 4 x 4 x 3 = 48 conditions, 6 cells each.
Every cell also gets its own manufacturing variation (rate multipliers, knee threshold, resistance).

Per-cycle EARLY-LIFE features (what the model sees, first K = 100 cycles), F = 25 channels:
    5 scalar features      discharge capacity, internal resistance, mean discharge voltage, charge time, temp rise
    10 SOC-interval dQ/dV  differential capacity in 10 SOC bins: LAM flattens the mid-SOC peaks (an early,
                           capacity-independent signature of k_lam -> knee timing)
    10 SOC-interval V      mean discharge voltage per SOC bin: LLI shifts the curve along SOC, resistance lowers it
These SOC-interval statistics are my stand-in for the paper's "state-of-charge interval statistics".  The toy is
CONSTRUCTED so that they carry information about the knee that capacity alone does not; see README.
"""
from __future__ import annotations

import numpy as np

TEMPS = [15.0, 25.0, 35.0, 45.0]
CRATES = [0.5, 1.0, 2.0, 3.0]
DODS = [0.6, 0.8, 1.0]
K_EARLY = 100                                   # early-life cycles given to the model
N_MAX = 3000                                    # longest life simulated / predicted
NB = 10                                         # SOC bins
SOC_C = (np.arange(NB) + 0.5) / NB              # bin centres
F_SCALAR = ["capacity", "resistance", "v_mean", "t_charge", "temp_rise"]
N_FEAT = len(F_SCALAR) + 2 * NB


def conditions() -> list[tuple[float, float, float]]:
    return [(t, c, d) for t in TEMPS for c in CRATES for d in DODS]


def ocv(soc: np.ndarray) -> np.ndarray:
    """Generic NMC-like open-circuit-voltage curve (V) with two plateaus (-> two dQ/dV peaks)."""
    s = np.clip(soc, 0.0, 1.0)
    return (3.0 + 1.1 * s + 0.12 * np.tanh((s - 0.3) / 0.05) + 0.10 * np.tanh((s - 0.65) / 0.06)
            - 0.25 * np.exp(-s / 0.04))


# --------------------------------------------------------------------------------------------- one cell
def make_cell(cond: tuple[float, float, float], seed: int) -> dict:
    """Hidden aging parameters of one cell under condition (T, C, DoD)."""
    T, C, D = cond
    r = np.random.default_rng(seed)
    arr = np.exp(0.045 * (T - 25.0))                                   # Arrhenius-like heat factor
    k_sei = 2.2e-3 * arr * D ** 0.6 * r.lognormal(0, 0.15)
    k_lam = 2.4e-5 * (0.4 + C) ** 1.1 * D * r.lognormal(0, 0.25)
    cold = np.exp(-0.05 * (T - 15.0))                                  # plating worse when cold
    k_knee = 3.0e-7 * cold * (0.3 + C) ** 1.5 * r.lognormal(0, 0.3)
    theta = 0.030 * r.lognormal(0, 0.30)                               # LAM fraction at which the knee starts
    return dict(cond=cond, k_sei=k_sei, k_lam=k_lam, k_knee=k_knee, theta=theta, n_k=theta / k_lam,
                r0=0.030 * r.lognormal(0, 0.08), seed=seed)


def soh_curve(cell: dict, n: np.ndarray) -> np.ndarray:
    """Noise-free SOH at cycles n (array)."""
    q = (cell["k_sei"] * np.sqrt(n) + cell["k_lam"] * n
         + cell["k_knee"] * np.maximum(0.0, n - cell["n_k"]) ** 2)
    return 1.0 - q


def lifetime(cell: dict) -> float:
    """First (fractional) cycle with SOH <= 0.8, capped at N_MAX."""
    n = np.arange(0, N_MAX + 1, dtype=float)
    s = soh_curve(cell, n)
    idx = np.nonzero(s <= 0.8)[0]
    if len(idx) == 0:
        return float(N_MAX)
    i = idx[0]
    return float(n[i - 1] + (s[i - 1] - 0.8) / (s[i - 1] - s[i]))     # linear interpolation


def cycle_features(cell: dict, n_cycles: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Early-life per-cycle features (n_cycles, N_FEAT) and the per-cycle discharge V(SOC) curves (n_cycles, 50)."""
    T, C, D = cell["cond"]
    n = np.arange(1, n_cycles + 1, dtype=float)
    q_sei = cell["k_sei"] * np.sqrt(n)
    lam = cell["k_lam"] * n                                            # LAM fraction
    soh = soh_curve(cell, n) + rng.normal(0, 6e-4, n_cycles)           # measured capacity (normalised)
    res = cell["r0"] * (1 + 4.0 * q_sei + 8.0 * lam) * np.exp(-0.012 * (T - 25)) + rng.normal(0, 2e-4, n_cycles)
    i_dis = 1.0                                                        # 1C discharge
    feats, curves = [], []
    soc_fine = np.linspace(0, 1, 50)
    for j in range(n_cycles):
        shift = 1.2 * q_sei[j]                                         # LLI shifts the stoichiometry window
        stretch = 1.0 - 6.0 * lam[j]                                   # LAM shrinks / flattens the plateaus
        s_eff = np.clip(0.5 + (soc_fine - 0.5) * stretch + shift, 0, 1)
        v = ocv(s_eff) - i_dis * res[j] + rng.normal(0, 1.5e-3, 50)
        curves.append(v)
        # SOC-interval statistics
        v_bins = np.array([v[(soc_fine >= b / NB) & (soc_fine <= (b + 1) / NB)].mean() for b in range(NB)])
        dv = np.gradient(v, soc_fine)
        dqdv_bins = np.array([1.0 / np.clip(dv[(soc_fine >= b / NB) & (soc_fine <= (b + 1) / NB)].mean(), 0.2, 50)
                              for b in range(NB)])
        dqdv_bins *= 1.0 + rng.normal(0, 0.004, NB)
        v_mean = v.mean()
        t_charge = 60.0 / C * D * soh[j] * (1 + 2.0 * res[j]) + rng.normal(0, 0.3)
        temp_rise = 2.0 * C ** 2 * res[j] / 0.03 * np.exp(-0.01 * (T - 25)) + rng.normal(0, 0.1)
        feats.append(np.concatenate([[soh[j], res[j], v_mean, t_charge, temp_rise], dqdv_bins, v_bins]))
    return np.asarray(feats, np.float32), np.asarray(curves, np.float32)


# normalisation constants (fixed, not fitted, so every fold uses the same transform)
F_MU = np.array([0.985, 0.035, 3.55, 40.0, 3.0] + [1.2] * NB + [3.5] * NB, np.float32)
F_SD = np.array([0.01, 0.01, 0.05, 30.0, 3.0] + [0.6] * NB + [0.35] * NB, np.float32)


def normalise(x: np.ndarray) -> np.ndarray:
    return (x - F_MU) / F_SD


# --------------------------------------------------------------------------------------------- dataset
def make_dataset(cells_per_cond: int = 6, seed: int = 0, traj_step: int = 10) -> dict:
    """All 48 x cells_per_cond cells.

    Returns dict with
        x      (M, K, F) float32 normalised early-life features
        cond   (M,) condition index,  cond_val (M, 3)
        life   (M,) lifetime (cycles to SOH 0.8)
        traj   (M, G) true noise-free SOH on the cycle grid `grid` (G = N_MAX / traj_step + 1), NaN after EOL
        grid   (G,) cycle numbers
        knee   (M,) True when the knee starts before EOL
    """
    conds = conditions()
    grid = np.arange(0, N_MAX + 1, traj_step, dtype=float)
    X, CI, CV, L, TR, KN = [], [], [], [], [], []
    for ci, cond in enumerate(conds):
        for k in range(cells_per_cond):
            cs = seed * 100_003 + ci * 101 + k
            cell = make_cell(cond, cs)
            f, _ = cycle_features(cell, K_EARLY, np.random.default_rng(cs + 7))
            life = lifetime(cell)
            tr = soh_curve(cell, grid)
            tr[grid > life] = np.nan
            X.append(normalise(f)); CI.append(ci); CV.append(cond); L.append(life); TR.append(tr)
            KN.append(cell["n_k"] < life)
    return dict(x=np.asarray(X, np.float32), cond=np.asarray(CI), cond_val=np.asarray(CV, np.float32),
                life=np.asarray(L, np.float32), traj=np.asarray(TR, np.float32), grid=grid.astype(np.float32),
                knee=np.asarray(KN))


def condition_folds(n_folds: int = 6, seed: int = 0) -> list[np.ndarray]:
    """Aging-condition-exclusive folds: each fold is a set of held-out CONDITION indices (no condition shared)."""
    r = np.random.default_rng(seed)
    idx = r.permutation(len(conditions()))
    return [np.sort(idx[f::n_folds]) for f in range(n_folds)]


if __name__ == "__main__":
    d = make_dataset()
    print({k: v.shape for k, v in d.items()})
    print("life: min %.0f median %.0f max %.0f" % (d["life"].min(), np.median(d["life"]), d["life"].max()))
    print("knee share %.2f, capped %d" % (d["knee"].mean(), (d["life"] >= N_MAX).sum()))
    for t in TEMPS:
        for c in CRATES:
            m = (d["cond_val"][:, 0] == t) & (d["cond_val"][:, 1] == c)
            print(t, c, "%.0f" % np.median(d["life"][m]), "knee %.2f" % d["knee"][m].mean())
