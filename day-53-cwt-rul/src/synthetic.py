"""Synthetic CC-charge cycling data. Hidden fade rate r drives cycle life L ~ 0.2/r (with noise).
r leaks into early signals two ways: (a) a per-cell 'quality' latent (R0, Q0 correlate with log r, rho~0.6 -- same
trick as Day 18's lesson, otherwise single-cycle prediction would be impossible), and (b) drift of resistance /
capacity over the cycles, which only becomes visible when more cycles are observed.
Signals per cycle (T samples): voltage, current, cumulative charge. NOT real cell data.
"""
import numpy as np

T_DEFAULT = 128


def make_cells(n_cells=900, n_cycles=50, T=T_DEFAULT, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_cells)
    logr = np.log(0.0006) + (np.log(0.004) - np.log(0.0006)) * (0.5 + 0.5 * np.tanh(0.6 * z + 0.8 * rng.standard_normal(n_cells)))
    r = np.exp(logr)                                                          # fade per cycle
    life = 0.2 / r * np.exp(0.12 * rng.standard_normal(n_cells))              # cycles to 80% capacity
    q_lat = np.tanh(0.6 * (logr - logr.mean()) / logr.std() + 0.8 * rng.standard_normal(n_cells))
    R0 = 0.05 * (1 + 0.35 * q_lat)                                            # ohm-ish
    Q0 = 1.0 - 0.06 * q_lat                                                   # normalised capacity
    t = np.linspace(0, 1, T)
    X = np.zeros((n_cells, n_cycles, 3, T), np.float32)
    for k in range(n_cycles):
        Rk = R0 * (1 + 6.0 * r * k)                                           # resistance growth
        Qk = Q0 * (1 - 0.8 * r * k)                                           # capacity fade
        I = 1.0 + 0.01 * rng.standard_normal((n_cells, T))                    # CC-like current with ripple
        soc = np.clip(t[None] / Qk[:, None], 0, 1.2)
        ocv = 3.4 + 0.5 * soc + 0.12 * np.sin(2 * np.pi * 3 * soc) * (1 + 40 * r[:, None] * k) \
            + 0.1 * np.exp(-((soc - 0.45) / 0.05) ** 2) * (1 - 0.5 * r[:, None] * k * 20)  # plateau-feature fades
        V = ocv + I * Rk[:, None] * 2 + 0.004 * rng.standard_normal((n_cells, T))
        Q = np.cumsum(I, 1) / T
        X[:, k, 0], X[:, k, 1], X[:, k, 2] = V, I, Q
    return X, life.astype(np.float32), r.astype(np.float32)


def split_idx(n, val_frac, test_frac, seed=0):
    p = np.random.default_rng(seed + 1).permutation(n)                        # split by CELL, never by cycle
    nv, nt = int(n * val_frac), int(n * test_frac)
    return p[nv + nt:], p[:nv], p[nv: nv + nt]


def pick_cycles(n_window, max_in=4):
    """Indices of cycles shown to the model when it may use cycles 1..n_window (evenly spaced)."""
    return np.unique(np.linspace(0, n_window - 1, min(n_window, max_in)).round().astype(int))
