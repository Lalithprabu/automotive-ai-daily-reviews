"""Synthetic multi-chemistry Li-ion cell fleet: early-life capacity matrices + full SOH trajectories.

NOT MATR / HUST / MIX data.  A physics-flavoured toy built so that
  * early-cycle data carries REAL but INCOMPLETE information about lifetime (like Severson et al.'s
    Delta-Q(V) variance feature), and
  * part of the life (hidden manufacturing defect + knee timing) is NOT observable from the first
    100 cycles -> a good model must output a DISTRIBUTION of futures, which is FlowBatt's point.

Per cell latents
    chem   in {0: LFP-like, 1: NMC-like, 2: NCA-like}      -> base life, discharge-curve shape
    stress ~ U(0, 1)   fast-charge / temperature severity  (observable through early fade + Delta-Q)
    qual   ~ N(0, 1)   manufacturing quality                (observable, weakly, through Delta-Q shape)
    hidden ~ N(0, 1)   defect that only shows late          (NOT observable early)
    kappa  ~ U(.55,.85) knee position as a fraction of EOL  (NOT observable early)

    N_eol  = base[chem] * exp(-0.75 stress + 0.22 qual + 0.20 hidden)
    SOH(n) = 1 - d1 * n / N_k - d2 * ((n - N_k)_+ / (N_eol - N_k))^2 ,   N_k = kappa * N_eol
             d2 chosen so SOH(N_eol) = 0.80 exactly (80 % end-of-life)
"""
from __future__ import annotations

import numpy as np

H_CYC = 2550            # horizon in cycles
L = 256                 # SOH grid points
GRID = np.linspace(0, H_CYC, L)          # cycle of each grid point (10 cycles / point)
EARLY = 100             # early cycles in the capacity matrix
NV = 100                # voltage points
PREFIX = 10             # observed SOH prefix points (grid 0..9 = cycles 0..90)
EOL = 0.80
SOH_FLOOR = 0.55
V = np.linspace(0.0, 1.0, NV)            # normalised voltage axis (high -> low during discharge)
BASE_LIFE = np.array([1900.0, 1150.0, 800.0])
CHEM_NAMES = ("LFP", "NMC", "NCA")


def sample_latents(rng: np.random.Generator) -> dict:
    chem = int(rng.integers(0, 3))
    stress = float(rng.uniform(0, 1))
    qual = float(rng.normal())
    hidden = float(rng.normal())
    kappa = float(rng.uniform(0.55, 0.85))
    n_eol = BASE_LIFE[chem] * np.exp(-0.75 * stress + 0.22 * qual + 0.20 * hidden)
    n_eol = float(np.clip(n_eol, 260.0, 2450.0))
    d1 = float(np.clip(0.035 + 0.05 * stress + 0.01 * rng.normal(), 0.015, 0.10)) * kappa  # fade before knee
    return dict(chem=chem, stress=stress, qual=qual, hidden=hidden, kappa=kappa, n_eol=n_eol, d1=d1)


def soh_curve(lat: dict, n: np.ndarray) -> np.ndarray:
    """Noise-free SOH at cycles n."""
    n_eol, kappa, d1 = lat["n_eol"], lat["kappa"], lat["d1"]
    n_k = kappa * n_eol
    d2 = 0.20 - d1 * n_eol / n_k                      # > 0 because d1 / kappa <= 0.10
    lin = d1 * n / n_k
    knee = d2 * (np.clip(n - n_k, 0, None) / (n_eol - n_k)) ** 2
    return np.clip(1.0 - lin - knee, SOH_FLOOR, 1.0)


def discharge_shape(chem: int) -> np.ndarray:
    """Normalised cumulative discharge capacity Q(V)/Q_max on the voltage grid (0 at top of charge)."""
    if chem == 0:   # LFP: long flat plateau -> steep sigmoid in Q(V)
        return 1 / (1 + np.exp(-(V - 0.45) / 0.035))
    if chem == 1:   # NMC: sloped
        return 0.55 * V + 0.45 / (1 + np.exp(-(V - 0.6) / 0.08))
    return 0.75 * V ** 1.3 + 0.25 / (1 + np.exp(-(V - 0.35) / 0.06))


def capacity_matrix(lat: dict, soh_early: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(EARLY, NV) discharge-capacity-vs-voltage matrix for cycles 1..EARLY.

    The life information sits in a voltage-dependent distortion that grows with cycle number at a
    rate ~ 1/N_obs, where N_obs uses only the OBSERVABLE latents (chem, stress, qual) - not `hidden`
    and not `kappa`.  Its amplitude also has a qual-dependent location, so a CNN has to look at the
    shape, not only the mean.
    """
    chem, stress, qual = lat["chem"], lat["stress"], lat["qual"]
    n_obs = BASE_LIFE[chem] * np.exp(-0.75 * stress + 0.22 * qual)
    base = discharge_shape(chem)                                         # (NV,)
    n = np.arange(1, EARLY + 1)[:, None]                                 # (EARLY, 1)
    centre = 0.55 + 0.08 * np.tanh(qual)
    bump = np.exp(-0.5 * ((V - centre) / 0.09) ** 2)[None]               # (1, NV)
    amp = 9.0 * (n / EARLY) / n_obs                                       # Delta-Q grows ~ n / life
    Q = soh_early[:, None] * base[None] - amp * bump * base[None]
    Q = Q + 0.0015 * rng.normal(size=Q.shape) + 0.002 * rng.normal(size=(EARLY, 1))  # meas + per-cycle noise
    return Q.astype(np.float32)


def make_cell(rng: np.random.Generator) -> dict:
    lat = sample_latents(rng)
    soh = soh_curve(lat, GRID)
    # measured SOH: small noise + occasional rest-recovery spikes, as in real cycling data
    noise = 0.002 * rng.normal(size=L)
    spikes = (rng.random(L) < 0.03) * rng.uniform(0.002, 0.008, L)
    soh_meas = np.clip(soh + noise + spikes, SOH_FLOOR, 1.0)
    soh_meas[0] = 1.0
    soh_early = soh_curve(lat, np.arange(1, EARLY + 1).astype(float)) + 0.0015 * rng.normal(size=EARLY)
    cap = capacity_matrix(lat, soh_early, rng)
    return dict(cap=cap, soh=soh_meas.astype(np.float32), soh_clean=soh.astype(np.float32),
                prefix=soh_meas[:PREFIX].astype(np.float32), **lat)


def make_fleet(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cells = [make_cell(rng) for _ in range(n)]
    out = {k: np.stack([c[k] for c in cells]) for k in ("cap", "soh", "soh_clean", "prefix")}
    for k in ("chem", "stress", "qual", "hidden", "kappa", "n_eol", "d1"):
        out[k] = np.array([c[k] for c in cells])
    return out


def eol_cycle(soh: np.ndarray, thr: float = EOL) -> np.ndarray:
    """First crossing of `thr` (linear interpolation on the grid). soh (..., L) -> (...) cycles.
    Curves that never cross get the horizon."""
    soh = np.asarray(soh)
    below = soh < thr
    idx = np.where(below.any(-1), below.argmax(-1), L - 1)
    i0 = np.clip(idx - 1, 0, L - 1)
    s0 = np.take_along_axis(soh, i0[..., None], -1)[..., 0]
    s1 = np.take_along_axis(soh, idx[..., None], -1)[..., 0]
    frac = np.clip((s0 - thr) / np.maximum(s0 - s1, 1e-9), 0, 1)
    cyc = GRID[i0] + frac * (GRID[idx] - GRID[i0])
    return np.where(below.any(-1), cyc, H_CYC)
