"""
src/channel.py

BPSK-over-AWGN channel simulation and channel-observation LLR computation.

Bit-to-symbol mapping: b=0 -> +1, b=1 -> -1 (standard BPSK).
Noise: r = s + n, n ~ N(0, sigma^2).

Eb/N0 -> sigma^2 conversion (standard, unit symbol energy Es=1):
    Eb = Es / R = 1 / R                      (R = code rate = k/n)
    N0 = Eb / (Eb/N0)_linear
    sigma^2 = N0 / 2

Channel LLR for BPSK/AWGN (derivation: log[N(r;+1,sigma^2)/N(r;-1,sigma^2)]):
    LLR(r) = 2*r / sigma^2
  (LLR > 0 favors b=0, LLR < 0 favors b=1, consistent with our LLR = log(P(b=0)/P(b=1)) convention.)
"""
from __future__ import annotations

import numpy as np


def eb_n0_db_to_sigma2(eb_n0_db: float, code_rate: float) -> float:
    eb_n0_linear = 10.0 ** (eb_n0_db / 10.0)
    eb = 1.0 / code_rate  # Es = 1
    n0 = eb / eb_n0_linear
    sigma2 = n0 / 2.0
    return sigma2


def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    """0 -> +1, 1 -> -1."""
    return 1.0 - 2.0 * bits.astype(np.float64)


def awgn_transmit(bits: np.ndarray, eb_n0_db: float, code_rate: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    bits: [n] array of 0/1 (a full codeword).
    Returns received real-valued samples r: [n].
    """
    sigma2 = eb_n0_db_to_sigma2(eb_n0_db, code_rate)
    s = bits_to_bpsk(bits)
    noise = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=bits.shape)
    return s + noise


def received_to_llr(r: np.ndarray, eb_n0_db: float, code_rate: float) -> np.ndarray:
    """Channel-observation LLR vector from received samples r: [n] -> LLR: [n]."""
    sigma2 = eb_n0_db_to_sigma2(eb_n0_db, code_rate)
    return 2.0 * r / sigma2


def hard_decision_from_llr(llr: np.ndarray) -> np.ndarray:
    """LLR > 0 -> bit 0, LLR < 0 -> bit 1 (LLR = log P(0)/P(1))."""
    return (llr < 0).astype(np.uint8)
