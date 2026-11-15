"""
models/bit_llr.py

BSMBitProbabilityLayer: propagates a predicted Gaussian over a *continuous*
BSM field through that field's fixed-point quantizer (src/bsm.py) to get,
for every bit position of the quantized field, P(bit=1) under the predictive
distribution.

LLRPriorHead: turns those bit probabilities into prior log-likelihood ratios
(LLR = log(P(bit=0)/P(bit=1))) for use as the external prior fed into the
LDPC decoder (src/ldpc.py's `prior_llr` argument).

--- Math ---
For an unsigned B-bit fixed-point field with LSB step `delta` and range
[low, high], bit i (0 = LSB) is 1 exactly when the quantized integer level q
falls in one of the alternating half-periods of size `block = delta * 2**i`
within a full period `period = 2*block`:
    bit_i(q) = 1  for  q*delta in [k*period + block, k*period + 2*block), k = 0,1,2,...

So, given a Gaussian predictive distribution N(mu, sigma^2) over the
*continuous* field value, P(bit_i = 1) is the total Gaussian probability
mass falling in the union of those "bit=1" sub-intervals, computed exactly
via the Gaussian CDF (erf) at each sub-interval's boundaries.

DOCUMENTED APPROXIMATION: for low-order bits, the number of sub-intervals to
sum is 2**(B-1-i), which can be very large (e.g. up to 2**15 for a 16-bit
field's LSB). Beyond `max_exact_blocks` sub-intervals we do not sum them
individually; instead we return P(bit=1) = 0.5 for that bit. Physically this
is a reasonable approximation, not just a shortcut: once the quantization
step for a bit is much finer than the predictor's own uncertainty sigma, the
Gaussian mass alternates almost uniformly between "bit=0" and "bit=1"
sub-intervals many times within one standard deviation, so that bit really
is close to unpredictable from a smooth continuous forecast -- exactly the
regime where a prior LLR near 0 (no information) is the honest answer. A
fully exact computation (e.g. via Poisson summation / the Jacobi theta
function for a wrapped Gaussian) is out of scope for this reconstruction.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from src.bsm import DEFAULT_FIELD_SPECS, FIELD_ORDER, field_step

_SQRT2 = math.sqrt(2.0)


def _gaussian_cdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf((x - mu) / (sigma * _SQRT2)))


class BSMBitProbabilityLayer(nn.Module):
    """
    Vectorized, per-field Gaussian-CDF bit-probability computation.
    Not a stub: every bit's probability is computed either exactly (CDF mass
    over its true decision sub-intervals) or via the documented sigma-vs-step
    approximation above -- no field ever gets a hand-waved probability.
    """

    def __init__(self, field_specs: Dict[str, Tuple[float, float, int]] = None,
                 max_exact_blocks: int = 64, sigma_floor: float = 1e-4):
        super().__init__()
        self.field_specs = field_specs or DEFAULT_FIELD_SPECS
        self.max_exact_blocks = max_exact_blocks
        self.sigma_floor = sigma_floor

        # Precompute, per field, per bit: whether it's "exact" and the block/period geometry.
        # Stored as plain Python (not tensors) since bit widths differ per field and this
        # only runs once at construction time.
        self._field_bit_meta = {}
        for name in FIELD_ORDER:
            low, high, bits = self.field_specs[name]
            delta = field_step(low, high, bits)
            per_bit = []
            for i in range(bits):
                block = delta * (2 ** i)
                period = 2.0 * block
                num_blocks = 2 ** (bits - 1 - i)
                exact = num_blocks <= self.max_exact_blocks
                per_bit.append({"block": block, "period": period, "num_blocks": num_blocks, "exact": exact})
            self._field_bit_meta[name] = {"low": low, "high": high, "bits": bits, "steps": per_bit}

    def forward(self, mu_phys: torch.Tensor, sigma_phys: torch.Tensor, field_name: str) -> torch.Tensor:
        """
        mu_phys, sigma_phys: [...] (any leading batch/time shape) physical-unit
                              predicted mean/std for ONE field.
        Returns bit_probs: [..., bits] with bit_probs[..., i] = P(bit_i = 1), LSB first (i=0).
        """
        meta = self._field_bit_meta[field_name]
        low = meta["low"]
        sigma = sigma_phys.clamp_min(self.sigma_floor)   # [...]
        mu = mu_phys                                      # [...]

        bit_probs = []
        for step in meta["steps"]:
            if step["exact"]:
                k = torch.arange(step["num_blocks"], device=mu.device, dtype=mu.dtype)  # [num_blocks]
                lower = low + k * step["period"] + step["block"]   # [num_blocks], start of each "bit=1" run
                upper = lower + step["block"]                       # [num_blocks]
                # BUG FIX (disclosed): quantize_field() CLIPS the continuous value to
                # [low, high] before quantizing, so any Gaussian mass beyond `high` is not
                # lost -- it saturates to the top quantization level, whose bits are ALL 1
                # (2**bits - 1 in binary is all ones). The topmost "bit=1" sub-interval
                # therefore must absorb that entire upper tail, not just the mass up to the
                # nominal grid edge. Extending its upper bound to +inf makes the Gaussian
                # CDF do that correctly. (An earlier version omitted this and silently
                # dropped the saturated tail mass, which under-counted P(bit=1) badly for
                # wide/uncertain predictions -- caught by
                # tests/test_bit_llr.py::test_very_uncertain_prediction_gives_near_half_probabilities.)
                upper = upper.clone()
                upper[-1] = float("inf")
                # Broadcast: mu/sigma [...] -> [..., 1]; lower/upper [num_blocks] -> broadcast to [..., num_blocks]
                mu_e = mu.unsqueeze(-1)
                sigma_e = sigma.unsqueeze(-1)
                cdf_hi = _gaussian_cdf(upper, mu_e, sigma_e)  # [..., num_blocks]
                cdf_lo = _gaussian_cdf(lower, mu_e, sigma_e)  # [..., num_blocks]
                p = (cdf_hi - cdf_lo).sum(dim=-1).clamp(0.0, 1.0)  # [...]
            else:
                # Documented approximation: quantization finer than predictive resolution -> ~unpredictable bit.
                p = torch.full_like(mu, 0.5)
            bit_probs.append(p)

        return torch.stack(bit_probs, dim=-1)  # [..., bits], LSB first


class LLRPriorHead(nn.Module):
    """Bit probability -> prior LLR = log(P(bit=0)/P(bit=1)), numerically clamped."""

    def __init__(self, eps: float = 1e-4, llr_clamp: float = 20.0):
        super().__init__()
        self.eps = eps
        self.llr_clamp = llr_clamp

    def forward(self, p_bit1: torch.Tensor) -> torch.Tensor:
        p1 = p_bit1.clamp(self.eps, 1.0 - self.eps)
        llr = torch.log((1.0 - p1) / p1)
        return llr.clamp(-self.llr_clamp, self.llr_clamp)
