import torch
import pytest

from models.bit_llr import BSMBitProbabilityLayer, LLRPriorHead
from src.bsm import DEFAULT_FIELD_SPECS, FIELD_ORDER


@pytest.fixture(scope="module")
def layer():
    return BSMBitProbabilityLayer(max_exact_blocks=64)


@pytest.mark.parametrize("field_name", FIELD_ORDER)
def test_bit_prob_shape_and_range(layer, field_name):
    low, high, bits = DEFAULT_FIELD_SPECS[field_name]
    mu = torch.tensor([0.0, (low + high) / 2, high])
    sigma = torch.tensor([1.0, 0.5, (high - low) * 0.05])
    p = layer(mu, sigma, field_name)
    assert p.shape == (3, bits)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)


def test_very_confident_prediction_gives_near_deterministic_bits(layer):
    # A near-zero sigma should push most bit probabilities close to 0 or 1
    # (the quantized value is essentially known), not stuck at 0.5 everywhere.
    low, high, bits = DEFAULT_FIELD_SPECS["accel"]
    mu = torch.tensor([2.345])
    sigma = torch.tensor([1e-3])
    p = layer(mu, sigma, "accel")[0]
    # For a sharply peaked Gaussian, each bit's probability should be near 0 or 1
    # for all bits well inside the "exact" regime (accel has only 8 bits, all exact
    # since max blocks = 2**7=128 > max_exact_blocks=64 only for bit 0... check both extremes).
    extreme = ((p < 0.05) | (p > 0.95)).float().mean().item()
    assert extreme > 0.5, f"expected most bits to be confidently resolved, got fraction={extreme}"


def test_very_uncertain_prediction_gives_near_half_probabilities(layer):
    low, high, bits = DEFAULT_FIELD_SPECS["speed"]
    mu = torch.tensor([25.0])
    sigma = torch.tensor([(high - low) * 5.0])  # huge uncertainty relative to range
    p = layer(mu, sigma, "speed")[0]
    assert torch.allclose(p, torch.full_like(p, 0.5), atol=0.05)


def test_llr_prior_head_sign_and_clamp():
    head = LLRPriorHead(llr_clamp=20.0)
    p = torch.tensor([0.0, 0.5, 1.0, 0.999999, 0.000001])
    llr = head(p)
    assert llr.shape == p.shape
    assert torch.all(torch.isfinite(llr))
    assert torch.all(llr.abs() <= 20.0)
    # p near 0 (bit is very unlikely to be 1) -> large positive LLR (favors bit=0)
    assert llr[0] > 0
    # p near 1 (bit is very likely to be 1) -> large negative LLR (favors bit=1)
    assert llr[2] < 0
    # p == 0.5 -> LLR should be ~0
    assert abs(llr[1].item()) < 1e-4
