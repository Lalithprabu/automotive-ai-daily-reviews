import numpy as np
import pytest

from src.bsm import (
    BSMState, encode_bsm, decode_bsm, field_bit_spans,
    DEFAULT_FIELD_SPECS, FIELD_ORDER, MESSAGE_BITS, field_step,
)


def test_message_length():
    s = BSMState(x=0, y=0, speed=0, heading=0, accel=0)
    bits = encode_bsm(s)
    assert bits.shape == (MESSAGE_BITS,)
    assert bits.dtype == np.uint8
    assert set(np.unique(bits)).issubset({0, 1})


@pytest.mark.parametrize("state", [
    BSMState(x=123.456, y=-45.2, speed=27.8, heading=1.234, accel=-2.1),
    BSMState(x=-499.9, y=499.9, speed=0.0, heading=-3.14, accel=9.99),
    BSMState(x=0.0, y=0.0, speed=25.0, heading=0.0, accel=0.0),
    BSMState(x=499.9, y=-499.9, speed=49.9, heading=3.14, accel=-9.99),
])
def test_encode_decode_roundtrip_within_quantization_error(state):
    bits = encode_bsm(state)
    recovered = decode_bsm(bits)
    for name in FIELD_ORDER:
        low, high, nbits = DEFAULT_FIELD_SPECS[name]
        step = field_step(low, high, nbits)
        orig = getattr(state, name)
        reco = getattr(recovered, name)
        # Reconstruction error must be within one LSB step (quantization bound).
        assert abs(orig - reco) <= step + 1e-9, f"{name}: {orig} vs {reco} (step={step})"


def test_padding_bits_are_zero():
    s = BSMState(x=1.0, y=2.0, speed=3.0, heading=0.5, accel=-1.0)
    bits = encode_bsm(s)
    payload_bits = sum(b for _, _, b in DEFAULT_FIELD_SPECS.values())
    assert not bits[payload_bits:].any()


def test_field_bit_spans_cover_payload_exactly_once():
    spans = field_bit_spans()
    covered = np.zeros(MESSAGE_BITS, dtype=bool)
    for name, (start, nbits) in spans.items():
        assert not covered[start:start + nbits].any(), f"overlap in field {name}"
        covered[start:start + nbits] = True
    payload_bits = sum(b for _, _, b in DEFAULT_FIELD_SPECS.values())
    assert covered.sum() == payload_bits


def test_clipping_out_of_range_values():
    # Values far outside the declared range must clip, not raise or wrap.
    s = BSMState(x=1e6, y=-1e6, speed=1e6, heading=100.0, accel=-1e6)
    bits = encode_bsm(s)
    recovered = decode_bsm(bits)
    assert recovered.x == pytest.approx(DEFAULT_FIELD_SPECS["x"][1], abs=1.0)
    assert recovered.speed == pytest.approx(DEFAULT_FIELD_SPECS["speed"][1], abs=1.0)
