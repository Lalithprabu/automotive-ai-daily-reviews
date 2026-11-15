"""
src/bsm.py

Synthetic Basic Safety Message (BSM) format for this reconstruction.

SOURCING DISCLOSURE: SAE J2735 defines the real BSM standard with dozens of
fields and a specific ASN.1 encoding. The paper (arXiv:2609.25609) says only
that it operates on "the standard BSM message format" -- it gives no bit
widths, field list, or serialization details. Everything in this file is
this repo's own minimal reconstruction: five continuous kinematic fields
(x, y, speed, heading, accel), each fixed-point quantized to a documented
bit width and range (see config.yaml), concatenated into a 60-bit payload,
zero-padded to a 64-bit message so it fits neatly as the systematic message
portion of the rate-1/2, n=256 LDPC code in src/ldpc.py.

Bit ordering convention (must match models/bit_llr.py):
  - Within a field: bit 0 is the LSB, bit (B-1) is the MSB.
  - Across fields, the 64-bit message is the concatenation, in this order:
        x (16b) | y (16b) | speed (10b) | heading (10b) | accel (8b) | pad (4b, always 0)
    with each field's bits listed LSB-first inside its span.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple

FIELD_ORDER = ["x", "y", "speed", "heading", "accel"]

DEFAULT_FIELD_SPECS: Dict[str, Tuple[float, float, int]] = {
    "x":       (-500.0, 500.0, 16),
    "y":       (-500.0, 500.0, 16),
    "speed":   (0.0, 50.0, 10),
    "heading": (-np.pi, np.pi, 10),
    "accel":   (-10.0, 10.0, 8),
}

PAYLOAD_BITS = sum(b for _, _, b in DEFAULT_FIELD_SPECS.values())  # 60
MESSAGE_BITS = 64  # zero-padded to 64 for a clean systematic split in the LDPC code
assert PAYLOAD_BITS <= MESSAGE_BITS


def field_step(low: float, high: float, bits: int) -> float:
    """LSB step size for a field's fixed-point quantizer."""
    levels = (2 ** bits) - 1
    return (high - low) / levels


def quantize_field(value: np.ndarray, low: float, high: float, bits: int) -> np.ndarray:
    """Continuous value(s) -> unsigned integer code in [0, 2**bits - 1], clipped to range."""
    value = np.clip(value, low, high)
    step = field_step(low, high, bits)
    q = np.round((value - low) / step).astype(np.int64)
    return np.clip(q, 0, (2 ** bits) - 1)


def dequantize_field(q: np.ndarray, low: float, high: float, bits: int) -> np.ndarray:
    """Unsigned integer code -> reconstructed continuous value (bin center)."""
    step = field_step(low, high, bits)
    return low + q.astype(np.float64) * step


def int_to_bits_lsb_first(q: np.ndarray, bits: int) -> np.ndarray:
    """
    Vectorized unsigned-int -> bit array, LSB first.
    q: [...] integer array. Returns [..., bits] uint8 array.
    """
    shifts = np.arange(bits, dtype=np.int64)
    # Shape: [..., 1] >> [bits] via broadcasting -> [..., bits]
    out = (q[..., None] >> shifts) & 1
    return out.astype(np.uint8)


def bits_lsb_first_to_int(bits_arr: np.ndarray) -> np.ndarray:
    """Inverse of int_to_bits_lsb_first. bits_arr: [..., B] -> [...] integer."""
    B = bits_arr.shape[-1]
    weights = (1 << np.arange(B, dtype=np.int64))
    return (bits_arr.astype(np.int64) * weights).sum(axis=-1)


@dataclass
class BSMState:
    """One BSM's worth of continuous kinematic state."""
    x: float
    y: float
    speed: float
    heading: float
    accel: float


def encode_bsm(state: BSMState, field_specs: Dict[str, Tuple[float, float, int]] = None) -> np.ndarray:
    """
    BSMState -> 64-bit message (numpy uint8 array of 0/1), LSB-first per field,
    fields concatenated in FIELD_ORDER, zero-padded to MESSAGE_BITS.
    """
    field_specs = field_specs or DEFAULT_FIELD_SPECS
    bit_chunks = []
    for name in FIELD_ORDER:
        low, high, bits = field_specs[name]
        val = np.array(getattr(state, name), dtype=np.float64)
        q = quantize_field(val, low, high, bits)
        bit_chunks.append(int_to_bits_lsb_first(q, bits))
    payload = np.concatenate(bit_chunks, axis=-1)  # [60]
    pad = np.zeros(MESSAGE_BITS - payload.shape[-1], dtype=np.uint8)
    return np.concatenate([payload, pad], axis=-1)  # [64]


def decode_bsm(message_bits: np.ndarray, field_specs: Dict[str, Tuple[float, float, int]] = None) -> BSMState:
    """Inverse of encode_bsm: 64-bit message -> BSMState (reconstructed, quantized values)."""
    field_specs = field_specs or DEFAULT_FIELD_SPECS
    offset = 0
    values = {}
    for name in FIELD_ORDER:
        low, high, bits = field_specs[name]
        chunk = message_bits[offset:offset + bits]
        q = bits_lsb_first_to_int(chunk)
        values[name] = float(dequantize_field(np.array(q), low, high, bits))
        offset += bits
    return BSMState(**values)


def field_bit_spans(field_specs: Dict[str, Tuple[float, float, int]] = None) -> Dict[str, Tuple[int, int]]:
    """Field name -> (start_bit_index, num_bits) within the 64-bit message."""
    field_specs = field_specs or DEFAULT_FIELD_SPECS
    spans = {}
    offset = 0
    for name in FIELD_ORDER:
        _, _, bits = field_specs[name]
        spans[name] = (offset, bits)
        offset += bits
    return spans
