import numpy as np

from src.channel import eb_n0_db_to_sigma2, bits_to_bpsk, awgn_transmit, received_to_llr, hard_decision_from_llr


def test_bits_to_bpsk_mapping():
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    s = bits_to_bpsk(bits)
    assert np.allclose(s, [1.0, -1.0, 1.0, -1.0])


def test_sigma2_decreases_with_higher_ebn0():
    s_low = eb_n0_db_to_sigma2(-2.0, code_rate=0.5)
    s_high = eb_n0_db_to_sigma2(8.0, code_rate=0.5)
    assert s_high < s_low


def test_llr_high_snr_is_confident_and_correct_sign():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=200).astype(np.uint8)
    r = awgn_transmit(bits, eb_n0_db=12.0, code_rate=0.5, rng=rng)
    llr = received_to_llr(r, eb_n0_db=12.0, code_rate=0.5)
    hard = hard_decision_from_llr(llr)
    # At very high SNR, hard decisions should match the transmitted bits almost always.
    mismatch_rate = (hard != bits).mean()
    assert mismatch_rate < 0.02
