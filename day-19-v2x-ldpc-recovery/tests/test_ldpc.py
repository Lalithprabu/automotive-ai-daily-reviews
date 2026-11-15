"""
tests/test_ldpc.py

Covers:
  - LDPC construction sanity (G @ H^T == 0, achieved rate near target)
  - encode/decode round trip with no noise
  - BP decoder actually corrects bit-flip errors (no external prior)
  - THE EMPIRICAL CORE CLAIM OF THE PAPER'S MECHANISM: at a fixed, challenging
    Eb/N0, decoding WITH an accurate-but-imperfect prior LLR succeeds strictly
    more often, over many trials, than decoding with channel-LLR alone. This
    mirrors (not reproduces) the paper's claim that prediction-derived prior
    LLRs recover messages that fail a prediction-free decode.
"""
import numpy as np
import pytest

from src.ldpc import build_ldpc_code, bp_decode
from src.channel import awgn_transmit, received_to_llr


@pytest.fixture(scope="module")
def code():
    return build_ldpc_code(n=256, k_target=128, wc=4, wr=8, seed=1234)


def test_construction_sanity(code):
    assert code.n == 256
    assert 100 <= code.k <= 150  # achieved k may differ slightly from k_target due to rank deficiency
    check = (code.G.astype(np.int64) @ code.H.astype(np.int64).T) % 2
    assert not check.any(), "G @ H^T must be all-zero (mod 2) for a valid systematic (G,H) pair"


def test_encode_produces_valid_codeword(code):
    rng = np.random.default_rng(0)
    for _ in range(20):
        msg = rng.integers(0, 2, size=code.k).astype(np.uint8)
        cw = code.encode(msg)
        assert cw.shape == (code.n,)
        syn = code.syndrome(cw)
        assert not syn.any(), "encoded codeword must satisfy H @ c = 0 (mod 2)"
        # systematic: first k bits of the codeword equal the message
        assert np.array_equal(cw[:code.k], msg)


def test_bp_decode_roundtrip_no_noise(code):
    rng = np.random.default_rng(1)
    msg = rng.integers(0, 2, size=code.k).astype(np.uint8)
    cw = code.encode(msg)
    # Very high SNR -> channel LLR should point strongly at the true bits.
    r = awgn_transmit(cw, eb_n0_db=10.0, code_rate=code.rate, rng=rng)
    llr = received_to_llr(r, eb_n0_db=10.0, code_rate=code.rate)
    decoded, ok, iters = bp_decode(code, llr, max_iters=30)
    assert ok
    assert np.array_equal(decoded, cw)


def test_bp_decode_corrects_bit_flips_without_prior(code):
    rng = np.random.default_rng(2)
    msg = rng.integers(0, 2, size=code.k).astype(np.uint8)
    cw = code.encode(msg)
    corrupted_llr = np.full(code.n, 5.0)  # start as if all bits confidently 0
    corrupted_llr = np.where(cw == 1, -5.0, corrupted_llr)  # confident, correct LLR for every bit
    # Now flip a handful of bits' LLR sign (simulate channel errors) -- a small number
    # of errors should still be correctable by belief propagation alone.
    flip_positions = rng.choice(code.n, size=4, replace=False)
    corrupted_llr[flip_positions] *= -1

    decoded, ok, iters = bp_decode(code, corrupted_llr, max_iters=30)
    assert ok, "BP should correct a small number of bit errors with no external prior"
    assert np.array_equal(decoded, cw)


def test_prior_llr_improves_recovery_rate_at_challenging_snr(code):
    """
    THE EMPIRICAL CORE CLAIM: with an informative (but not perfect) prior LLR,
    the decoder recovers strictly more messages than with channel-LLR alone,
    at a fixed challenging Eb/N0. This is this repo's own reconstruction of the
    paper's headline mechanism -- NOT a reproduction of the paper's 87.27%
    figure (that number was measured on the paper's own dataset/code).
    """
    rng = np.random.default_rng(123)
    eb_n0_db = 0.5  # deliberately harsh -- channel-alone decode should fail often
    n_trials = 150
    prior_confidence = 3.5      # LLR magnitude representing a "fairly confident" prediction
    prior_field_coverage = 0.75  # fraction of bits the "predictor" actually has an opinion about

    succ_no_prior = 0
    succ_with_prior = 0
    for _ in range(n_trials):
        msg = rng.integers(0, 2, size=code.k).astype(np.uint8)
        cw = code.encode(msg)
        r = awgn_transmit(cw, eb_n0_db=eb_n0_db, code_rate=code.rate, rng=rng)
        llr = received_to_llr(r, eb_n0_db=eb_n0_db, code_rate=code.rate)

        _, ok0, _ = bp_decode(code, llr, max_iters=30)
        succ_no_prior += int(ok0)

        # Simulate an imperfect-but-informative prior: correct direction on
        # `prior_field_coverage` of bits, informed by the true bits (proxy for what a
        # well-trained motion predictor + BSMBitProbabilityLayer would produce for a
        # temporally-correlated message), silent (LLR=0) elsewhere.
        prior = np.zeros(code.n)
        mask = rng.random(code.n) < prior_field_coverage
        prior[mask] = np.where(cw[mask] == 0, prior_confidence, -prior_confidence)

        _, ok1, _ = bp_decode(code, llr, prior_llr=prior, max_iters=30)
        succ_with_prior += int(ok1)

    rate_no_prior = succ_no_prior / n_trials
    rate_with_prior = succ_with_prior / n_trials

    print(f"\n[empirical core test] Eb/N0={eb_n0_db}dB, n_trials={n_trials}: "
          f"no-prior success={succ_no_prior}/{n_trials} ({rate_no_prior:.1%}), "
          f"with-prior success={succ_with_prior}/{n_trials} ({rate_with_prior:.1%})")

    assert rate_with_prior > rate_no_prior, (
        f"prior-aided decoding ({rate_with_prior:.1%}) should beat channel-only "
        f"decoding ({rate_no_prior:.1%}) at this Eb/N0"
    )
