"""
src/ldpc.py

From-scratch regular(ish) LDPC code: Gallager-style parity-check matrix
construction, systematic encoding, and a normalized min-sum belief-propagation
decoder that accepts an *external prior-LLR vector* (from the prediction path)
added in at each variable node alongside the channel LLR.

SOURCING DISCLOSURE: the paper does not publish its LDPC code's (n, k, wc, wr)
or its exact decoder variant. This file is this repo's own small, from-scratch
construction (n=256, k=128, rate 1/2) chosen only to be fast on CPU and to
exercise the real mechanism the paper describes: combining prediction-derived
prior LLRs with channel LLRs in an iterative decoder, retried after a first,
prediction-free attempt fails its CRC.

Construction method: Gallager's original 1962 method #1 for regular LDPC
codes -- build one "base" sub-matrix with an exact row-weight block structure
(so every column in it has weight 1), then stack `wc` copies of it with
independently permuted columns. This gives an (approximately) (wc, wr)-regular
H with exactly n*wc/wr rows by construction.

That raw H is not immediately systematic. We run Gaussian elimination over
GF(2) to rearrange it into H_sys = [P | I_m] (m = rank, k = n - m), which
also gives us a matching generator matrix G_sys = [I_k | P^T] for encoding.
NOTE: after elimination, the *systematic* H_sys is generally no longer
exactly (wc, wr)-regular (row-additions during elimination change row
weights) -- this is normal and does not affect correctness of BP, which only
needs the sparse (possibly irregular) adjacency structure of H_sys, not
exact regularity.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _build_base_pcm(n: int, wc: int, wr: int, seed: int) -> np.ndarray:
    """Gallager method #1 regular-LDPC base construction. Returns H: [m, n] in {0,1}."""
    if n % wr != 0:
        raise ValueError("n must be divisible by wr for this base construction "
                          "(each row-block sub-matrix must tile the n columns exactly)")
    sub_rows = n // wr
    m = sub_rows * wc

    # Base sub-matrix: row i has a contiguous block of `wr` ones -> each column
    # appears in exactly one row -> column weight 1 within this sub-block.
    H1 = np.zeros((sub_rows, n), dtype=np.uint8)
    for row in range(sub_rows):
        H1[row, row * wr:(row + 1) * wr] = 1

    rng = np.random.default_rng(seed)
    blocks = [H1]
    for _ in range(wc - 1):
        perm = rng.permutation(n)
        blocks.append(H1[:, perm])
    H = np.vstack(blocks).astype(np.uint8)  # [m, n], column weight == wc, row weight == wr
    return H


def _gf2_rref(H: np.ndarray):
    """Gaussian elimination over GF(2). Returns (rref, pivot_cols, rank)."""
    A = H.copy() % 2
    m, n = A.shape
    pivot_cols = []
    row = 0
    for col in range(n):
        pivot_row = None
        for r in range(row, m):
            if A[r, col] == 1:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        if pivot_row != row:
            A[[row, pivot_row]] = A[[pivot_row, row]]
        # Eliminate this column from every other row (full reduction, not just below).
        mask = (A[:, col] == 1)
        mask[row] = False
        A[mask] = (A[mask] + A[row]) % 2
        pivot_cols.append(col)
        row += 1
        if row == m:
            break
    rank = row
    return A, pivot_cols, rank


@dataclass
class LDPCCode:
    n: int
    k: int
    H: np.ndarray          # [m, n] systematic parity-check matrix (permuted column order)
    G: np.ndarray          # [k, n] systematic generator matrix, G @ H.T = 0 (mod 2)
    perm: np.ndarray       # column permutation applied relative to the raw Gallager construction
    row_edges: list        # row_edges[c] = list of variable indices connected to check c
    col_edges: list        # col_edges[v] = list of check indices connected to variable v

    @property
    def rate(self) -> float:
        return self.k / self.n

    def encode(self, message_bits: np.ndarray) -> np.ndarray:
        """message_bits: [k] -> codeword: [n] = message @ G (mod 2). First k bits are systematic."""
        assert message_bits.shape[-1] == self.k
        return (message_bits.astype(np.uint8) @ self.G) % 2

    def syndrome(self, codeword_bits: np.ndarray) -> np.ndarray:
        return (self.H.astype(np.int64) @ codeword_bits.astype(np.int64)) % 2


def build_ldpc_code(n: int = 256, k_target: int = 128, wc: int = 3, wr: int = 6,
                     seed: int = 1234) -> LDPCCode:
    """
    Construct a rate ~ (n-m)/n regular LDPC code.

    IMPORTANT (real bug hit + fix, disclosed here and in the README): an earlier
    version of this function derived the decoder's parity-check matrix directly
    from the fully row-reduced (systematic) form [P | I_m]. That is
    mathematically valid (same null space / same set of codewords) but
    DESTROYS SPARSITY: Gaussian elimination on a sparse random-like matrix
    causes heavy fill-in (row weights observed to explode from ~8 to 60-70+),
    and belief propagation on a near-dense "low density" parity-check matrix
    performs very poorly (way more short cycles, far weaker per-edge
    reliability) -- in testing this actually made BP *increase* the number of
    bit errors instead of correcting them.

    Fix: keep two matrices instead of one.
      - `H` (used for decoding and the syndrome check) is the ORIGINAL sparse
        Gallager-constructed matrix (only column-permuted, never row-reduced),
        so it keeps its exact construction-time row/column weights.
      - `G` (used only for encoding) is derived via Gaussian elimination on a
        *separate* working copy. Row-reducing a matrix never changes its (right)
        null space, so codewords satisfying the dense eliminated form also
        satisfy the original sparse H -- G is valid for `H` even though it was
        computed from a densified intermediate.

    k_target is only a hint (it determines wc/wr upstream); the achieved
    k = n - rank(H_raw) can differ slightly from k_target when the raw
    Gallager construction isn't full rank (common for small block-structured
    codes) -- callers must use the returned `code.k`, not assume k_target exactly.
    """
    H_raw = _build_base_pcm(n, wc, wr, seed)  # [m0, n], sparse: row weight wr, col weight wc, exactly

    # Elimination is only used here to (a) find the rank/message length and
    # (b) find a systematic column split -- its OUTPUT MATRIX is discarded
    # for decoding purposes; only `pivot_cols` and `rank` are kept from it.
    H_rref, pivot_cols, rank = _gf2_rref(H_raw)
    non_pivot_cols = [c for c in range(n) if c not in set(pivot_cols)]
    perm = np.array(non_pivot_cols + pivot_cols, dtype=np.int64)  # message cols first, then "identity" cols

    m = rank
    k = n - m

    # G is derived from the densified systematic form (fine: it's only used for
    # the O(k*n) encoding matmul, not for iterative decoding).
    H_full_rref = H_rref[:rank]        # [m, n], the (dense) row-reduced independent rows
    H_sys_dense = H_full_rref[:, perm]  # [m, n] = [P | I_m]
    P = H_sys_dense[:, :k]               # [m, k]
    I_k = np.eye(k, dtype=np.uint8)
    G = np.concatenate([I_k, P.T], axis=1)  # [k, n] = [I_k | P^T], in the permuted column order

    # The DECODER's parity-check matrix: original sparse H_raw, columns permuted
    # to match G's bit ordering, all m0 rows kept (including any that are linearly
    # dependent on the others -- harmless redundant checks, not used for rank).
    H_sparse = H_raw[:, perm]  # [m0, n], sparse: row weight wr, col weight wc preserved exactly

    # Sanity: every one of G's basis codewords must satisfy every row of H_sparse,
    # including the possibly-redundant ones (guaranteed since row-reduction preserves
    # the null space, but we check it explicitly rather than assume it).
    check = (G.astype(np.int64) @ H_sparse.astype(np.int64).T) % 2
    if check.any():
        raise RuntimeError("LDPC construction failed internal consistency check (G @ H_sparse^T != 0)")

    m0 = H_sparse.shape[0]
    row_edges = [list(np.nonzero(H_sparse[c])[0]) for c in range(m0)]
    col_edges = [list(np.nonzero(H_sparse[:, v])[0]) for v in range(n)]

    return LDPCCode(n=n, k=k, H=H_sparse.astype(np.uint8), G=G.astype(np.uint8),
                     perm=perm, row_edges=row_edges, col_edges=col_edges)


def bp_decode(code: LDPCCode, channel_llr: np.ndarray, prior_llr: np.ndarray = None,
              max_iters: int = 30, alpha: float = 0.75):
    """
    Normalized min-sum belief-propagation decoding.

    channel_llr: [n] LLR from the channel observation.
    prior_llr:   [n] optional prediction-derived prior LLR (zeros if None) -- this is
                 the mechanism the paper describes: a probabilistic motion prediction
                 contributes a prior belief about each bit, combined with the channel
                 LLR before/through decoding.
    alpha: normalized min-sum scaling factor (compensates min-sum's known
           overestimation of check-node reliability vs. exact sum-product; 0.75 is a
           common textbook default).

    Returns (decoded_bits [n] uint8, success: bool, num_iters_used: int).
    """
    n = code.n
    m = code.H.shape[0]
    H_mask = code.H.astype(bool)  # [m, n]

    L0 = channel_llr.astype(np.float64).copy()
    if prior_llr is not None:
        L0 = L0 + prior_llr.astype(np.float64)
    L0 = np.clip(L0, -20.0, 20.0)

    # Variable-to-check messages, initialized to the combined prior LLR on every edge.
    M_vc = np.where(H_mask, L0[None, :], 0.0)  # [m, n]

    col_idx = np.arange(n)

    for it in range(1, max_iters + 1):
        # ---- Check node update (normalized min-sum), fully vectorized over [m, n] ----
        abs_vc = np.abs(M_vc)
        abs_masked = np.where(H_mask, abs_vc, np.inf)  # [m, n]

        min1 = abs_masked.min(axis=1)                  # [m]
        argmin1 = abs_masked.argmin(axis=1)             # [m]
        abs_masked2 = abs_masked.copy()
        abs_masked2[np.arange(m), argmin1] = np.inf
        min2 = abs_masked2.min(axis=1)                  # [m]

        # magnitude excluding self: min1 everywhere except at argmin1 column, where it's min2
        is_argmin = (col_idx[None, :] == argmin1[:, None])  # [m, n]
        excl_mag = np.where(is_argmin, min2[:, None], min1[:, None])  # [m, n]
        excl_mag = np.where(np.isfinite(excl_mag), excl_mag, 0.0)

        # sign product excluding self, via masked prefix/suffix cumulative products
        # (avoids division-by-zero issues when a message is exactly 0).
        sign_for_prod = np.where(H_mask, np.sign(M_vc), 1.0)
        sign_for_prod = np.where(sign_for_prod == 0.0, 1.0, sign_for_prod)  # treat exact-0 as +1
        ones_col = np.ones((m, 1))
        prefix = np.cumprod(np.concatenate([ones_col, sign_for_prod[:, :-1]], axis=1), axis=1)
        suffix = np.cumprod(
            np.concatenate([sign_for_prod[:, 1:], ones_col], axis=1)[:, ::-1], axis=1
        )[:, ::-1]
        excl_sign = prefix * suffix  # [m, n], product of all other signs in the row

        M_cv = alpha * excl_sign * excl_mag
        M_cv = np.where(H_mask, M_cv, 0.0)

        # ---- Variable node update ----
        colsum = M_cv.sum(axis=0)  # [n], sum over all checks touching each variable
        total_llr = np.clip(L0 + colsum, -30.0, 30.0)  # [n]

        M_vc = L0[None, :] + colsum[None, :] - M_cv
        M_vc = np.where(H_mask, np.clip(M_vc, -30.0, 30.0), 0.0)

        # ---- Hard decision + syndrome check (early stop) ----
        x_hat = (total_llr < 0).astype(np.uint8)
        syn = code.syndrome(x_hat)
        if not syn.any():
            return x_hat, True, it

    return x_hat, False, max_iters
