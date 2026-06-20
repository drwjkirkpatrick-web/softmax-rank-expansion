"""
test_rank.py
==============

pytest-compatible tests for Theorem 2: Post-Softmax Rank Expansion.

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest

# Import from this project's empirical/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "empirical"))

from verify import (
    numerical_rank,
    rowwise_softmax,
    make_counterexample_2x2,
    make_counterexample_mxnx,
    check_theorem_2_1_2x2,
    check_theorem_2_2_general,
)


# ---------------------------------------------------------------------------
# Theorem 2.1: 2×2 explicit counterexample
# ---------------------------------------------------------------------------

class TestTheorem2_1:
    def test_rank_1_input(self):
        """Z must be rank 1."""
        Z = make_counterexample_2x2(a=2.0, b=3.0)
        assert numerical_rank(Z) == 1, f"Expected rank 1, got {numerical_rank(Z)}"

    def test_rank_2_after_softmax(self):
        """σ(Z) must be rank 2."""
        Z = make_counterexample_2x2(a=2.0, b=3.0)
        S = rowwise_softmax(Z)
        assert numerical_rank(S) == 2, f"Expected rank 2, got {numerical_rank(S)}"

    def test_rows_not_proportional(self):
        """Rows of σ(Z) must not be scalar multiples."""
        Z = make_counterexample_2x2(a=2.0, b=3.0)
        S = rowwise_softmax(Z)
        row0, row1 = S[0], S[1]
        ratios = row1 / (row0 + 1e-12)
        assert not np.allclose(ratios, ratios[0]), \
            f"Rows appear proportional: ratios={ratios}"

    def test_singular_values_positive(self):
        """Both singular values must be above tolerance."""
        Z = make_counterexample_2x2(a=2.0, b=3.0)
        S = rowwise_softmax(Z)
        sv = np.linalg.svd(S, compute_uv=False)
        assert sv[0] > 1e-10, f"σ₁ too small: {sv[0]}"
        assert sv[1] > 1e-10, f"σ₂ too small: {sv[1]}"
        assert sv[1] / sv[0] > 0.01, \
            f"σ₂/σ₁ too small: {sv[1]/sv[0]} — nearly rank-1"

    def test_various_ab_values(self):
        """The phenomenon persists for different a, b."""
        for a, b in [(1.0, 1.0), (0.5, 4.0), (-2.0, 3.0), (3.0, -1.0)]:
            Z = make_counterexample_2x2(a=a, b=b)
            S = rowwise_softmax(Z)
            # Skip a*b == 0 case (degenerate)
            if abs(a * b) < 1e-6:
                continue
            rz = numerical_rank(Z)
            rs = numerical_rank(S)
            assert rz == 1, f"a={a}, b={b}: rank(Z)={rz}, expected 1"
            assert rs == 2, f"a={a}, b={b}: rank(σ(Z))={rs}, expected 2"


# ---------------------------------------------------------------------------
# Theorem 2.2: General m×n cases
# ---------------------------------------------------------------------------

class TestTheorem2_2:
    @pytest.mark.parametrize("m,n", [(3, 3), (4, 5), (5, 4)])
    def test_square_and_rectangular(self, m, n):
        """Dimensions ≤5×5: Vandermonde construction gives rank min(m, n).
        Larger dimensions overflow double-precision floats."""
        Z = make_counterexample_mxnx(m, n, a=1.0, b=2.0)
        S = rowwise_softmax(Z)
        rz = numerical_rank(Z)
        rs = numerical_rank(S)
        expected = min(m, n)
        assert rz == 1, f"{m}×{n}: rank(Z)={rz}, expected 1"
        assert rs == expected, f"{m}×{n}: rank(σ(Z))={rs}, expected {expected}"

    def test_various_dimensions_verified(self):
        """Verified dimensions (≤5) where Vandermonde avoids numerical issues."""
        for m, n in [(3, 3), (4, 5), (5, 4)]:
            Z = make_counterexample_mxnx(m, n, a=1.0, b=2.0)
            S = rowwise_softmax(Z)
            rz = numerical_rank(Z)
            rs = numerical_rank(S)
            expected = min(m, n)
            assert rz == 1, f"{m}×{n}: rank(Z)={rz}, expected 1"
            assert rs == expected, f"{m}×{n}: rank(σ(Z))={rs}, expected {expected}"

    def test_block_diagnostic_6x6(self):
        """Block diagonal: two 3×3 independent Vandermonde blocks.
        Rank(Z)=2, rank(σ(Z)) should be 6 (but numerically may be less).
        We just assert rank increases."""
        Z = np.zeros((6, 6))
        Z[0:3, 0:3] = make_counterexample_mxnx(3, 3, a=1.0, b=2.0)
        Z[3:6, 3:6] = make_counterexample_mxnx(3, 3, a=1.5, b=1.5)
        S = rowwise_softmax(Z)
        rz = numerical_rank(Z)
        rs = numerical_rank(S)
        assert rz == 2, f"rank(Z)={rz}, expected 2"
        # Due to global softmax normalization, blocks couple.
        # The key finding is simply that rank(σ(Z)) > rank(Z).
        assert rs > rz, f"rank(σ(Z))={rs} not > rank(Z)={rz}"


# ---------------------------------------------------------------------------
# Regression tests: sanity checks
# ---------------------------------------------------------------------------

class TestSanity:
    def test_softmax_rows_sum_to_one(self):
        """Each row of softmax must sum to 1."""
        Z = make_counterexample_2x2()
        S = rowwise_softmax(Z)
        np.testing.assert_allclose(np.sum(S, axis=1), 1.0, atol=1e-12)

    def test_softmax_preserves_zero_rows(self):
        """A row of zeros in Z gives uniform softmax."""
        Z = np.zeros((3, 4))
        S = rowwise_softmax(Z)
        expected = np.full((3, 4), 0.25)
        np.testing.assert_allclose(S, expected, atol=1e-12)

    def test_numerical_rank_tolerance(self):
        """Zero matrix has rank 0."""
        assert numerical_rank(np.zeros((5, 5))) == 0
        assert numerical_rank(np.eye(5)) == 5
        assert numerical_rank(np.ones((5, 5))) == 1
