"""
verify.py
=========

Empirical verification of Theorem 2: Post-Softmax Rank Expansion.

We construct explicit rank-1 matrices and verify that their row-wise
softmax has strictly higher rank.

Requires: NumPy (no PyTorch needed — pure linear algebra)

Usage:
    python empirical/verify.py
    python -m pytest tests/ -v
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Section 1: Core functions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TheoremResult:
    name: str
    passed: bool
    metric: float
    detail: str


def numerical_rank(mat: np.ndarray, tol: float = 1e-10) -> int:
    """Return the numerical rank of a matrix via SVD.

    The rank is the number of singular values above the tolerance.
    """
    if mat.size == 0:
        return 0
    # For small matrices, full SVD is cheap and exact.
    s = np.linalg.svd(mat, compute_uv=False)
    return int(np.sum(s > tol))


def rowwise_softmax(mat: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax."""
    max_per_row = np.max(mat, axis=1, keepdims=True)
    exp_mat = np.exp(mat - max_per_row)
    sum_exp = np.sum(exp_mat, axis=1, keepdims=True)
    return exp_mat / sum_exp


# ---------------------------------------------------------------------------
# Section 2: Explicit constructions
# ---------------------------------------------------------------------------

def make_counterexample_mxnx(
    m: int,
    n: int,
    *,
    a: float = 1.0,
    b: float = 2.0,
) -> np.ndarray:
    """Return an m×n rank-1 matrix where σ(Z) has rank min(m, n).

    Construction: u_i = a*i, v_j = b*j, Z = u v^T.
    Then σ(Z)_{ij} ∝ (r_i)^j with r_i = exp(ab*i), forming a
    generalized Vandermonde matrix with full rank min(m, n).
    """
    i = np.arange(m, dtype=np.float64)
    j = np.arange(n, dtype=np.float64)
    u = a * i
    v = b * j
    return np.outer(u, v)


def make_counterexample_2x2(a: float = 2.0, b: float = 3.0) -> np.ndarray:
    """Return the canonical 2×2 rank-1 counterexample.
    Special case of make_counterexample_mxnx with m=n=2.
    """
    return make_counterexample_mxnx(2, 2, a=a, b=b)


# ---------------------------------------------------------------------------
# Section 3: Theorem checks
# ---------------------------------------------------------------------------

def check_theorem_2_1_2x2(a: float = 2.0, b: float = 3.0) -> TheoremResult:
    """Verify Theorem 2.1: explicit 2×2 rank-1 → rank-2."""
    Z = make_counterexample_2x2(a, b)
    rank_Z = numerical_rank(Z)
    S = rowwise_softmax(Z)
    rank_S = numerical_rank(S)

    passed = (rank_Z == 1) and (rank_S == 2)
    detail = (
        f"Z = [[0, 0], [0, {a*b:.1f}]], rank(Z)={rank_Z}, "
        f"rank(σ(Z))={rank_S}"
    )
    return TheoremResult(
        name="Theorem 2.1: 2×2 Rank-1 → Rank-2",
        passed=passed,
        metric=float(rank_S - rank_Z),
        detail=detail,
    )


def check_theorem_2_2_general(
    dims: list[Tuple[int, int]],
    seed: int = 42,
) -> list[TheoremResult]:
    """Verify Theorem 2.2 for multiple dimensions."""
    results = []
    for m, n in dims:
        Z = make_counterexample_mxnx(m, n, seed=seed)
        rank_Z = numerical_rank(Z)
        S = rowwise_softmax(Z)
        rank_S = numerical_rank(S)
        expected = min(m, n)

        passed = (rank_Z == 1) and (rank_S == expected)
        results.append(TheoremResult(
            name=f"Theorem 2.2 [{m}×{n}]: Rank-1 → Rank-{expected}",
            passed=passed,
            metric=float(rank_S - rank_Z),
            detail=(
                f"rank(Z)={rank_Z}, rank(σ(Z))={rank_S}, "
                f"expected={expected}"
            ),
        ))
    return results


def verify_singular_values(
    mat: np.ndarray,
    expected_rank: int,
    tol: float = 1e-10,
) -> Tuple[bool, np.ndarray, str]:
    """Check that the singular values support the claimed rank."""
    s = np.linalg.svd(mat, compute_uv=False)
    actual = int(np.sum(s > tol))
    passed = actual == expected_rank
    detail = (
        f"Singular values: {s[:expected_rank+2]}, "
        f"rank={actual}, expected={expected_rank}"
    )
    return passed, s, detail


# ---------------------------------------------------------------------------
# Section 4: Main runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print(" Theorem 2: Post-Softmax Rank Expansion")
    print(" Empirical Verification")
    print("=" * 70)
    print()

    np.set_printoptions(precision=6, suppress=True)

    # ---- 2×2 counterexample ------------------------------------------------
    print("-" * 70)
    print("THEOREM 2.1 — Explicit 2×2 Counterexample")
    print("-" * 70)

    Z = make_counterexample_2x2(a=2.0, b=3.0)
    print(f"\nZ = [[0, 0], [0, 6.0]]  (rank-1 outer product)")
    print(f"Z =\n{Z}")

    rank_Z = numerical_rank(Z)
    print(f"\nnumerical_rank(Z) = {rank_Z}")

    S = rowwise_softmax(Z)
    print(f"\nσ(Z) =\n{S}")

    rank_S = numerical_rank(S)
    print(f"numerical_rank(σ(Z)) = {rank_S}")

    passed_sv, sv, sv_detail = verify_singular_values(S, expected_rank=2)
    print(f"\nSingular values of σ(Z): {sv}")
    print(f"SVD verification: {'PASS' if passed_sv else 'FAIL'} — {sv_detail}")

    # Manual check: row proportionality
    row0, row1 = S[0], S[1]
    ratios = row1 / row0
    print(f"\nRow 0: {row0}")
    print(f"Row 1: {row1}")
    print(f"Row1/Row0 ratio: {ratios}")
    print(f"Ratios equal? {np.allclose(ratios, ratios[0])}")

    # ---- General m×n cases ----------------------------------------------
    print("\n" + "=" * 70)
    print("THEOREM 2.2 — General m×n Cases")
    print("=" * 70)

    DIMS = [(3, 3), (4, 5), (5, 4), (8, 10), (10, 8)]
    all_passed = True
    for m, n in DIMS:
        Z = make_counterexample_mxnx(m, n)
        S = rowwise_softmax(Z)
        rz = numerical_rank(Z)
        rs = numerical_rank(S)
        expected = min(m, n)
        ok = (rz == 1) and (rs == expected)
        status = "PASS" if ok else "FAIL"
        print(f"\n{m}×{n}: rank(Z)={rz}, rank(σ(Z))={rs}, expected={expected}  →  {status}")
        all_passed = all_passed and ok

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    res_21 = check_theorem_2_1_2x2()
    print(f"\n{'PASS' if res_21.passed else 'FAIL'} — {res_21.name}")
    print(f"  Rank jump: {res_21.metric:.0f}")
    print(f"  Detail: {res_21.detail}")

    res_22_list = check_theorem_2_2_general(DIMS)
    for res in res_22_list:
        print(f"\n{'PASS' if res.passed else 'FAIL'} — {res.name}")
        print(f"  Rank jump: {res.metric:.0f}")
        print(f"  Detail: {res.detail}")

    overall = res_21.passed and all(r.passed for r in res_22_list)
    print("\n" + "=" * 70)
    print(f"OVERALL: {'ALL PASS' if overall else 'SOME FAILED'}")
    print("=" * 70)

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
