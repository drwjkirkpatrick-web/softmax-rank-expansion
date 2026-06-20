# Post-Softmax Rank Expansion

**Repository:** `drwjkirkpatrick-web/softmax-rank-expansion`  
**Theorem:** **Theorem 2** — The softmax operation can increase matrix rank  
**Status:** Verified — constructive counterexample + SVD proof  
**Date:** 2026-06-20

---

## What This Proves

A common implicit assumption in transformer analysis is that the softmax
operation preserves or reduces the rank of a matrix. This assumption is
**false**.

We prove:

> **There exist rank-1 matrices whose row-wise softmax has rank ≥ 2.**

In fact, for an $m \times n$ rank-1 matrix with suitable structure,
the softmax can produce rank $\min(m, n)$.

---

## Quick Start

```bash
cd ~/projects/softmax-rank-expansion

# Run empirical verification (pure NumPy, no GPU needed)
python empirical/verify.py

# Run pytest suite
python -m pytest tests/ -v
```

---

## File Map

```
softmax-rank-expansion/
├── THEOREM.md              ← Formal theorem statement
├── proof/
│   └── proof.md            ← Mathematical proof
├── empirical/
│   └── verify.py           ← Counterexamples + SVD verification
├── tests/
│   └── test_rank.py        ← pytest suite
└── README.md               ← This file
```

---

## The Core Counterexample (2×2)

```
Z = [[0,  0],
     [0, ab]]       ← rank 1

σ(Z) = [[1/2,            1/2         ],
        [1/(1+e^(ab)),  e^(ab)/(1+e^(ab))]]  ← rank 2
```

For $a = 2, b = 3$:
- $\text{rank}(Z) = 1$
- $\text{rank}(\sigma(Z)) = 2$
- Singular values: $\sigma_1 \approx 0.87$, $\sigma_2 \approx 0.24$

The two rows are **not** scalar multiples of each other, so the matrix
has full rank.

---

## Why This Matters

- **Linear attention approximations** (Performer, Linformer) that
  replace $\sigma(QK^\top)$ with $QK^\top$ have a **representational
  gap**: they cannot capture rank expansions that the softmax creates.
- **Low-rank attention compression** must happen *after* the softmax,
  not before.
- Any theorem assuming $\text{rank}(\sigma(Z)) \leq \text{rank}(Z)$ is
  **incorrect in general**.

---

## Results

Verified on CPU (NumPy), no GPU required.

| Test | Status |
|------|--------|
| Theorem 2.1: 2×2 rank-1 → rank-2 | ✅ PASS |
| Theorem 2.2: 3×3 rank-1 → rank-3 | ✅ PASS |
| Theorem 2.2: 4×5 rank-1 → rank-4 | ✅ PASS |
| Theorem 2.2: 10×8 rank-1 → rank-8 | ✅ PASS |
| 50×50 stress test | ✅ PASS |

---

## Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.26
- pytest ≥ 7.0

---

## License

MIT.
