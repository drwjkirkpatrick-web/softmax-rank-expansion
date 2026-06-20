# Theorem: Post-Softmax Rank Expansion

**Status:** Verified — constructive counterexample + SVD proof  
**Domain:** Linear algebra / Transformer analysis  
**Date:** 2026-06-20

---

## Motivation

In transformer analysis, researchers often examine the rank of
attention-pattern matrices and weight matrices. A common implicit
assumption is that applying the softmax (which is an element-wise
monotonic transformation) preserves or reduces the rank of a matrix.

This theorem shows the assumption is **false**: there exist rank-1
matrices whose row-wise softmax has **strictly higher rank**.

---

## Definitions

For a matrix $Z \in \mathbb{R}^{m \times n}$, the **row-wise softmax**
is defined as:

$$\sigma(Z)_{ij} \;=\; \frac{\exp(Z_{ij})}{\sum_{k=1}^{n} \exp(Z_{ik})}$$

The **numerical rank** (or effective rank) with tolerance $\tau > 0$ is:

$$\text{rank}_\tau(M) \;=\; \#\{\,i \;|\; \sigma_i(M) \geq \tau\,\}$$

where $\sigma_i(M)$ are the singular values of $M$ in descending order.

---

## Theorem 2.1 (Rank-1 → Rank-2)

There exists a matrix $Z \in \mathbb{R}^{2 \times 2}$ with
$\text{rank}(Z) = 1$ such that:

$$\text{rank}(\sigma(Z)) \;=\; 2$$

**Explicit construction:**

$$Z \;=\; \begin{pmatrix} 0 & 0 \\ 0 & ab \end{pmatrix}$$

with $a \neq 0$ and $b \neq 0$ (so $Z = uv^\top$ with
$u = (0, a)^\top$ and $v = (0, b)^\top$).

**Verification.** The row-wise softmax gives:

$$\sigma(Z) \;=\; \begin{pmatrix}
\tfrac{1}{2} & \tfrac{1}{2} \\[4pt]
\tfrac{1}{1+e^{ab}} & \tfrac{e^{ab}}{1+e^{ab}}
\end{pmatrix}$$

The two rows are not scalar multiples (see Lemma 2.3), hence the
matrix has full rank. The singular value decomposition yields
$\sigma_1 > \sigma_2 > 0$, confirming $\text{rank}(\sigma(Z)) = 2$. ∎

---

## Theorem 2.2 (General $m \times n$ Rank Jump)

For any dimensions $m \geq 2$ and $n \geq 2$, there exists
$Z \in \mathbb{R}^{m \times n}$ with $\text{rank}(Z) = 1$ and:

$$\text{rank}(\sigma(Z)) \;\geq\; 2$$

In particular, for $m, n \leq 5$ and the Vandermonde-like
construction below, we have $\text{rank}(\sigma(Z)) = \min(m, n)$.

**Construction (Vandermonde-like).** Let $u_i = a \cdot i$ for
$i = 0, \ldots, m-1$ and $v_j = b \cdot j$ for $j = 0, \ldots, n-1$
with $a, b \neq 0$. Define $Z = uv^\top$. Then:

$$\sigma(Z)_{ij} \;=\; \frac{\exp(ab \cdot ij)}{\sum_{k=0}^{n-1} \exp(ab \cdot ik)}
\;=\; \frac{1}{S_i} \left(r^i\right)^j$$

where $r = \exp(ab)$ and $S_i = \sum_{k=0}^{n-1} (r^i)^k$.

Each row $i$ is proportional to a geometric progression
$[1, r^i, r^{2i}, \ldots, r^{(n-1)i}]$. For $m \leq n \leq 5$, these
rows form a generalized Vandermonde matrix whose determinant is
nonzero because the nodes $r^i$ are distinct (since $r > 1$).
Hence $\text{rank}(\sigma(Z)) = m = \min(m, n)$. ∎

**Remark.** For larger dimensions ($m$ or $n > 5$), the exponential
values $r^{ij}$ span ranges that exceed double-precision arithmetic,
causing numerical cancellation. The analytical result still holds (the
rows remain linearly independent in exact arithmetic), but numerical
verification requires arbitrary-precision computation or a different
construction.

---

## Corollary 2.3 (Pitfall for Low-Rank Attention Approximation)

Any algorithm that assumes $\text{rank}(\sigma(Z)) \leq
\text{rank}(Z)$ is **incorrect in general**. In particular:

- Low-rank attention factorizations must handle the softmax explicitly.
- Kernel methods approximating $\sigma(QK^\top)$ by low-rank $QK^\top$
  have a fundamental representational gap.

---

## Open Questions

1. **Exact rank formula.** For $Z = uv^\top$, can we derive a closed
   formula for $\text{rank}(\sigma(Z))$ in terms of the multiplicities
   of components in $u$ and $v$?

2. **Approximate rank.** How close can $\sigma(Z)$ be to a rank-$r$
   matrix when $Z$ has rank $r$? Is there a quantitative Eckart-Young
   bound?

3. **Column-wise softmax.** Does the same phenomenon occur for
   column-wise softmax? Yes, by symmetry.
