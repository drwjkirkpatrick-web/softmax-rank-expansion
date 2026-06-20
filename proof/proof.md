# Proof: Post-Softmax Rank Expansion

## Lemma 2.1 (Structure of Softmax of Rank-1 Matrix)

Let $Z = uv^\top$ with $u \in \mathbb{R}^m$ and $v \in \mathbb{R}^n$.
The row-wise softmax is:

$$\sigma(Z)_{ij} \;=\; \frac{\exp(u_i v_j)}{\sum_{k=1}^{n} \exp(u_i v_k)}
\;=\; \frac{1}{S_i} \exp(u_i v_j)$$

where $S_i = \sum_{k=1}^{n} \exp(u_i v_k)$.

**Proof.** Direct substitution of $Z_{ij} = u_i v_j$ into the softmax
formula. ∎

---

## Lemma 2.2 (Proportionality Condition)

Two rows $i$ and $j$ of $\sigma(Z)$ are proportional if and only if
$u_i = u_j$.

**Proof.** Row $i$ is proportional to the vector
$[\exp(u_i v_1), \exp(u_i v_2), \ldots, \exp(u_i v_n)]$.
Rows $i$ and $j$ are proportional iff there exists $c \neq 0$ such
that $\exp(u_i v_k) = c \cdot \exp(u_j v_k)$ for all $k$. Taking logs:
$(u_i - u_j) v_k = \ln c$ for all $k$.

If $u_i = u_j$, then $\ln c = 0$ and $c = 1$ — the rows are identical.

If $u_i \neq u_j$, then $v_k = \frac{\ln c}{u_i - u_j}$ must be constant
for all $k$. This holds only if all components of $v$ are equal. ∎

---

## Lemma 2.3 (2×2 Counterexample)

For $Z = \begin{pmatrix} 0 & 0 \\ 0 & ab \end{pmatrix}$ with
$a \neq 0, b \neq 0$:

$$\sigma(Z) \;=\; \begin{pmatrix}
\tfrac{1}{2} & \tfrac{1}{2} \\[4pt]
\tfrac{1}{1+e^{ab}} & \tfrac{e^{ab}}{1+e^{ab}}
\end{pmatrix}$$

and the two rows are linearly independent.

**Proof.** Suppose $\alpha \cdot [\tfrac{1}{2}, \tfrac{1}{2}] + \beta
\cdot [\tfrac{1}{1+e^{ab}}, \tfrac{e^{ab}}{1+e^{ab}}] = [0, 0]$.

From the first component:
$$\frac{\alpha}{2} + \frac{\beta}{1+e^{ab}} = 0 \quad\Rightarrow\quad
\beta = -\frac{\alpha(1+e^{ab})}{2}$$

Substituting into the second component:
$$\frac{\alpha}{2} + \frac{e^{ab}}{1+e^{ab}} \cdot
\left(-\frac{\alpha(1+e^{ab})}{2}\right) = 0$$
$$\frac{\alpha}{2} - \frac{\alpha e^{ab}}{2} = 0$$
$$\alpha(1 - e^{ab}) = 0$$

Since $ab \neq 0$, we have $e^{ab} \neq 1$, so $\alpha = 0$ and thus
$\beta = 0$. The rows are linearly independent. ∎

---

## Proof of Theorem 2.1

By Lemma 2.3, the explicit $2 \times 2$ construction yields two linearly
independent rows, so the matrix has full rank 2. The SVD of
$\sigma(Z)$ has two positive singular values, confirming the rank
computation numerically. ∎

---

## Proof of Theorem 2.2 (General Case)

Let $u_i = a \cdot i$ and $v_j = b \cdot j$ with $a, b \neq 0$.
Then:

$$\sigma(Z)_{ij} \;=\; \frac{\exp(ab \cdot ij)}{\sum_{k=0}^{n-1} \exp(ab \cdot ik)}
\;=\; \frac{1}{S_i} \left(r^i\right)^j$$

where $r = \exp(ab) > 1$ and $S_i = \sum_{k=0}^{n-1} (r^i)^k$.

Each row $i$ is proportional to the geometric progression
$[1, r^i, r^{2i}, \ldots, r^{(n-1)i}]$.

For $i = 0, 1, \ldots, m-1$, the rows form a generalized Vandermonde
matrix. The standard Vandermonde matrix with nodes $x_0, x_1, \ldots,
x_{m-1}$ and $m \leq n$ has full row rank because the determinant
of any $m \times m$ submatrix is $\prod_{0 \leq i < j < m} (x_j - x_i)$,
which is nonzero when all $x_i$ are distinct. Here $x_i = r^i$, and
since $r > 1$, all $x_i$ are distinct.

Hence the $m$ rows are linearly independent. With $n$ columns and $m$
rows:

$$\text{rank}(\sigma(Z)) \;=\; \min(m, n)$$ ∎

---

## Remark: Why This Matters for Transformers

In a transformer with $m$ queries and $n$ keys, the pre-softmax
attention matrix is $A = QK^\top / \sqrt{d_k}$. If the queries and
keys are perfectly aligned (e.g., $Q = K$, $A$ is symmetric
positive-definite, typically full rank), the softmax is harmless.

However, in **linear attention** approximations (e.g., Performer,
Linformer, RNN-style recurrences), researchers approximate:

$$\sigma(QK^\top) \approx Q K^\top$$

by treating the softmax as approximately linear or absorbing it into a
kernel. Our theorem shows this approximation is **fundamentally flawed**
when $QK^\top$ is low rank — the softmax can *increase* the rank,
meaning the low-rank approximation discards essential information.

A practical consequence: if you project attention to rank $r$ before
the softmax, the softmax of the reconstruction may still have higher
rank than $r$. The correct approach is to approximate *after* the
softmax, not before.
