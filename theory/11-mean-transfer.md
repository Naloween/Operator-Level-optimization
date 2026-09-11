# Why the nonlinearity bounds the implicit bias: a Perron–Frobenius mechanism

Continues [`10-mismatch.md`](10-mismatch.md), which reduced the implicit bias of a factored
model to one superoperator and bounded it by the anisotropy of the context Grams, but left
the **drift** term — the input dependence — uncontrolled. This file computes the drift's mean
in closed form under a hypothesis that is realistic and measurable, and extracts from it a
mechanism that explains the depth behaviour actually observed.

The result, in one line: **for the linear part the context Grams obey a congruence recursion
whose spectrum spreads exponentially in depth; the nonlinear part obeys a Perron–Frobenius
recursion whose spectrum converges to a fixed profile. The nonlinearity is what makes the
anisotropy — and hence the bias — bounded uniformly in depth.**

---

## 1. The hypothesis

CReLU layers, `W_j D(z) = S_j + \Delta_j E_{j-1}` with `E = \mathrm{diag}(\varepsilon)`,
`\varepsilon \in \{\pm1\}^d` the layer's sign pattern (Lemma 2 of file 01).

> **(H-R) Rademacher gates.** Under the data distribution, the layer sign vectors
> `\varepsilon_1,\dots,\varepsilon_{L-1}` are mutually independent, each uniform on
> `\{\pm 1\}^d`.

**Measured support**, on trained CReLU networks (`olo.theory.modes.compare`): mean sign at the
finite-sample noise floor, and cross-input and cross-layer sign agreement within `0.02` of
`0.5`. So the first and second moments match (H-R) closely. **Full mutual independence is
strictly stronger than what has been checked**, and is certainly false exactly — `\varepsilon_2`
is a deterministic function of `\varepsilon_1` and `x`. (H-R) is a modelling hypothesis with
measured first- and second-moment support, used in the same spirit as a mean-field assumption.

---

## 2. The two elementary transfers

> **Lemma 1 [EXACT under (H-R), one layer].** For `M = S + \Delta E` with `E` Rademacher,
> $$\mathbb E\big[M X M^\top\big] = S X S^\top + \Delta\,\mathrm{Dg}(X)\,\Delta^\top \;=:\; \mathfrak F(X),$$
> $$\mathbb E\big[M^\top X M\big] = S^\top X S + \mathrm{Dg}\big(\Delta^\top X \Delta\big) \;=:\; \mathfrak B(X),$$
> where `\mathrm{Dg}(\cdot)` keeps the diagonal and zeroes the rest.

*Proof.* `MXM^\top = SXS^\top + SXE\Delta^\top + \Delta EXS^\top + \Delta EXE\Delta^\top`. The
middle two vanish since `\mathbb E[E] = 0`. For the last,
`(\mathbb E[EXE])_{ij} = X_{ij}\mathbb E[\varepsilon_i\varepsilon_j] = X_{ij}\delta_{ij}`, which is
`\mathrm{Dg}(X)`. The second identity is the same computation with the factors transposed. ∎
Verified symbolically.

Both are **positive maps**: `X \succeq 0 \Rightarrow \mathfrak F(X) \succeq 0`, since
`\mathrm{Dg}` preserves the PSD cone.

---

## 3. The mean contexts and the mean transfer

Write `A_l = M_L\cdots M_{l+1}` (using patterns `\varepsilon_l,\dots,\varepsilon_{L-1}`) and
`B_l = D(\varepsilon_{l-1})N_l` with `N_l = M_{l-1}\cdots M_2W_1` (using
`\varepsilon_1,\dots,\varepsilon_{l-2}`). **The two index-sets are disjoint**, so under (H-R)
`A_l` and `B_l` are independent — this is the structural fact the whole file rests on.

> **Theorem 2 [under (H-R)].** With `\Phi_l(X) := \mathbb E[A_lXA_l^\top]` and
> `\Psi_l(X) := \mathbb E[N_l^\top X N_l]`,
> $$\Phi_l = \mathfrak F_L\circ\mathfrak F_{L-1}\circ\cdots\circ\mathfrak F_{l+1}, \qquad \Psi_l = \Psi_2\circ\mathfrak B_2\circ\cdots\circ\mathfrak B_{l-1},$$
> with `\Phi_L = \mathrm{Id}` and `\Psi_2(X) = W_1^\top XW_1`. The **mean context Grams** are
> `P_l := \mathbb E[A_lA_l^\top] = \Phi_l(I)` and `Q_l := \mathbb E[B_l^\top B_l] = \Psi_l(I)`,
> the latter using `D^\top D = I` (CReLU gate isometry) to remove the outer gate.

*Proof.* `A_{l-1} = A_lM_l` with `M_l` depending on `\varepsilon_{l-1}`, independent of
`A_l`'s patterns. Hence
`\Phi_{l-1}(X) = \mathbb E[A_lM_lXM_l^\top A_l^\top] = \Phi_l(\mathbb E[M_lXM_l^\top]) = \Phi_l(\mathfrak F_l(X))`
by Lemma 1 and the tower property. Iterating from `\Phi_L = \mathrm{Id}` gives the composition.
The `\Psi` recursion is identical with `\mathfrak B`. ∎

> **Corollary 3 (mean transfer).** Under (H-R), the transfer superoperator of file 10 has mean
> $$\mathbb E\big[\mathcal T\big](H) \;=\; \sum_{l=1}^{L} P_l\,H\,Q_l ,$$
> with `P_l, Q_l` given in closed form by Theorem 2.

*Proof.* `\mathcal T(H) = \sum_l A_lA_l^\top H B_l^\top B_l` and `A_l \perp B_l`, so the
expectation factorises across the two sides of `H`. ∎

**This is the first closed form for the mean of the object that has been the obstruction since
[`03-dynamics.md`](03-dynamics.md).** Everything below reads structure off it.

---

## 4. The dichotomy

Theorem 2 makes the anisotropy of `P_l` — which is what file 10's bound needs — the result of
iterating `\mathfrak F`. Two extreme cases behave completely differently.

> **Theorem 4 (linear branch: exponential spreading).** If `\Delta_j = 0` for all `j`, then
> `\mathfrak F_j(X) = S_jXS_j^\top` and `P_l = (S_L\cdots S_{l+1})(S_L\cdots S_{l+1})^\top`. The
> log-spectrum of `P_l` therefore spreads at the Lyapunov rate of the `S`-chain: **linearly in
> `L-l`**, unless the `S_j` are orthogonal, in which case `P_l = I` for every `l`.

*Proof.* Immediate from Lemma 1 with `\Delta = 0`; the spreading statement is the standard
Furstenberg–Kesten/Oseledets behaviour of a product of matrices, with rate the gap between the
top two Lyapunov exponents (zero exactly when the factors are orthogonal). ∎

> **Theorem 5 (nonlinear branch: Perron–Frobenius saturation).** If `S_j = 0` for all `j`, then
> `\mathfrak F_j(X) = \Delta_j\,\mathrm{Dg}(X)\,\Delta_j^\top`, and the recursion **closes on
> diagonals**: writing `p^{(l)} := \mathrm{diag}(P_l)`,
> $$p^{(l-1)} \;=\; \big(\Delta_l\circ\Delta_l\big)\,p^{(l)},$$
> with `\circ` the entrywise square — a **nonnegative linear map**. If the `\Delta_j\circ\Delta_j`
> are primitive and equal to a common `K`, then `p^{(l)}/\|p^{(l)}\|` converges geometrically to
> the Perron eigenvector of `K`, at rate `|\lambda_2/\lambda_1|`. Consequently the spectrum of
> `P_l`, normalised, converges to a fixed profile and **its spread saturates: it is bounded
> uniformly in depth.**

*Proof.* `\mathrm{Dg}(X)` is diagonal, so `\mathfrak F(X) = \Delta\,\mathrm{diag}(x)\,\Delta^\top`
where `x = \mathrm{diag}(X)`; taking the diagonal of that gives
`\big(\Delta\,\mathrm{diag}(x)\,\Delta^\top\big)_{ii} = \sum_j \Delta_{ij}^2x_j`, i.e.
`(\Delta\circ\Delta)x`. Since only the diagonal of `P_l` feeds the next step, the iteration is
exactly the nonnegative recursion stated. Perron–Frobenius for a primitive nonnegative matrix
gives geometric convergence of the normalised iterate to the Perron vector, hence convergence
of `P_l/\mathrm{tr}\,P_l` to the fixed matrix `\Delta\,\mathrm{diag}(p^\ast)\,\Delta^\top/\mathrm{tr}(\cdot)`. ∎

**Numerically** (width 8, 5 seeds, normalised trace), the log-spectrum spread of `P` at depth
`k`:

| `k` | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| `\Delta = 0` (linear) | 9.80 | 20.73 | 37.12 | (36.8) | (38.7) |
| `S = 0` (nonlinear) | **6.44** | **6.37** | **6.37** | **6.37** | **6.37** |

The linear row grows linearly until it reaches `\log` of the floating-point range and the
measurement saturates artificially; the nonlinear row is *converged* by `k = 5`.

---

## 5. Why the nonlinear branch contracts: Schur–Horn

The mechanism in Theorem 5 is not an accident of the `S = 0` corner.

> **Theorem 6 (the nonlinear branch never sharpens the spectrum).** For Hermitian `X`,
> `\mathrm{diag}(X) \prec \lambda(X)` (majorization). Hence for every convex `\varphi`,
> `\sum_i\varphi(\mathrm{Dg}(X)_{ii}) \le \sum_i\varphi(\lambda_i(X))`; in particular
> `\mathrm{tr}\,\mathrm{Dg}(X) = \mathrm{tr}\,X` and
> `\|\mathrm{Dg}(X)\|_F \le \|X\|_F`, with equality iff `X` is diagonal.

*Proof.* Schur–Horn. `\mathrm{diag}(X) = \mathrm{diag}(U\Lambda U^\top)` gives
`\mathrm{diag}(X)_i = \sum_j |U_{ij}|^2\lambda_j`, a doubly stochastic image of `\lambda`, and
Birkhoff–von Neumann plus Hardy–Littlewood–Pólya give the majorization. ∎
Verified over `2\times10^4` random Hermitian matrices, 0 violations.

So in the general map `\mathfrak F(X) = SXS^\top + \Delta\,\mathrm{Dg}(X)\,\Delta^\top`, the
**linear branch passes the full spectral spread to the next layer, while the nonlinear branch
first flattens it.** The nonlinearity is a contraction acting against the congruence chain's
expansion, and it is the presence of `\mathrm{Dg}` — the loss of off-diagonal information in
the neuron basis — that does it.

---

## 6. What this says about the implicit bias

File 10, Theorem 5.1: the relative bias obeys
`\beta \le \sum_l w_l(a_l+b_l+a_lb_l)` with `a_l = \|\widetilde P_l\|/p_l` the anisotropy of the
upper context Gram (and `b_l` likewise). Combining:

> **Corollary 7.** Under (H-R), *in mean*:
> * **(linear-dominated, `\Delta \approx 0`)** `a_l` grows exponentially in `L-l` unless the
>   `S_j` are orthogonal, so the bound on `\beta` is **unbounded in depth**. Orthogonal `S_j`
>   give `a_l = 0` and `\beta = 0` (recovering file 10's Cor. 4.2).
> * **(nonlinearity-dominated, `S \approx 0`, `\Delta\circ\Delta` primitive)** `a_l` converges
>   to a fixed value, so `\beta` is **bounded uniformly in `L`**.

**This is the mechanism the measurements were pointing at.** It predicts, and the earlier
measurements show:

| configuration | this theory | measured (file 09 §5) |
|---|---|---|
| CReLU looks-linear (`\Delta = 0`, `S` non-orthogonal after training) | linear branch, spreading, strong bias | `d/dt\log\mathrm{PR} = -0.38` — rank falls |
| CReLU Xavier (`\Delta` large) | nonlinear branch, saturating, weak bias | `+0.008` — rank does *not* fall |

The apparent paradox of file 09 §5 — that a *well-conditioned* start develops the low-rank
bias while a badly-conditioned one does not — is exactly Theorem 4 versus Theorem 5. A
looks-linear network has `\Delta = 0`, so it is *entirely* in the linear branch and inherits
the congruence chain's exponential spreading. A Xavier network has large `\Delta`, so the
Perron–Frobenius branch dominates and its context anisotropy saturates.

**The nonlinearity is not what causes the low-rank bias. It is what bounds it.**

---

## 7. Status and limits

| Result | Statement | Status |
|---|---|---|
| Lemma 1 | `\mathbb E[MXM^\top] = SXS^\top+\Delta\mathrm{Dg}(X)\Delta^\top` | exact under (H-R); symbolic |
| Thm 2 | mean contexts as compositions of `\mathfrak F,\mathfrak B` | exact under (H-R) |
| Cor. 3 | `\mathbb E[\mathcal T](H) = \sum_l P_lHQ_l` | exact under (H-R) |
| Thm 4 | linear branch: Lyapunov spreading | standard |
| Thm 5 | nonlinear branch: Perron–Frobenius saturation | proved; matches numerics |
| Thm 6 | `\mathrm{Dg}` is majorization-decreasing | Schur–Horn |
| Cor. 7 | the bias is depth-bounded iff the nonlinear branch dominates | follows from 10's Thm 5.1 |

**Limits, stated.**

* **(H-R) is a modelling hypothesis.** Its first and second moments are measured; full mutual
  independence is not, and is false exactly. Everything in §3 onwards inherits this.
* **Corollary 3 gives the mean of `\mathcal T`, not of `\mathcal T(G_x)`.** The drift term of
  file 10 is `\mathbb E_x[\mathcal D(G_x)]`, which equals
  `\mathbb E[\mathcal D]\,\mathbb E[G_x] + \mathrm{Cov}(\mathcal D, G_x)`; this file computes the
  first piece and says nothing about the covariance. Since `G_x` and `\varepsilon(x)` are both
  functions of `x`, that covariance is not zero.
* **Theorem 5 assumes the `\Delta_j\circ\Delta_j` share a primitive limit.** Layer-varying
  `\Delta` gives an inhomogeneous product of nonnegative matrices; Birkhoff-contraction
  arguments should still give saturation, but that is not proved here.
* **File 10's Theorem 5.1 is one-sided**, so Corollary 7 bounds how large the bias can be, not
  how large it is.
* CReLU only. The ReLU gate is a projector, not a sign matrix, so Lemma 1 does not apply; the
  analogue would need `\mathbb E[DXD]` for Bernoulli `D`, which has a different structure
  (it does not kill the off-diagonal, it scales it by `p^2`).
