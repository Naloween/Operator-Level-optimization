# The mode ensemble: exact second moments and what they force

Builds on [`01-crelu-structure.md`](01-crelu-structure.md). §1–§3 are unconditional; §4
carries one hypothesis, stated explicitly, and §5 records what is *not* implied.

Throughout, $\varepsilon$ is a **Rademacher mode**: the entries $\varepsilon_{\ell,i}$ are
i.i.d. uniform on $\{\pm1\}$, independent across layers and coordinates. Write
$E_\ell = \operatorname{diag}(\varepsilon_\ell)$ and $M_\ell = S_\ell + \Delta_\ell E_{\ell-1}$.

For a symmetric matrix $Y$, $\operatorname{Dg}(Y)$ denotes the diagonal matrix with the same
diagonal as $Y$ (off-diagonal entries zeroed).

---

## 1. The transfer operator is exact

> **Theorem 5 (one layer).** For any fixed symmetric $X$,
> $$\mathbb{E}_{\varepsilon}\big[M^\top X M\big] \;=\; S^\top X S \;+\; \operatorname{Dg}\!\big(\Delta^\top X \Delta\big),$$
> where $M = S + \Delta E$ and $E = \operatorname{diag}(\varepsilon)$ with i.i.d. Rademacher
> $\varepsilon$.

*Proof.* Expand:

$$M^\top X M = S^\top X S \;+\; S^\top X \Delta E \;+\; E \Delta^\top X S \;+\; E \Delta^\top X \Delta E.$$

Take expectations term by term.

- $\mathbb{E}[E] = 0$ since $\mathbb{E}[\varepsilon_i] = 0$, so the two cross terms vanish
  (they are linear in $E$, and $S^\top X \Delta$ is a constant matrix).
- For the last term, write $Y := \Delta^\top X \Delta$. Then
  $(EYE)_{ij} = \varepsilon_i Y_{ij}\varepsilon_j$, so
  $\mathbb{E}[(EYE)_{ij}] = \mathbb{E}[\varepsilon_i \varepsilon_j]\, Y_{ij} = \delta_{ij} Y_{ij}$,
  using independence for $i \ne j$ and $\varepsilon_i^2 = 1$ for $i = j$. That is exactly
  $\operatorname{Dg}(Y)$.

Summing gives the claim. $\blacksquare$

Define the **transfer operator** of layer $\ell$ on symmetric matrices:

$$\boxed{\;\mathcal{T}_\ell(X) \;:=\; S_\ell^\top X S_\ell \;+\; \operatorname{Dg}\!\big(\Delta_\ell^\top X \Delta_\ell\big).\;}$$

> **Theorem 6 (whole product).** With $\varepsilon$ Rademacher and independent across layers,
> $$\mathbb{E}_\varepsilon\big[J_\varepsilon^\top J_\varepsilon\big] \;=\; W_1^\top\,\big(\mathcal{T}_2 \circ \mathcal{T}_3 \circ \cdots \circ \mathcal{T}_L\big)(I_{n_L})\, W_1 .$$

*Proof.* By induction on the number of layers, integrating from the outside in. Let
$X_L := I$. For $\ell = L, L-1, \dots, 2$, the matrix $M_\ell$ depends on $\varepsilon_{\ell-1}$
only, and $\varepsilon_{\ell-1}$ is independent of $\varepsilon_1,\dots,\varepsilon_{\ell-2}$
which are the only randomness in $M_{\ell-1}\cdots M_2$. Conditioning on those and applying
Theorem 5 to the innermost expectation,

$$\mathbb{E}\big[(M_\ell \cdots M_2 W_1)^\top X_\ell (M_\ell\cdots M_2 W_1)\big] = \mathbb{E}\big[(M_{\ell-1}\cdots M_2 W_1)^\top \mathcal{T}_\ell(X_\ell) (M_{\ell-1}\cdots M_2 W_1)\big],$$

i.e. $X_{\ell-1} = \mathcal{T}_\ell(X_\ell)$. Unrolling from $\ell = L$ down to $\ell = 2$ and
finally conjugating by $W_1$ (deterministic) gives the statement. $\blacksquare$

**This is exact — no approximation, no asymptotics, no assumption on the weights.** It holds
at every training time, for trained weights, for any architecture of this form.

*Numerical check.* Monte Carlo over $2\times10^5$ random modes matches the single-layer
formula to $7.8\times10^{-4}$ relative (sampling error $\sim n^{-1/2} = 2.2\times10^{-3}$);
over $4\times10^4$ random full products the recursion matches to $7.2\times10^{-3}$ relative,
with the leading eigenvalues agreeing to three digits
($0.0499, 0.0084, 0.0040, 0.0016$ vs $0.0498, 0.0085, 0.0040, 0.0016$).

---

## 2. Interpretation: mode averaging decoheres in the neuron basis

The two terms of $\mathcal{T}$ play different roles.

- $S^\top X S$ is ordinary conjugation by the linear part; it is basis-covariant.
- $\operatorname{Dg}(\Delta^\top X \Delta)$ destroys all off-diagonal entries **in the neuron
  basis**.

So averaging over modes is a *decoherence* operation with a preferred basis, and the
preferred basis is the one the nonlinearity is written in. This is the precise sense in which
the gates inject structure that no basis-free linear analysis can see: the sign randomness
acts coordinatewise, so it privileges the coordinates.

> **Corollary 6.1 (energy identity).** $\operatorname{tr} \mathcal{T}(X) = \operatorname{tr}\!\big(X(SS^\top + \Delta\Delta^\top)\big)$.

*Proof.* $\operatorname{tr}(S^\top X S) = \operatorname{tr}(X S S^\top)$, and
$\operatorname{tr}\operatorname{Dg}(Y) = \operatorname{tr}(Y) = \operatorname{tr}(\Delta^\top X \Delta) = \operatorname{tr}(X\Delta\Delta^\top)$. $\blacksquare$

So total energy is transported by $SS^\top + \Delta\Delta^\top$ regardless of the decoherence;
only its *distribution across directions* is affected by $\operatorname{Dg}$.

---

## 3. Positivity and stabilization

Let $\mathcal{S}^d$ be the symmetric $d\times d$ matrices and $\mathcal{K} \subset \mathcal{S}^d$
the closed convex cone of positive semidefinite matrices, with interior
$\operatorname{int}\mathcal{K}$ the positive definite ones. $\mathcal{K}$ is a proper cone
(closed, convex, pointed, with nonempty interior).

> **Theorem 7 (positivity).** $\mathcal{T}_\ell(\mathcal{K}) \subseteq \mathcal{K}$.

*Proof.* Let $X \succeq 0$. Then $S^\top X S \succeq 0$, since
$v^\top S^\top X S v = (Sv)^\top X (Sv) \ge 0$. For the second term, the $i$-th diagonal entry
of $\Delta^\top X \Delta$ is $(\Delta e_i)^\top X (\Delta e_i) \ge 0$, so
$\operatorname{Dg}(\Delta^\top X\Delta)$ is a diagonal matrix with nonnegative entries, hence
PSD. A sum of PSD matrices is PSD. $\blacksquare$

> **Theorem 8 (exact criterion for strict positivity).** Let $X \succeq 0$ and $u \ne 0$. Then
> $$u^\top \mathcal{T}_\ell(X)\, u = 0 \iff \operatorname{span}\Big(\{S_\ell u\} \cup \{\Delta_\ell e_i : i \in \operatorname{supp} u\}\Big) \subseteq \ker X .$$
> Consequently $\mathcal{T}_\ell(X) \succ 0$ for **all** $X \succeq 0$, $X \ne 0$, if and only if
> that span equals $\mathbb{R}^d$ for every $u \ne 0$.

*Proof.* Expand the quadratic form:

$$u^\top \mathcal{T}_\ell(X) u = (S_\ell u)^\top X (S_\ell u) \;+\; \sum_{i} u_i^2\, (\Delta_\ell e_i)^\top X (\Delta_\ell e_i).$$

Every term is $\ge 0$ because $X \succeq 0$, so the sum vanishes iff each term does. With
$X = (X^{1/2})^2$, $(Sv)^\top X (Sv) = \|X^{1/2}S u\|^2$, so the first term vanishes iff
$S_\ell u \in \ker X$; the $i$-th summand vanishes iff $u_i = 0$ or
$\Delta_\ell e_i \in \ker X$. Together these say exactly that the stated span lies in
$\ker X$. For the second sentence: such a nonzero $X \succeq 0$ exists iff the span is a
*proper* subspace (take $X$ the orthogonal projector onto its complement); so strict
positivity holds iff no proper span occurs. $\blacksquare$

> **Corollary 8.1.** For $d \ge 3$, $\mathcal{T}_\ell$ is **never** strictly positive on
> $\mathcal{K}\setminus\{0\}$.

*Proof.* Take $u = e_i$. Then $\operatorname{supp} u = \{i\}$ and the span is
$\operatorname{span}\{S_\ell e_i,\, \Delta_\ell e_i\}$, of dimension at most $2 < d$. By
Theorem 8 there is a nonzero $X \succeq 0$ — the projector onto that span's complement — with
$e_i^\top \mathcal{T}_\ell(X) e_i = 0$. $\blacksquare$

*Numerical check.* Constructing exactly that $X$ for a random depth-5 width-6 network gives
$e_i^\top \mathcal{T}(X) e_i = 4.1\times10^{-17}$ and $\lambda_{\min}(\mathcal{T}(X)) = 4.1\times10^{-17}$:
singular, as predicted.

So the naive route to Perron–Frobenius is closed. What is available instead is
**primitivity**: it suffices that some *power* of the operator reaches the interior.

> **Definition.** A positive linear map $\mathcal{T}$ on $\mathcal{K}$ is *primitive* if there
> is $n \ge 1$ with $\mathcal{T}^n(\mathcal{K}\setminus\{0\}) \subseteq \operatorname{int}\mathcal{K}$.

> **Theorem 9 (stabilization under primitivity).** If $\mathcal{T} = \mathcal{T}_2\circ\cdots\circ\mathcal{T}_L$
> is primitive, then
>
> 1. its spectral radius $\lambda > 0$ is a simple eigenvalue with eigenvector
>    $X_\star \in \operatorname{int}\mathcal{K}$;
> 2. every other eigenvalue has modulus $< \lambda$;
> 3. for every $X_0 \in \mathcal{K}\setminus\{0\}$, $\lambda^{-n}\mathcal{T}^n(X_0) \to c X_\star$
>    with $c > 0$, geometrically.

*Proof.* $\mathcal{T}$ is a linear map leaving the proper cone $\mathcal{K}$ invariant
(Theorem 7). Primitivity means some power maps $\mathcal{K}\setminus\{0\}$ into the interior,
which is the hypothesis of the Perron–Frobenius/Krein–Rutman theorem for cone-preserving
linear maps; it yields (1) and (2). For (3), a map sending $\mathcal{K}\setminus\{0\}$ into
$\operatorname{int}\mathcal{K}$ has image of finite diameter in the Hilbert projective metric,
so by Birkhoff's contraction theorem it is a strict contraction there with ratio
$\tanh(\operatorname{diam}/4) < 1$; iterating contracts to the unique fixed ray, which is
$\mathbb{R}_{>0} X_\star$. $\blacksquare$

**Primitivity is checkable, and is not automatic.** It holds in the instances tested — the
adversarial $X$ of Corollary 8.1 satisfies $\mathcal{T}^2(X) \succ 0$ with
$\lambda_{\min} = 4.0\times10^{-2}$, so $n = 2$ there — but it is a property of
$(S_\ell, \Delta_\ell)$ and can fail, for instance if $\Delta_\ell = 0$ (looks-linear
initialization) and some $S_\ell$ is singular. It should be verified rather than assumed;
`tests/test_theory_transfer.py` checks it numerically.

**What Theorem 9 does and does not say.** It says the mode-averaged Gram, normalized, has a
well-defined limiting *shape* $X_\star$ reached exponentially fast in depth. It does **not**
say $X_\star$ is ill-conditioned. Stabilization is not low-rankness. Any claim that depth
*produces* low rank has to come from what training does to $(S_\ell, \Delta_\ell)$, which is
the subject of [`03-dynamics.md`](03-dynamics.md).

---

## 4. Realized modes versus typical modes

The ensemble above averages over all $2^{(L-1)d}$ modes. The data realizes only some of them.

*Measured*, on CReLU networks trained by plain gradient descent from looks-linear
initialization ($d=10$, $B=64$ inputs, $L\eta$ held fixed):

| statistic | Rademacher value | measured ($L=64$, 2000 steps) |
|---|---|---|
| $\|\mathbb{E}_x[\varepsilon(x)]\|$ (mean sign) | $0$ (noise floor $B^{-1/2} = 0.125$) | $0.173$ |
| cross-input sign agreement | $0.5$ | $0.524$ |
| cross-layer sign agreement | $0.5$ | $0.505$ |
| $\mathbb{E}\log(s_1/s_2)$ | — | random modes $0.097$, **realized $0.215$** |

So realized modes are close to typical in low-order statistics but their operators are
**more** spectrally separated than typical ones, by a factor $\approx 2.2$ at $L = 64$.

> **Hypothesis (H-mode).** For the modes realized by the data,
> $$\operatorname{sep}\big(J_{\varepsilon(x)}\big) \;\ge\; \mathbb{E}_\varepsilon\big[\operatorname{sep}(J_\varepsilon)\big]$$
> where $\operatorname{sep}(J) := \log(\sigma_1(J)/\sigma_2(J))$.

This is *one-sided*, which is the direction the measurement supports, and it is what a
separation **lower bound** for the ensemble would need in order to transfer to the data. It is
falsifiable by the measurement above run at other depths, widths, tasks and training lengths;
`olo.theory.modes.compare` computes both sides.

**Degeneracy to keep in mind.** At looks-linear initialization $\Delta = 0$, so every mode
gives the same operator and the ensemble is a point mass. Mode structure is *created* by
training; there is nothing to average over at $t = 0$.

---

## 5. What is not implied

Recorded explicitly, because each was checked and failed.

- **Not** that $J_\varepsilon$ concentrates around $\mathbb{E}_\varepsilon[J_\varepsilon]$.
  Theorem 6 is a second-moment statement; it constrains averages, not individual modes.
- **Not** that the ensemble's spectral gap grows exponentially in depth. Measured on trained
  networks at depths $4$–$128$, $\log(\lambda_1/\lambda_2)$ per layer is small and irregular
  ($10^{-4}$ to $10^{-2}$), with no clean Lyapunov rate. A rate would need conditions on
  $(S_\ell,\Delta_\ell)$ that training is not known to supply.
- **Not** anything about the *dynamics*. Theorems 5–9 hold at frozen weights. The coupling
  between modes through the shared weight gradient is untouched, and is the open problem.
