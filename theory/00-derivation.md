# Spectral dynamics of deep networks, without choosing a basis

This is the audit document. It defines every object, derives every claim, and tags each step
**[EXACT]**, **[ASSUMPTION]** or **[MEASURED]**. It is self-contained: nothing is imported
from the other files.

It replaces an earlier version built on per-direction quantities. That version rested on an
alignment assumption which is measured to fail qualitatively (§9), and the whole point of
what follows is that the assumption is never needed.

---

## 1. Setup

**Network.** `L` layers `W_1,…,W_L` with **gates** `D_l` between them. The **operator** is the
input–output Jacobian

$$J \;=\; W_L\,D_{L-1}\,W_{L-1}\cdots D_1\,W_1 . \tag{1.1}$$

| model | gate `D_l` |
|---|---|
| deep linear | `I` |
| FGLN | fixed diagonal |
| ReLU MLP | `diag(1[z>0])` |
| CReLU MLP | `[diag(1[z>0]); −diag(1[z<0])]` |

**Gate pattern.** A *pattern* `ε` fixes all the gates by hand rather than reading them off an
input; `J_ε` is the resulting operator. When `ε` is the pattern an input `x` produces,
`J_ε = J(x)` — **[MEASURED]** to `1e-14`. Patterns matter for one reason: `t ↦ J_ε(t)` is a
smooth (polynomial) function of the weights, whereas `t ↦ J(x)` jumps whenever `x` crosses a
region boundary.

**Contexts.** For each layer, split the product around `W_l`:

$$A_l := W_LD_{L-1}\cdots W_{l+1}D_l,\qquad B_l := D_{l-1}W_{l-1}\cdots W_1,\qquad J = A_l W_l B_l . \tag{1.2}$$

**Weight gradient.** `Γ_l := ∂\mathcal L/∂W_l` — **one matrix per layer**, computed from the
real loss on the real data. Every pattern is driven by the same `Γ`; patterns differ only in
how they compose it.

> **(1.3) [EXACT]** Holding a pattern fixed, under `\dot W_l = -Γ_l`,
> $$\dot J_\varepsilon \;=\; -\sum_{l=1}^{L} A_l^\varepsilon\,Γ_l\,B_l^\varepsilon .$$

*Proof.* With the gates fixed, `J_ε` is a polynomial in the weights; apply the product rule
to (1.2), which holds for every `l` simultaneously. ∎

That is the only dynamical input below. Everything else is linear algebra applied to
`(J, \dot J)`.

---

## 2. What the low-rank bias is, and why the obvious route forces an assumption

The bias is the claim that training makes `J`'s spectrum **concentrate** — a few singular
values grow relative to the rest, so the operator becomes effectively low rank.

The obvious way to study this is to track each singular value `s_k` and ask whether the large
ones grow faster. That runs into a problem immediately. "Direction `k`" is defined by the
singular vectors `u_k, v_k`, and to turn `\dot s_k = u_k^\top \dot J v_k` into something
interpretable one writes `\dot J` from (1.3) and needs the matrices `A_lA_l^\top` and
`B_l^\top B_l` to be **diagonal in the bases `\{u_k\},\{v_k\}`**. They are not, in general;
§9 measures what that costs, and it is not a small correction.

**So we will not track individual singular values.** The rest of this file uses only
quantities that do not refer to any basis.

---

## 3. Traces, and why `M = J^\top J`

### 3.1 The bridge

A trace is basis-free: `\operatorname{tr}(A)` is the same number however you coordinatise.
To exploit that we need the spectrum expressed through traces, and the object that does it is

$$M \;:=\; J^\top J, \qquad \text{eigenvalues } \mu_k = s_k^2 .$$

`M` is symmetric positive semidefinite, so it has a genuine eigendecomposition, and

$$\operatorname{tr}(M^p) \;=\; \sum_k \mu_k^{\,p} \tag{3.1}$$

— the **power sums of the spectrum**. That is the whole role of `M^p`: `p` is a dial that
weights the spectrum. `p = 1` counts total energy `\sum_k\mu_k = \|J\|_F^2`; larger `p`
weights the top of the spectrum more heavily. We will only ever need `p = 1` and `p = 2`.
(Higher `p` gives finer shape information at no extra cost in assumptions, but is not used.)

### 3.2 An effective rank made of two traces

$$\boxed{\ \mathrm{PR} \;:=\; \frac{\operatorname{tr}(M)^2}{\operatorname{tr}(M^2)} \;=\; \frac{\big(\sum_k\mu_k\big)^2}{\sum_k\mu_k^2}\ }$$

the **participation ratio**. If `r` of the `\mu_k` are equal and the rest zero,
`\mathrm{PR} = r` exactly; for an isometry `\mathrm{PR} = d`, for a rank-one operator
`\mathrm{PR} = 1`. It is a smooth, scale-invariant count of how many directions are alive —
`\mathrm{PR}(cM) = \mathrm{PR}(M)`, so it measures the spectrum's *shape* and ignores its
size. **The low-rank bias is the statement `\mathrm{PR}` decreases.**

Two traces. No eigenvectors.

---

## 4. The exact dynamics of `PR`

### 4.1 The object `X`, and what `x_k` is

Put

$$X \;:=\; J^\top \dot J .$$

`X` is *not* symmetric and has no special structure; it is just the natural pairing of the
operator with its velocity. Its role is fixed by one fact. Let `w_k` be a unit eigenvector of
`M` for a simple eigenvalue `\mu_k`, and write

$$x_k := w_k^\top X\, w_k \qquad\text{(the diagonal of } X \text{ in } M\text{'s eigenbasis).}$$

> **(4.1) [EXACT, given simple `\mu_k`]** `\dot\mu_k = 2x_k`.

*Proof.* `\dot M = \dot J^\top J + J^\top\dot J = X^\top + X`. First-order perturbation theory
for a simple eigenvalue of a symmetric matrix gives
`\dot\mu_k = w_k^\top \dot M w_k = w_k^\top(X+X^\top)w_k = 2w_k^\top X w_k`. ∎
**[MEASURED]** against finite differences of `\mathrm{eigvalsh}`, agreeing to the `O(h)`
truncation error.

So **`x_k` is (half) the growth rate of eigenvalue `k`**, and `x_k/\mu_k` is its *relative*
growth rate — the quantity that says whether direction `k` is gaining share.

**The point that makes all of this work.** We never compute `x_k`, and never form `M`'s
eigenbasis. We only need two sums of them, and both are traces:

$$\operatorname{tr}(X) = \sum_k x_k, \qquad \operatorname{tr}(MX) = \sum_k \mu_k x_k . \tag{4.2}$$

*Proof.* Evaluate each trace in `M`'s eigenbasis, where `M = \mathrm{diag}(\mu)`: the diagonal
of `X` is `x_k` and the diagonal of `MX` is `\mu_kx_k`. Traces are basis-independent, so the
values hold however they are computed. ∎ **[MEASURED]** to `4\times10^{-13}`.

### 4.2 The two derivatives

> **(4.3) [EXACT]** `\dfrac{d}{dt}\operatorname{tr}(M) = 2\operatorname{tr}(X)` and
> `\dfrac{d}{dt}\operatorname{tr}(M^2) = 4\operatorname{tr}(MX)`.

*Proof.* `\dot M = X + X^\top`, so `\frac{d}{dt}\operatorname{tr}M = \operatorname{tr}(X)+\operatorname{tr}(X^\top) = 2\operatorname{tr}(X)`.
For the second, `\frac{d}{dt}\operatorname{tr}(M^2) = \operatorname{tr}(\dot MM + M\dot M) = 2\operatorname{tr}(M\dot M) = 2[\operatorname{tr}(MX)+\operatorname{tr}(MX^\top)]`,
and `\operatorname{tr}(MX^\top) = \operatorname{tr}\big((MX^\top)^\top\big) = \operatorname{tr}(XM) = \operatorname{tr}(MX)`
using `M = M^\top`. ∎

### 4.3 The velocity of the effective rank

Since `\log\mathrm{PR} = 2\log\operatorname{tr}(M) - \log\operatorname{tr}(M^2)`,

> **(4.4) [EXACT]**
> $$\frac{d}{dt}\log\mathrm{PR} \;=\; \frac{4\operatorname{tr}(X)}{\operatorname{tr}(M)} \;-\; \frac{4\operatorname{tr}(MX)}{\operatorname{tr}(M^2)} .$$

Four traces of matrices you already have. **[MEASURED]** against finite differences of
`\log\mathrm{PR}` on trained CReLU networks: relative error `7\times10^{-8}` to
`2\times10^{-6}`.

---

## 5. The criterion, and what `Cheb` means

> **(5.1) Theorem 21 [EXACT].** With `\omega_k := x_k/\mu_k` the relative growth rate,
> $$\frac{d}{dt}\log \mathrm{PR} \;=\; \frac{-2\,\mathrm{Cheb}}{\operatorname{tr}(M)\operatorname{tr}(M^2)}, \qquad \mathrm{Cheb} \;:=\; \sum_{k,j}\mu_k\mu_j\,(\omega_k-\omega_j)(\mu_k-\mu_j).$$
> Since the prefactor is positive, **`PR` falls if and only if `\mathrm{Cheb} \ge 0`.**

*Proof.* Put (4.4) over a common denominator: the numerator is
`4[\operatorname{tr}(X)\operatorname{tr}(M^2) - \operatorname{tr}(MX)\operatorname{tr}(M)]`.
Using (4.2) this is `4[\sum_kx_k\sum_j\mu_j^2 - \sum_k\mu_kx_k\sum_j\mu_j]`. Now expand
`\mathrm{Cheb}`, writing `\mu_k\mu_j(\omega_k-\omega_j) = x_k\mu_j - x_j\mu_k`:

$$\mathrm{Cheb} = \sum_{k,j}(x_k\mu_j - x_j\mu_k)(\mu_k-\mu_j) = \sum_{k,j}\big[x_k\mu_j\mu_k - x_k\mu_j^2 - x_j\mu_k^2 + x_j\mu_k\mu_j\big].$$

The first and fourth double sums each equal `\operatorname{tr}(MX)\operatorname{tr}(M)`, the
second and third each equal `\operatorname{tr}(X)\operatorname{tr}(M^2)` (relabel `k\leftrightarrow j`).
So `\mathrm{Cheb} = 2\operatorname{tr}(MX)\operatorname{tr}(M) - 2\operatorname{tr}(X)\operatorname{tr}(M^2)`,
which is `-\tfrac12` times the numerator. ∎ Verified symbolically.

**What `Cheb` is.** Each term compares two directions. If the one with the larger eigenvalue
also has the larger *relative* growth rate, then `(\omega_k-\omega_j)` and `(\mu_k-\mu_j)`
share a sign and the term is positive. `\mathrm{Cheb}` is the `\mu_k\mu_j`-weighted sum of
those comparisons — a **covariance between the relative growth rate and the eigenvalue**. So
Theorem 21 says exactly:

> **The effective rank falls if and only if bigger directions grow relatively faster.**

That is the rich-get-richer statement, as an exact equivalence, with no assumption anywhere.
Note the two degenerate cases it gets right: if `\omega` is the same for every direction the
sum vanishes (a uniform rescaling changes no ratio), and if the spectrum is flat it vanishes
too (nothing to concentrate).

---

## 6. When the criterion can be evaluated in advance

Theorem 21 is a *condition*, not a prediction. To predict, one needs to know how `\omega_k`
depends on `\mu_k`. One case closes completely.

> **(6.1) Lemma 22 [EXACT].** Suppose `\omega_k = C\mu_k^{\,\theta-1}`, i.e. `x_k = C\mu_k^{\,\theta}`.
> Write `f(p) := \sum_k\mu_k^{\,p}` for the spectral moment function. Then
> $$\mathrm{Cheb} \;=\; 2C\big[f(1+\theta)f(1) - f(\theta)f(2)\big] \;\ge\; 0 \iff \theta \ge 1$$
> (for `C>0`, strictly unless the spectrum is degenerate).

*Proof.* Substituting `x_k = C\mu_k^\theta` into the trace form of `\mathrm{Cheb}` gives the
bracket directly. For the sign: `p\mapsto\log f(p)` is convex (Hölder), and the two exponent
pairs `\{1+\theta,1\}` and `\{\theta,2\}` have the same sum `2+\theta`. For a log-convex `f`,
`f(a)f(b)` at fixed `a+b` increases with `|a-b|`. Here `|(1+\theta)-1| = \theta` against
`|\theta-2| = |2-\theta|`, and `\theta > 2-\theta \iff \theta>1`. ∎
**[MEASURED]** 0 violations over `2\times10^5` random spectra and exponents.

> **(6.2) Corollary [EXACT, given balance and alignment].** For a balanced depth-`L` network
> under pure growth (`\dot J = cJ`-like forcing, i.e. `G = -cJ`), every layer contributes
> `\mu^{1+(L-l)/L+(l-1)/L}` to `x`, independently of `l`, so `\theta = 2 - 1/L`. Hence
> **`PR` strictly falls for every `L \ge 2`**, with equality at `L = 1`.

A single matrix rescaled uniformly does not change its effective rank; two or more layers do.
This is the low-rank bias, and the mechanism is **Hölder's inequality**, not alignment.

The hypotheses of (6.2) are exactly the ones (6.1) does *not* need: Lemma 22 holds for any
network whose relative rate happens to follow a power law, and `\theta` is measurable.

---

## 7. What the criterion says about real networks

**[MEASURED]** `d/dt\log\mathrm{PR}` from (4.4), CReLU on MNIST after 400 steps:

| init | `L` | `d/dt\log\mathrm{PR}` | |
|---|---|---|---|
| looks-linear | 4 | −0.0220 | rank **falls** |
| looks-linear | 16 | −0.3796 | rank **falls** |
| Xavier | 4 | **+0.1456** | rank **rises** |
| Xavier | 16 | **+0.0076** | rank **rises** |

From a near-isometric start the network *acquires* low-rank structure; from a random start —
where it begins collapsed, effective rank 1.5–5 of 12 — training slowly *undoes* it. **The
bias is something a well-conditioned network develops, not something a badly conditioned one
suffers.**

**Caveat.** Single snapshots, one task. The sign is exact and assumption-free at the point
measured; a trajectory across depths, seeds and datasets has not been run and is the obvious
next experiment.

---

## 8. Ledger

**Proved with no assumptions beyond a simple spectrum:** (1.3) the per-pattern velocity;
(4.1) `\dot\mu_k = 2x_k`; (4.2) the two traces; (4.3), (4.4) the derivatives; **Theorem 21**,
the exact criterion. None of these mentions gates, so all hold for deep linear, FGLN, ReLU
and CReLU alike.

**Proved under stated hypotheses:** Lemma 22 needs the relative rate to be a power law in the
eigenvalue (measurable); Corollary 6.2 additionally needs balance, alignment, and pure-growth
forcing.

**Measured, not proved:** §7's signs; `\theta` on real tasks; the claim that a random
initialization supplies the collapse (its initial separation grows linearly in `L` — 3.5, 8.2,
15.1, 32.2 nats at `L = 2,4,8,16` — and training conserves it, final `r/r(0)` = 0.99, 1.01).

**Not established.** Predicting `\mathrm{Cheb}` or `\theta` for a general network and loss.
This is the same difficulty as before, but it is now posed on trace moments, so no basis is
chosen and no alignment assumption can arise. Whether it is easier there is unknown.
Everything is width ≤ 16, depth ≤ 32, two task families.

---

## 9. What this supersedes, and why

The earlier framework tracked per-direction gains `c_k := \sum_l\|A_l^\top u_k\|^2\|B_lv_k\|^2`
and split `\dot s_k = -c_kg_k` into "architecture" and "task". Writing `\dot s_k = (\dot J)_{kk}`
in the operator's bases and expanding (1.3),

$$\dot s_k = -\sum_l\sum_{i,j}(\tilde A_l)_{ki}\,\tilde G_{ij}\,(\tilde B_l)_{jk},\qquad \tilde A_l = U^\top A_lA_l^\top U,\ \tilde B_l = V^\top B_l^\top B_lV,$$

that split keeps only `(i,j)=(k,k)`. Discarding the rest is the **alignment assumption**, and
under its failure direction `k` is driven by gradient components in *other* directions.

**[MEASURED]**, comparing the exponent from the exact velocity with the one from that
surrogate, on networks trained 400 steps:

| model | init | task | exact | surrogate |
|---|---|---|---|---|
| CReLU | looks-linear | MNIST | 4.668 | 4.379 |
| CReLU | Xavier | teacher–student | **−0.016** | **+0.734** |
| CReLU | Xavier | MNIST | **−0.050** | **+0.686** |
| deep linear | Xavier | MNIST | **−0.003** | **+0.667** |

Six of six Xavier cells, both architectures, both tasks: the surrogate reports a low-rank
bias where the exact dynamics have none. Wrong in **sign**, not magnitude. Defining
`g_k := -\dot s_k/c_k` makes the split an identity but does not help — `g_k` then absorbs the
off-diagonal coupling, so it is no longer "the task".

Results from that framework that **survive**, because they never used the assumption: the
exact velocity (1.3); that realized and random gate patterns give indistinguishable rates
(correlation `+0.997` over 408 snapshots), so no ensemble-to-data transfer argument is
needed; and, for CReLU specifically, that one gradient step from a looks-linear configuration
gives `\Delta S_l = -\tfrac\eta2\mathbb E_b[R_b]` and `\Delta\Delta_l = -\tfrac\eta2\mathbb E_b[R_bE_b]`
exactly — so the residual/gate-sign correlation is the only source of nonlinearity, and a
batch closed under negation makes it vanish term by term (`\Delta` stays at `2\times10^{-16}`
over 2000 steps).

What does **not** survive is the claim that the architecture contributes a `2-2/L`
amplification to the *spectral dynamics*. That is a statement about the induced step's
diagonal, and it predicts the spectrum only where the alignment assumption approximately
holds — near-isometric networks, not the regime one trains in from a random start.
