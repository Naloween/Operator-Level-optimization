# Dropping the alignment assumption: spectral dynamics without a basis

Answers the objection that (A2) of [`00-derivation.md`](00-derivation.md) — the context Grams
being diagonal in the operator's singular bases — supports everything else and fails where it
matters. It does, and it does. This file removes the need for it.

---

## 1. Why (A2) appeared at all

Tracking individual singular values requires the singular *vectors*, and collapsing the
velocity to `ṡ_k = −c_k g_k` then requires the Grams `A_lA_lᵀ`, `B_lᵀB_l` to be diagonal in
those vectors. Measured, that fails qualitatively away from near-isometric networks: the
diagonal surrogate reports a low-rank bias (`ψ ≈ +0.7`) where the exact dynamics have none
(`ψ ≈ 0`), in six of six Xavier cells across two architectures and two tasks.

The fix is to stop asking for individual `s_k`.

---

## 2. Basis-free functionals

Let `M := JᵀJ`, with eigenvalues `μ_k = s_k²`. For any `p`,

$$\frac{d}{dt}\operatorname{tr}(M^p) \;=\; 2p\,\operatorname{tr}\!\big(M^{p-1}J^\top \dot J\big) \tag{2.1}\quad\textbf{[EXACT]}$$

— no eigenvectors appear. Define the **participation ratio**

$$\mathrm{PR} \;:=\; \frac{\operatorname{tr}(M)^2}{\operatorname{tr}(M^2)},$$

a smooth effective rank: `PR = d` for an isometry, `PR = 1` for a rank-one operator. From
(2.1), `d/dt\log \mathrm{PR}` is four traces — computable in any network with nothing assumed.
**[MEASURED]** against finite differences of `\log\mathrm{PR}` on trained networks: relative
error `7\times10^{-8}` to `2\times10^{-6}`.

---

## 3. The criterion, as an exact equivalence

Let `x_k` be the diagonal of `J^\top\dot J` in `M`'s own eigenbasis — which the traces select
automatically, with nothing assumed. Note `x_k/μ_k = ω_k`, the log-velocity of file 00.

> **Theorem 21 [EXACT].**
> $$\frac{d}{dt}\log \mathrm{PR} \;=\; \frac{-2\,\mathrm{Cheb}}{\operatorname{tr}(M)\operatorname{tr}(M^2)}, \qquad \mathrm{Cheb} := \sum_{k,j}\mu_k\mu_j\Big(\frac{x_k}{\mu_k}-\frac{x_j}{\mu_j}\Big)\big(\mu_k-\mu_j\big).$$
> Hence **the effective rank falls if and only if the relative growth rate `ω_k` is
> positively correlated with the eigenvalue `μ_k`**, in the Chebyshev sense above.

*Proof.* `d/dt\operatorname{tr}M = 2\operatorname{tr}(X)` and
`d/dt\operatorname{tr}M^2 = 4\operatorname{tr}(MX)` with `X := J^\top\dot J`, by (2.1). So
`d/dt\log\mathrm{PR} = 4\operatorname{tr}(X)/\operatorname{tr}(M) - 4\operatorname{tr}(MX)/\operatorname{tr}(M^2)`,
whose numerator over the common denominator is
`4[\operatorname{tr}(X)\operatorname{tr}(M^2) - \operatorname{tr}(MX)\operatorname{tr}(M)]`. Expanding in
`M`'s eigenbasis, `\operatorname{tr}(X)=\sum_k x_k`, `\operatorname{tr}(MX)=\sum_k\mu_k x_k`, and
symmetrising the double sum gives `-2\,\mathrm{Cheb}`. Verified symbolically. ∎

**No alignment, no singular vectors, no hypothesis.** The *condition* for the low-rank bias
was always assumption-free; what needed (A2) was **predicting** it from the architecture. That
prediction can now be attempted on trace moments, which are boundable without choosing a basis.

---

## 4. One case where the prediction becomes a theorem

> **Lemma 22 [EXACT].** If `x_k = C\mu_k^{\theta}` then
> `\mathrm{Cheb} \ge 0 \iff f(1+\theta)f(1) \ge f(\theta)f(2)` for the spectral moment
> function `f(p) := \sum_k \mu_k^{p}`, and since `f` is log-convex (Hölder) this holds
> **iff `\theta \ge 1`**, strictly unless the spectrum is degenerate.

*Proof.* Substituting `x_k = C\mu_k^\theta` into `\mathrm{Cheb}` gives
`2C[f(1+\theta)f(1) - f(\theta)f(2)]`. Both exponent pairs sum to `2+\theta`; log-convexity of
`f` means the pair with the larger separation dominates, and `|(1+\theta)-1| = \theta` against
`|\theta-2| = |2-\theta|`. ∎ **[MEASURED]** 0 violations over `2\times10^5` random spectra.

> **Corollary 22.1.** For a balanced depth-`L` network under pure growth (`G = -cJ`), each
> layer contributes `\mu^{1+(L-l)/L+(l-1)/L}` to `x`, independent of `l`, so
> `\theta = 2 - 1/L`. Hence **the effective rank strictly falls for every `L \ge 2`**, with
> equality at `L = 1`.

A single matrix rescaled uniformly does not change its effective rank; two or more layers do.
That is the low-rank bias, obtained without ever diagonalising a Gram matrix — the mechanism
is Hölder, not alignment.

---

## 5. What the exact criterion says about real networks

**[MEASURED]** `d/dt\log\mathrm{PR}` on CReLU networks after 400 steps on MNIST:

| init | `L` | `d/dt \log \mathrm{PR}` | |
|---|---|---|---|
| looks-linear | 4 | −0.0220 | rank **falls** |
| looks-linear | 16 | −0.3796 | rank **falls** |
| Xavier | 4 | **+0.1456** | rank **rises** |
| Xavier | 16 | **+0.0076** | rank **rises** |

Read with everything else in `theory/`, this is consistent and it inverts the usual story:
from a near-isometric start the network *acquires* low-rank structure, while from a random
start — where it begins collapsed, effective rank 1.5–5 out of 12 — training slowly *undoes*
it. The bias is something a well-conditioned network develops, not something a badly
conditioned one suffers.

**Caveat.** These are single snapshots at one training time, on one task. The sign is exact
and assumption-free at the point measured; a trajectory across depths, seeds and datasets is
the obvious next experiment and has not been run.

---

## 6. Status

| Result | Statement | Status |
|---|---|---|
| (2.1) | `d/dt tr(M^p) = 2p tr(M^{p-1}J^ᵀ\dot J)` | exact; matches finite differences to 1e-7 |
| Thm 21 | rank falls ⟺ `ω_k` correlates with `μ_k` | **exact, no assumptions**; symbolic |
| Lemma 22 | power-law rate: rank falls ⟺ `θ ≥ 1` | proved (Hölder); 2e5 instances |
| Cor. 22.1 | balanced, pure growth: `θ = 2 - 1/L`, falls for `L ≥ 2` | proved for that case |
| §5 | rank falls at looks-linear, rises at Xavier | measured, snapshots only |

**Still open.** Predicting `θ` — or bounding `Cheb` — for a general network and a general
loss. That is the same difficulty as before, but it is now posed on trace moments rather than
on individual directions, so no basis has to be chosen and (A2) never arises. Whether it is
easier there is not yet known.
