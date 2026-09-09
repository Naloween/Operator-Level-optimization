# Balancedness was never the hypothesis. Mode-independence was.

Builds on [`04-instability.md`](04-instability.md). Numbering continues from Theorem 17.

File 04's rate law rests on the *balanced* deep-linear flow — every layer carrying
`s^{1/L}` in aligned bases — and that is a bad hypothesis. No trained network satisfies it,
Xavier initialization is far from it, and the 40-cell grid of §5.1 was run on a network made
**diagonal by construction**, so it verified the algebra in the most favourable setting that
exists rather than showing the law applies anywhere. This file replaces the hypothesis.

The answer is not "the exponent degrades with imbalance". It is sharper and better:

> **A layerwise imbalance shared by every mode leaves the bias exponent exactly `2 - 2/L`,
> however extreme it is.** Only imbalance that *varies with the mode* moves it, and then by
> an amount that is measurable rather than assumed.

Measured: a bottleneck layer 20× smaller than its neighbours (`K/L = 24`) leaves
`d log c/d log s` at `1.750000...` against a balanced `1.750`. Meanwhile a bottleneck placed
on the *large modes* drops the same exponent to `-1.57`.

---

## 1. The decomposition

The mode gain of Proposition 3.1 — what weight-space gradient descent multiplies mode `k`'s
gradient by — is, for an aligned diagonal network with layer scales `a_{l,k}`,

$$c_k \;=\; \sum_l \prod_{j\ne l} a_{j,k}^2 \;=\; s_k^2 \sum_l a_{l,k}^{-2}, \qquad s_k = \prod_l a_{l,k}. \tag{1.1}$$

Factor out the balanced part by measuring each layer against the balanced reference
`s_k^{1/L}`. Write `a_{l,k} = s_k^{1/L} x_{l,k}^{-1/2}`, so that `prod_l x_{l,k} = 1`
identically, and

$$\boxed{\;c_k \;=\; s_k^{\,2-2/L}\, K_k, \qquad K_k := \sum_l x_{l,k} = \sum_l \big(a_{l,k}\, s_k^{-1/L}\big)^{-2}.\;} \tag{1.2}$$

> **Lemma 18.** `K_k >= L`, with equality if and only if mode `k` is balanced
> (`a_{1,k} = \cdots = a_{L,k}`).

*Proof.* `prod_l x_{l,k} = prod_l a_{l,k}^{-2} s_k^{2} = s_k^{-2} s_k^{2} = 1`, so by AM–GM
`K_k = sum_l x_{l,k} >= L (prod_l x_{l,k})^{1/L} = L`, with equality iff all `x_{l,k}` are
equal, i.e. all `a_{l,k}` are equal. ∎

`K` is scale-free and is a genuine imbalance measure with a hard floor. Taking logarithms of
(1.2) and differentiating **across modes at fixed time** — which is the direction Lemma 12's
rate lives in, because separation is a comparison *between* modes — gives

$$\frac{d\log c}{d\log s} \;=\; \Big(2 - \frac{2}{L}\Big) \;+\; \frac{d\log K}{d\log s}. \tag{1.3}$$

That much is a definition. The content is what makes the second term vanish.

> **Theorem 19 (mode-independent imbalance is free).** If the normalized layer scales
> `x_{l,k}` do not depend on `k` — equivalently, if every layer is scaled by a factor common
> to all modes — then `K_k` is constant across modes, `d\log K/d\log s = 0`, and the bias
> exponent is exactly `2 - 2/L`, whatever the imbalance.

*Proof.* If `x_{l,k} = x_l` for all `k` then `K_k = sum_l x_l =: K` is a constant, so
`log c_k = (2 - 2/L) log s_k + log K` is affine in `log s_k` with slope exactly `2 - 2/L`. ∎

The proof is two lines; the point is what it removes. **Balancedness (`K = L`) is a special
case of a hypothesis that only needs `K` to be the *same* for every mode, not small.** A
network with a severe bottleneck layer, a badly scaled initialization, or per-layer learning
rates is unbalanced and still has the balanced exponent — provided the distortion is not
mode-selective.

> **Corollary 19.1 (rate and bias are separate knobs).** By Lemma 18, imbalance raises `K`
> above `L` and therefore *increases* the mode gain: an unbalanced network takes larger
> operator steps for the same gradient. By Theorem 19 that costs nothing in bias. Imbalance
> speeds the dynamics up and leaves the exponent alone.

Measured `K/L` at Xavier initialization runs from 3.0 at `L = 4` to **49.6** at `L = 32` —
a fiftyfold speed-up of the operator dynamics — while the exponent moves by less than 0.2.

---

## 2. What actually moves the exponent

Everything is in `d log K / d log s`: how the imbalance correlates with mode size.

**Verified exact.** Over 24 cases (depths 4–32 × six imbalance patterns), the residual
`|slope - (2 - 2/L) - d log K/d log s|` has maximum **2.0e-15**. At `L = 8`:

| imbalance pattern | `K/L` | measured slope | balanced | `d log K/d log s` |
|---|---|---|---|---|
| balanced | 1.00 | 1.7500 | 1.750 | −0.0000 |
| mode-indep: linear in `l` | 10.02 | **1.7500** | 1.750 | −0.0000 |
| mode-indep: 20× bottleneck | 24.23 | **1.7500** | 1.750 | −0.0000 |
| mode-indep: random | 8.87 | **1.7500** | 1.750 | 0.0000 |
| mode-DEP: linear in `k` | 1.23 | 2.1285 | 1.750 | +0.3785 |
| mode-DEP: bottleneck on big modes | 2.35 | **−1.5696** | 1.750 | −3.3196 |

The last row matters: mode-dependent imbalance can not only shrink the bias but **reverse its
sign**, so that larger modes grow *relatively slower*. Nothing about depth prevents that.

---

## 3. What real initializations do

`studies/imbalance.py`, six cells per row (widths 8 and 16 × three seeds), measured with no
alignment assumption (§4).

| model | init | L | balanced | measured | `d log K/d log s` | `K/L` |
|---|---|---|---|---|---|---|
| deep linear | xavier | 4 | 1.500 | 1.206 ± 0.129 | −0.294 ± 0.129 | 3.0 |
| deep linear | xavier | 8 | 1.750 | 1.566 ± 0.056 | −0.184 ± 0.056 | 7.1 |
| deep linear | xavier | 16 | 1.875 | 1.709 ± 0.054 | −0.166 ± 0.054 | 24.7 |
| deep linear | xavier | 32 | 1.938 | 1.835 ± 0.031 | −0.103 ± 0.031 | 42.8 |
| CReLU | xavier | 4 | 1.500 | 1.266 ± 0.073 | −0.234 ± 0.073 | 3.0 |
| CReLU | xavier | 32 | 1.938 | 1.749 ± 0.055 | −0.188 ± 0.055 | 49.6 |
| CReLU | haar | 4 | 1.500 | 0.893 ± 0.102 | −0.607 ± 0.102 | 1.7 |
| CReLU | haar | 32 | 1.938 | 1.727 ± 0.028 | −0.211 ± 0.028 | 21.9 |

Three readings.

1. **The correction is always negative.** Every random initialization tested, at every depth
   and width and seed, has `d log K/d log s < 0`. Random initialization is mode-dependent in
   a way that *weakens* the low-rank bias relative to the balanced prediction. The balanced
   law is an upper bound in practice, not an estimate.
2. **It shrinks with depth** (−0.29 → −0.10 for deep linear), so the balanced law becomes a
   better approximation as networks get deeper, not worse.
3. **It persists through training.** Over 3000 steps on a linear teacher the correction stays
   within about 0.1 of its initial value at every depth (e.g. CReLU `L = 32`: −0.140 → −0.157)
   while `K/L` moves by up to a factor of two. It is a property of the initialization's
   structure, not a transient.

**Looks-linear CReLU and Haar deep linear return `nan`, correctly.** Their operators are
isometries, so `log s` has no spread and no slope exists. Without that guard the regression
fits rounding noise and returns a plausible number; it did, in the first version of this
measurement, and the numbers were nonsense.

---

## 4. The measurement is alignment-free

Everything above except the *interpretation* of `K` in terms of layer scales is measured
without assuming alignment or diagonality. The per-layer contribution to the mode gain is
exactly

$$q_{l,k} = \big\|A_\ell^\top u_k\big\|^2 \big\|B_\ell v_k\big\|^2, \qquad c_k = \sum_\ell q_{l,k},$$

which is the `(k,k)` entry of the induced step `\sum_\ell A_\ell A_\ell^\top G B_\ell^\top B_\ell`
under `G = u_k v_k^\top` — verified against that sum to `1e-9`. So `c_k` and
`K_k = c_k s_k^{2/L-2}` are computable in any network, and (1.3) is then arithmetic. What
needs the aligned diagonal picture is only Lemma 18 and the reading of `K` as a statement
about layers.

---

## 5. Two derivatives that are easy to confuse, and I confused them

An earlier version of this file claimed the exponent becomes `2 - 2/L_{\text{eff}}` with
`L_{\text{eff}} = H^2/H_2` a participation ratio of `H = \sum_\ell a_\ell^{-2}`. That formula
is **correct for the wrong derivative**, and the error is worth recording because the two
quantities look identical on paper.

* **Along-trajectory.** Under gradient flow, `W_{\ell+1}^\top W_{\ell+1} - W_\ell W_\ell^\top`
  is conserved (Du et al. 2018), which pins one mode's layer scales to `a_\ell^2 = t + C_\ell`
  and makes it a one-dimensional system. Differentiating (1.1) along *that* path gives
  `d\log c/d\log s = 2 - 2H_2/H^2`, verified symbolically. This describes how a single mode's
  own gain changes as that mode grows over time.
* **Cross-mode.** Separation is `r_{jk} = \log s_j - \log s_k`, a comparison between *different
  modes at the same time*. That derivative is (1.3), and it is the one Lemma 12's rate uses.

The two coincide at balance and diverge otherwise, which is exactly why the mistake survived
a symbolic check. It was caught by the numbers: the `L_eff` formula predicted errors up to
1.24 where the balanced law was exact to 1e-15. `trajectory_exponent_from_scales` is kept in
`olo.theory.imbalance`, clearly labelled, so the distinction stays visible.

---

## 6. What this settles, and what it does not

**Settles.** The balanced hypothesis in file 04 §2 can be weakened to mode-independence of
the imbalance, which is a much weaker and more plausible condition, and the weakening is
exact rather than approximate (Theorem 19). Whatever else is wrong with the rate law, *it is
not sensitive to the overall balance of the network*.

**Does not settle.**

* The exponent `2 - 2/L` is the *gain*'s exponent. Turning it into a statement about `\dot s_k`
  still needs the modal reduction of file 04 (2.1)–(2.2), and in particular the claim that the
  Gram matrices `A_\ell A_\ell^\top`, `B_\ell^\top B_\ell` are diagonal in `P`'s singular bases.
  That is a genuine alignment assumption and it is **not** established here. It is what your
  Thm 6.1 supplies, with its own "subproduct large enough" condition.
* Whether the measured `d log K/d log s < 0` at random initialization is enough to explain the
  observed conservation of separation in the Xavier runs of file 04 §5.6. The sign is right and
  the effect is consistent, but a correction of −0.2 on an exponent of 1.9 is not obviously
  large enough to account for `r(t)/r(0) ≈ 1.0`, and I have not closed that gap.
* Everything is width ≤ 16, depth ≤ 32, one task family.

| Result | Statement | Status |
|---|---|---|
| Lemma 18 | `K >= L`, equality iff balanced | unconditional (aligned diagonal) |
| (1.3) | `d log c/d log s = (2-2/L) + d log K/d log s` | definitional; verified to 2e-15 |
| Thm 19 | mode-independent imbalance leaves the exponent exact | unconditional |
| Cor. 19.1 | imbalance raises the rate, not the bias | unconditional |
| §3 | random init always gives `d log K/d log s < 0` | measured, 6 cells per row |
