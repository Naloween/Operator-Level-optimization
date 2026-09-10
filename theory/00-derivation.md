# Spectral dynamics of deep factored networks: a self-contained derivation

This file is the audit document. It defines every object used anywhere in `theory/`,
derives every claim from scratch, and tags each step as **[EXACT]**, **[ASSUMPTION]**, or
**[MEASURED]**. Nothing is assumed from the other files; where a result is proved elsewhere
the proof is reproduced here in full.

Read §8 first if you want the ledger of what is and is not established.

---

## 1. Setup

**Network.** `L` layers with weights `W_1, …, W_L`. Between them sit **gates** `D_l`, matrices
that may depend on the input. The **operator** (input–output Jacobian) is

$$J \;=\; W_L\,D_{L-1}\,W_{L-1}\cdots D_1\,W_1 . \tag{1.1}$$

Four instances, all covered by what follows:

| model | gate `D_l` | shape |
|---|---|---|
| deep linear | `I` | `d×d` |
| FGLN | fixed diagonal | `d×d` |
| ReLU MLP | `diag(1[z>0])` | `d×d` |
| CReLU MLP | `[diag(1[z>0]); −diag(1[z<0])]` | `2d×d` |

**Gate pattern.** A *pattern* `ε` is a choice of all the gates, made by hand rather than read
off an input. Write `J_ε` for the resulting operator. A pattern need not be realized by any
input. When `ε = ε(x)` is the pattern input `x` produces, `J_ε = J(x)` — verified to `1e-14`.

**Contexts.** For each layer `l`, split the product either side of `W_l`:

$$A_l := W_L D_{L-1}\cdots W_{l+1}D_l, \qquad B_l := D_{l-1}W_{l-1}\cdots W_1, \qquad J = A_l\,W_l\,B_l . \tag{1.2}$$

This holds for every `l` simultaneously. `A_L = I`, `B_1 = I`.

**Spectrum.** `J = \sum_k s_k u_k v_k^\top` with `s_1 \ge \dots \ge s_d \ge 0`. A **direction**
is an index `k`. (Files 01–03 call a *pattern* a "mode"; files 04–06 call a *direction* a
"mode". Neither word is used below.)

**Weight gradient.** `Γ_l := ∂\mathcal{L}/∂W_l`, one matrix per layer, computed from the real
loss on the real data.

---

## 2. What "low-rank bias" means

Put `m_k := \log s_k` (for `s_k > 0`). Then

* `\bar m = \frac1d\sum_k m_k = \log \bar s` where `\bar s` is the geometric mean — the
  operator's **scale**;
* the spread of `m` is its **shape**. Define the **separation** between two directions
  `r_{jk} := m_j - m_k = \log(s_j/s_k)`.

Multiplying every `s_k` by one constant moves `\bar m` and leaves every `r_{jk}` fixed. So
the low-rank bias is a statement about the *spread* of `m` and about nothing else. Condition
number and effective rank are both functions of that spread.

---

## 3. The exact dynamics

### 3.1 Derivative of a singular value

> **(3.1) [EXACT, given A1]** If `s_k` is a **simple** singular value and `J(t)` is
> differentiable, then `\dot s_k = u_k^\top \dot J\, v_k`.

*Proof.* Simplicity makes `u_k, v_k` differentiable, and `s_k = u_k^\top J v_k`. Then

$$\dot s_k = \dot u_k^\top J v_k + u_k^\top \dot J v_k + u_k^\top J \dot v_k = s_k\,\dot u_k^\top u_k + u_k^\top \dot J v_k + s_k\, v_k^\top \dot v_k,$$

using `Jv_k = s_k u_k` and `u_k^\top J = s_k v_k^\top`. Unit norm gives
`\dot u_k^\top u_k = \tfrac12 \tfrac{d}{dt}\|u_k\|^2 = 0`, likewise for `v_k`. ∎

> **(A1) Simple spectrum.** At a crossing of singular values the derivative is only
> one-sided. **[ASSUMPTION]**, generic, standard.

### 3.2 Velocity at a fixed pattern

> **(3.2) [EXACT]** Hold the pattern `ε` fixed. Under `\dot W_l = -Γ_l`,
> $$\dot s_k(\varepsilon) \;=\; -\sum_{l=1}^{L} u_k^\top A_l^\varepsilon\, Γ_l\, B_l^\varepsilon\, v_k .$$

*Proof.* With the gates fixed, `J_ε` is a polynomial in the weights and the product rule
gives `\dot J_\varepsilon = \sum_l A_l^\varepsilon \dot W_l B_l^\varepsilon` from (1.2).
Substitute into (3.1). ∎

Two things this buys, and they are the reason for working at fixed patterns:

* `t \mapsto J_\varepsilon(t)` is smooth, whereas `t \mapsto J(x)` jumps when an input crosses
  a region boundary;
* `Γ_l` is **one matrix shared by every pattern**. All cross-input coupling is inside it.
  Different patterns are the *same* dynamics composed differently.

Verified against finite differences of `J_ε`'s singular values to `1e-6`.

### 3.3 Separation

> **(3.3) [EXACT]** With `\omega_k := \dot s_k/s_k` (the **log-velocity**),
> $$\dot r_{jk} = \omega_j - \omega_k .$$

*Proof.* `\dot m_k = \dot s_k/s_k = \omega_k` by the chain rule; `r_{jk} = m_j - m_k`. ∎

> **(3.4) [EXACT]** Let `\lambda` be the least-squares slope of `\omega` on `m` across
> directions, and `e_k` the residual: `\omega_k = a + \lambda m_k + e_k`. Then
> $$\dot r_{jk} = \lambda\, r_{jk} + (e_j - e_k) \qquad\text{identically.}$$

*Proof.* Substitute the regression form into (3.3). The decomposition defines `e`, so
nothing is assumed. ∎

**This is where `d\log(\text{separation})/dt` comes from, and it is exact.** `λ > 0` means
every separation grows; the residual is the part not explained by a size-dependent rate.
Note also that the *mean* of `ω` cancels from (3.3) entirely: it is pure rescaling.

---

## 4. The gain, and the one place alignment enters

### 4.1 Definition

> **(4.1) [DEFINITION]** The **gain** of direction `k` is
> $$c_k := \sum_l \big\|A_l^\top u_k\big\|^2\,\big\|B_l v_k\big\|^2 .$$

`c_k` is computable in any network with no hypothesis. Its meaning:

> **(4.2) [EXACT]** `c_k` is the `(k,k)` entry of the induced operator step
> `\Delta J = -\eta\sum_l A_lA_l^\top G B_l^\top B_l` under `G = u_kv_k^\top`.

*Proof.* `u_k^\top \big(\sum_l A_lA_l^\top u_kv_k^\top B_l^\top B_l\big) v_k
= \sum_l (u_k^\top A_lA_l^\top u_k)(v_k^\top B_l^\top B_l v_k)`, which is the display. ∎
Checked against that sum to `1e-9`.

### 4.2 The factorization `\dot s_k = -c_k g_k`

Suppose the gradient is a true gradient flow on a shared operator, `Γ_l = A_l^\top G B_l^\top`
with `G = ∂\mathcal L/∂J`. Then (3.2) reads
`\dot s_k = -\sum_l u_k^\top A_lA_l^\top G B_l^\top B_l v_k`. To collapse this to `-c_k g_k`
with `g_k := u_k^\top G v_k` one needs the Gram matrices `A_lA_l^\top` and `B_l^\top B_l` to be
**diagonal in the bases `\{u_k\}, \{v_k\}`** — i.e. each subproduct aligned with the whole
product.

> **(A2) Alignment.** `U^\top A_lA_l^\top U` and `V^\top B_l^\top B_l V` are diagonal.
> **[ASSUMPTION]**, and **it fails qualitatively in the regime that matters.**

Being precise about what fails. Since `\dot s_k = (\dot J)_{kk}` in the operator's own bases
is exact, and `\dot J = -\sum_l A_lA_l^\top G B_l^\top B_l`, the exact velocity is

$$\dot s_k = -\sum_l \sum_{i,j} (\tilde A_l)_{ki}\, \tilde G_{ij}\, (\tilde B_l)_{jk}, \qquad \tilde A_l = U^\top A_lA_l^\top U,\ \tilde B_l = V^\top B_l^\top B_l V,\ \tilde G = U^\top G V .$$

The reduction keeps only `(i,j) = (k,k)`. **Under (A2)'s failure, direction `k`'s singular
value is driven by gradient components in *other* directions**, routed through the Grams'
off-diagonal. That is a coupling, not a small correction.

**[MEASURED]** on networks trained 400 steps, comparing the exponent computed from the exact
velocity against the one from the diagonal surrogate:

| model | init | task | `L` | rel. error | cos | `ψ` exact | `ψ` diagonal |
|---|---|---|---|---|---|---|---|
| CReLU | looks-linear | MNIST | 16 | 0.023 | 1.000 | 4.668 | 4.379 |
| CReLU | looks-linear | teacher–student | 16 | 0.013 | 1.000 | −0.266 | −0.354 |
| CReLU | Xavier | teacher–student | 16 | 0.229 | 0.998 | **−0.016** | **+0.734** |
| CReLU | Xavier | MNIST | 16 | 0.439 | 0.968 | **−0.050** | **+0.686** |
| deep linear | Xavier | MNIST | 16 | 0.362 | 0.981 | **−0.003** | **+0.667** |
| deep linear | Xavier | teacher–student | 16 | 0.805 | 0.899 | **−0.083** | **+0.667** |

**Near the linear manifold the reduction is accurate (1–5% error, exponents agreeing to
~0.2). At Xavier it gets the sign wrong**: the diagonal surrogate reports `ψ ≈ +0.7` — a
low-rank bias — where the exact dynamics have `ψ ≈ 0`, none. High cosine does not save it,
because the disagreement is in how the velocity is *distributed across directions*, which is
exactly what an exponent measures.

**Consequence, stated bluntly.** Everything built on `c_k` (§5, including Theorem 20) is a
rigorous theory of the *induced step's diagonal*. It predicts spectral dynamics only where
(A2) approximately holds — i.e. near-isometric networks. From a random initialization it does
not, and asserting `2-2/L` there would be wrong in sign, not merely in magnitude.

**The way around (A2), used everywhere below.** Do not assume it. *Define*

> **(4.3) [DEFINITION]** `g_k := -\dot s_k / c_k`, so that `\dot s_k = -c_k g_k` holds
> **identically**.

This is legitimate but changes the meaning: `g_k` is no longer "the loss gradient's component
on direction `k`". It is the residual after dividing out a known geometric factor. What makes
that a genuine decomposition rather than a tautology is that `c_k` is computed independently
of `\dot s_k`, and §5 shows `c_k` has structure.

Everything from here uses (4.3), so **(A2) is not assumed anywhere below.**

---

## 5. The gain's exponent

### 5.1 The balanced closed form

Suppose the factorization is *balanced and aligned*: all matrices simultaneously
diagonalizable with layer `l` carrying singular value `a_{l,k}` on direction `k`, and all
`a_{l,k} = s_k^{1/L}`.

> **(5.1) [EXACT, given balance+alignment]** `c_k = L\,s_k^{2-2/L}`.

*Proof.* Diagonality gives `\|A_l^\top u_k\|^2 = \prod_{j>l}a_{j,k}^2` and
`\|B_lv_k\|^2 = \prod_{j<l}a_{j,k}^2`, so
`c_k = \sum_l \prod_{j\ne l}a_{j,k}^2 = s_k^2\sum_l a_{l,k}^{-2}`, using
`s_k = \prod_l a_{l,k}`. Balance gives `\sum_l a_{l,k}^{-2} = L s_k^{-2/L}`. ∎

This is the classical `2-2/L` rich-get-richer exponent (Saxe 2014; Arora et al. 2019).

### 5.2 Dropping balance

Write `a_{l,k} = s_k^{1/L}x_{l,k}^{-1/2}`, forcing `\prod_l x_{l,k} = 1`, and

> **(5.2) [DEFINITION]** `c_k = s_k^{2-2/L}K_k`, `K_k := \sum_l x_{l,k}`, so
> $$\frac{d\log c}{d\log s} = \Big(2-\frac2L\Big) + \frac{d\log K}{d\log s} \tag{5.3}$$
> where both derivatives mean least-squares slopes across directions. **[EXACT]** — (5.3) is
> arithmetic once `K` is defined, whatever the network.

> **(5.4) [EXACT, aligned]** `K_k \ge L`, equality iff direction `k` is balanced.
> *Proof.* `\prod_l x_{l,k} = 1`, so AM–GM gives `\sum_l x_{l,k}\ge L(\prod_l x_{l,k})^{1/L}=L`. ∎

> **(5.5) [EXACT, aligned]** If `x_{l,k}` does not depend on `k`, then `K` is constant across
> directions, `d\log K/d\log s = 0`, and the exponent is *exactly* `2-2/L` however extreme
> the imbalance. **[MEASURED]** to `2\times10^{-15}` with a 20× bottleneck layer.
> *Proof.* `\log c_k = (2-2/L)\log s_k + \log K`, affine with slope `2-2/L`. ∎

So *balancedness* was never the necessary hypothesis for deep linear networks;
*direction-independence of the imbalance* is, and it is far weaker. **[MEASURED]**: this does
**not** transfer to CReLU — a layerwise rescaling that provably leaves every gate pattern
identical (0 gate flips, `\|\Delta J\|/\|J\| = 3\times10^{-16}`) still moves the exponent by up
to 0.33.

### 5.3 Theorem 20: a certificate with no alignment assumption

> **(5.6) Lemma A [EXACT].** Let `x,y\in\mathbb R^n`, `x` non-constant, `b` the least-squares
> slope of `y` on `x`. If every `y_i` lies in an interval of length `w`, then
> `|b| \le w/(2\,\mathrm{sd}(x))`.

*Proof.* `\langle\tilde x,\mathbb 1\rangle = 0` gives
`\langle\tilde x,\tilde y\rangle = \langle\tilde x, y-c\mathbb 1\rangle` for any `c`. Take `c`
the band's **midpoint**, so `|y_i-c|\le w/2`. Cauchy–Schwarz:
`|b| \le \|y-c\mathbb 1\|/\|\tilde x\| \le (w/2)\sqrt n/(\sqrt n\,\mathrm{sd}(x))`. ∎
Tight at `x=(-1,1), y=(0,w)`; worst ratio `0.912` over `2\times10^5` random instances.

> **(5.7) Lemma B [EXACT, no hypotheses at all].** With `g := \prod_l\|W_l\|_2`,
> $$c_k \;\ge\; L\,s_k^2\,g^{-2/L} .$$

*Proof.* For each `l`, put `a = A_l^\top u_k`, `b = B_lv_k`. Then
`s_k = u_k^\top A_lW_lB_lv_k = \langle a, W_lb\rangle \le \|a\|\|W_l\|\|b\|`, so the `l`-th term
of `c_k` is `\ge s_k^2/\|W_l\|^2 \ge 0`. AM–GM over the `L` terms gives
`c_k \ge L(\prod_l s_k^2/\|W_l\|^2)^{1/L} = Ls_k^2g^{-2/L}`. ∎
Tight for an orthogonal chain (ratio exactly `1.0000`). This is (5.4) without alignment; the
price is the factor `(s_k/g)^{2/L}`, which tends to 1 with depth.

> **(5.8) Theorem 20 [EXACT].** Let `σ` be the standard deviation of `\log s_k` over the
> directions with `s_k>0`, and let all `\log K_k` lie in a band of width `w`. Then
> $$\Big|\frac{d\log c}{d\log s} - \Big(2-\frac2L\Big)\Big| \;\le\; \frac{w}{2σ} .$$
> With Lemma B supplying the band's floor, `w \le \log(Λ/L) + (2/L)\log(g/s_{\min})` where
> `Λ := \max_k K_k` is the single measured input.

*Proof.* By (5.3) the left side is `|d\log K/d\log s|`; apply Lemma A. ∎

**[MEASURED]**, 0 violations, across **deep linear, FGLN, ReLU MLP and CReLU MLP**, depths
4–32, six seeds × six random gate patterns:

| regime | certified bound | measured deviation |
|---|---|---|
| near the linear manifold | **0.021 – 0.065** | 0.002 – 0.005 |
| far from it (`δ ≈ 1`) | 0.27 – 0.79 | 0.15 – 0.49 |

Nothing in Lemmas A, B or Theorem 20 mentions gates, CReLU, balance or alignment.

---

## 6. Architecture versus task

### 6.1 The exponent decomposition

From `\omega_k = \dot s_k/s_k = -c_kg_k/s_k` (using (4.3)), taking logs,
`\log|\omega_k| = \log c_k + \log|g_k| - m_k`. Slopes of logs add:

> **(6.1) [EXACT]** With `ψ := d\log|\omega|/d\log s` and `p := d\log|g|/d\log s`,
> $$ψ \;=\; \frac{d\log c}{d\log s} - 1 + p \;=\; \Big(1-\frac2L\Big) + p + \underbrace{\frac{d\log K}{d\log s}}_{|\cdot|\ \le\ w/2σ\ \text{by (5.8)}} .$$

**[MEASURED]**: the identity holds to `1.8\times10^{-15}`.

So the exponent splits into an **architecture** term `φ := 1-2/L` (proved, universal over
patterns, saturating at 1), a **task** term `p` (one measurable number), and a **remainder**
certified by Theorem 20.

### 6.2 What `p` is

`p` measures how the effective per-direction drive scales with the direction's size —
equivalently, how the target's spectrum sits relative to the operator's current one. `p>0`:
the loss pushes hardest on directions already large, so architecture and task pull together.
`p<0`: the task opposes the bias. **[MEASURED]**

| task | init | median `p` | |
|---|---|---|---|
| MNIST | looks-linear | **+10.95** | reinforces |
| MNIST | Xavier | −1.03 | opposes |
| teacher–student (orthogonal target) | Xavier | −1.40 | opposes |

Mechanism: ten classes through width 16 make the operator gradient effectively rank ≤ 10.

### 6.3 The gap I had wrong, and it matters

`ψ` and `λ` (from (3.4)) are **different slopes**: `ψ` is the slope of `\log|\omega|` on `m`,
`λ` the slope of `\omega` on `m`. They coincide only if `\omega` is a power law in `s`:

> **(A3) Power law.** `\omega_k \approx -C s_k^{ψ}`, under which `λ \approx ψ\,\bar\omega`.
> **[ASSUMPTION]**. **[MEASURED]** and it is *weak here*: the ratio `λ/(ψ\bar\omega)` ranges
> **0.37 – 2.77**, with `R^2` of the power-law fit at **0.06 – 0.59**.

An earlier version of `07-patterns.md` wrote (6.1) with `λ` on the left. That is wrong;
only `ψ` obeys it. The consequence is important and limiting:

* **`ψ` is what decomposes** into architecture + task + certified remainder.
* **`λ` is what drives separation**, exactly, by (3.4).
* The bridge between them is (A3), which does not hold well in the measured regimes.

What survives without (A3) is the **sign**: **[MEASURED]** `\mathrm{sign}(\Delta r) = \mathrm{sign}(λ)`
in **100%** of intervals where the power-law form fits (`R^2\ge0.9`, n=148) and 89% over all
519. So the direction of the effect is reliable; its magnitude via `ψ` is not, unless `R^2`
is high — which is why `R^2` should be reported alongside every such number.

---

## 7. Patterns

**[MEASURED]** Bias rates at realized versus Rademacher gate patterns, 408 snapshots from 48
training runs:

$$\operatorname{corr}(λ_{\text{realized}}, λ_{\text{random}}) = +0.997,\qquad \operatorname{median}|λ_{\text{realized}}-λ_{\text{random}}| = 0.024$$

against a median `|λ|` of `0.138`. The bias is a property of the weights, not of which
regions the data selects. So no transfer argument from a pattern ensemble to the data is
needed — this removes the hypothesis `(H-mode)` of `03-dynamics.md` entirely.

**[MEASURED]** The gain exponent at arbitrary patterns, sweeping the nonlinearity
`δ := \max_l\|\Delta_l\|/\|S_l\|` (CReLU only; `W_l=[P_l|Q_l]`, `S_l=(P_l-Q_l)/2`,
`\Delta_l=(P_l+Q_l)/2`, so `W_lD(z)=S_l+\Delta_l\,\mathrm{diag}(\operatorname{sign}z)` and
`δ=0` iff the network is exactly linear):

| `L` | `2-2/L` | `δ=0` | `0.05` | `0.15` | `0.40` | `0.82` |
|---|---|---|---|---|---|---|
| 8 | 1.750 | 1.744 | 1.748 | 1.736 | 1.649 | 1.554 |
| 32 | 1.938 | 1.936 | 1.937 | 1.922 | 1.860 | 1.890 |

Within 1% to `δ=0.15`, within 15% at `δ=0.8` — far better than the perturbative
`(1+δ)^L\approx10^5` bound at `L=32,δ=0.4`, because a pattern change acts on `c_k` mostly as
a *direction-independent* factor, which cancels in a log-log slope by (5.5).

---

## 8. Ledger

**Proved, no assumptions beyond (A1):** (3.1)–(3.4) the exact dynamics and the separation
identity; (4.2) the gain's meaning; (5.3) the `K` decomposition; Lemma A; **Lemma B**;
**Theorem 20**; (6.1) the exponent identity. Lemmas A, B and Theorem 20 hold for *any*
factored operator — verified on four architectures including ReLU, whose square gates kill
directions outright.

**Proved under stated extra hypotheses:** (5.1) needs balance and alignment; (5.4), (5.5)
need alignment.

**CReLU-specific** (not in this file; `04-instability.md` §4): from a looks-linear
configuration `\Delta S_l = -\tfrac\eta2\mathbb E_b[R_b]` and
`\Delta\Delta_l = -\tfrac\eta2\mathbb E_b[R_bE_b]` exactly, so the residual/gate-sign
correlation is the only source of nonlinearity; a negation-closed batch makes it vanish term
by term and the network stays exactly linear (`2\times10^{-16}` over 2000 steps).

**Assumptions, with measured cost:**

| | statement | cost |
|---|---|---|
| (A1) | simple singular values | generic |
| (A2) | alignment | **1–5% near an isometry; sign-wrong at Xavier** (§4.2) |
| (A3) | `\omega` is a power law in `s` | `λ/(ψ\bar\omega)\in[0.37,2.77]`, `R^2` 0.06–0.59 |

(4.3) makes `\dot s_k = -c_kg_k` an identity and so avoids *stating* (A2) — but it does not
avoid *needing* it. Without (A2) the residual `g_k` absorbs the off-diagonal coupling, and
the split into "architecture `c_k`" and "task `g_k`" stops being a split into architecture and
task. **The decomposition is only interpretable where (A2) approximately holds.**

**Measured, not proved:** `p` on two task families; the gain exponent's robustness in `δ`;
`(H-mode)`'s irrelevance; the depth wall living in `r(0)` (Xavier `r(0)` grows linearly in
`L`: 3.5, 8.2, 15.1, 32.2 nats, and training conserves it — final `r/r(0)` = 0.99, 1.01).

**Not established.** That `c_k` governs spectral dynamics away from near-isometric networks —
measured false at Xavier, where it predicts the wrong sign. Any bound on `p`. Any bound on the remainder in terms of `δ` alone —
**impossible**, since `K` is unbounded above (one layer scale to zero sends `K\to\infty` at
fixed `\prod x_l`). Anything at width > 16 or depth > 32 outside the earlier MNIST sweeps.

---

## 9. The picture in one paragraph

A gated network is, at every instant, a *family* of linear networks indexed by gate patterns,
all sharing one weight gradient. Each pattern's operator moves by
`\Delta J_\varepsilon = -\eta\sum_l A_l^\varepsilon Γ_l B_l^\varepsilon`, which in that
pattern's singular basis multiplies direction `k` by the gain `c_k \propto s_k^{2-2/L}`.
The architecture is therefore an **amplifier with a fixed, depth-set, bounded exponent**,
applied to whatever anisotropy the data and loss present; it does not create low-rank
structure. What the spectrum does is decided by three independent things: the **seed** `r(0)`
(multiplicative, and at random initialization it grows linearly in depth), the **task
exponent** `p` (positive on real data, negative on well-conditioned targets), and the
**architecture exponent** `1-2/L` (saturating, so depth 1024 and 256 differ by 0.6%).
