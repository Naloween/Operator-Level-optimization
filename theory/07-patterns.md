# Low-rank bias at a fixed gate pattern: the mechanism is universal, the outcome is not

Builds on [`01-crelu-structure.md`](01-crelu-structure.md) and picks up the question
[`03-dynamics.md`](03-dynamics.md) left open. This is the file that answers *"is there a
low-rank bias in the nonlinear setting, and for which gate patterns?"*

---

## 0. Naming, fixed

Two unrelated objects were both called "mode" across files 01–06. From here:

| term | is | ranges over | files |
|---|---|---|---|
| **pattern** `ε` | a gate pattern — one sign vector per layer, realized or not | `{±1}^{(L−1)d}` | 01–03, 07 |
| **direction** `k` | a singular index of an operator | `1…d` | 04–06 |

"Mode-independent imbalance" in file 05 meant *direction*-independent. `(H-mode)` in file 03
is about *patterns*. The collision was mine.

---

## 1. Why patterns make the problem tractable

Two facts, both exact.

**(a) `J_ε` is smooth where `J(x)` is not.** `J_ε = W_L D(ε_{L−1}) ⋯ D(ε_1) W_1` is a
polynomial in the weights with no discontinuity, whereas `t ↦ J(x)` jumps whenever an input
crosses a region boundary. Fixing the pattern removes the jump by fiat.

**(b) Every pattern is driven by the same gradient.** The weight gradient
`Γ_ℓ := ∂L/∂W_ℓ` is *one matrix per layer*, computed once from the real loss on the real
data — so all the cross-input coupling that file 03 §2 isolated is already inside it. The
patterns differ only in how they compose it:

$$\dot s_k(\varepsilon) \;=\; -\sum_\ell u_k^\top A_\ell^\varepsilon\, \Gamma_\ell\, B_\ell^\varepsilon\, v_k \qquad\text{(exact)} \tag{1.1}$$

with `J_ε = A_ℓ^ε W_ℓ B_ℓ^ε` holding by construction. **No approximation, no alignment
assumption, no perturbative expansion.** This is the reformulation that makes the nonlinear
question answerable: *the dynamics of every pattern is a fixed-gates linear network driven
by a shared `Γ`.*

Verified: `J_ε = J(x)` to `1e-14` at realized patterns; `ṡ_k(ε)` matches a finite difference
of `J_ε`'s singular values to `1e-6` (`tests/test_theory_patterns.py`).

**The split.** Define the pattern's own **mode gain** and **drive**:

$$c_k^\varepsilon := \sum_\ell \big\|(A_\ell^\varepsilon)^\top u_k\big\|^2 \big\|B_\ell^\varepsilon v_k\big\|^2, \qquad g_k^\varepsilon := -\dot s_k(\varepsilon)/c_k^\varepsilon .$$

Then `ṡ_k = −c_k^ε g_k^ε` **identically** — a change of variables, not a hypothesis. By
Lemma 12 the bias rate is `b = d log|ω|/d log s` with `ω_k = ṡ_k/s_k`, so

$$b(\varepsilon) \;=\; \underbrace{\frac{d\log c^\varepsilon}{d\log s}}_{\text{geometry of the pattern}} \;-\; 1 \;+\; \underbrace{\frac{d\log|g^\varepsilon|}{d\log s}}_{\text{what the gradient asks}} \tag{1.2}$$

The first term is a property of the *network*; the second of the *task*. They behave
completely differently, which is why every earlier attempt to measure "the bias" as one
number was unstable.

---

## 2. The main result: the mechanism survives, at every pattern

Recall from file 01 that a CReLU layer splits exactly as `W_ℓ = [P_ℓ | Q_ℓ]`,
`S_ℓ := (P_ℓ − Q_ℓ)/2`, `Δ_ℓ := (P_ℓ + Q_ℓ)/2`, giving `W_ℓ D(z) = S_ℓ + Δ_ℓ diag(sign z)`.
So `S_ℓ` is the pattern-independent part of layer `ℓ` and `Δ_ℓ` is the part the gate pattern
multiplies. Define the **nonlinearity**

    δ := max_ℓ ‖Δ_ℓ‖ / ‖S_ℓ‖ ,

which is 0 exactly at a looks-linear configuration (`Δ = 0`, Cor. 2.1) and grows as the
network leaves the linear manifold. Sweep it from 0 (where every pattern gives the *same*
operator) upward, holding the spectrum seed fixed, and measure the
**gain exponent** of (1.2) at random Rademacher patterns. No gradient enters — this is pure
geometry.

| `L` | `2 − 2/L` | `δ=0` | `0.002` | `0.010` | `0.050` | `0.153` | `0.397` | `0.817` |
|---|---|---|---|---|---|---|---|---|
| 4 | 1.500 | 1.507 | 1.507 | 1.506 | 1.501 | 1.485 | 1.390 | 1.260 |
| 8 | 1.750 | 1.744 | 1.744 | 1.743 | 1.748 | 1.736 | 1.649 | 1.554 |
| 16 | 1.875 | 1.874 | 1.874 | 1.872 | 1.867 | 1.846 | 1.774 | 1.729 |
| 32 | 1.938 | 1.936 | 1.936 | 1.937 | 1.937 | 1.922 | 1.860 | 1.890 |

**The rich-get-richer exponent of the mode gain is `2 − 2/L` at an arbitrary gate pattern,
and it is robust to the nonlinearity** — within 1% out to `δ = 0.15`, and still within 15%
at `δ = 0.8`, where the network is emphatically not a perturbation of a linear one.

This is the positive answer, and it is stronger than expected in two ways.

*It holds for patterns no input realizes.* The measurement above uses Rademacher patterns.
The bias mechanism is a property of the weights, not of which regions the data selects.

*It is far better than the perturbative bound predicts.* At `L = 32, δ = 0.4` the standard
product bound gives `(1+δ)^L ≈ 10^5`, and file 03 §3(a) correctly ruled out perturbation
theory on that basis. Yet the exponent moves by 4%. The reason is the mechanism of file 05:
the pattern change acts on `c_k` largely as a **direction-independent** factor, and by
(1.3) of file 05 such a factor cancels in a log-log *slope*. Writing
`c_k^ε = s_k^{2−2/L} K_k^ε`, what the nonlinearity has to do to move the exponent is make
`K^ε` depend on the direction `k`, and it mostly does not.

---

## 3. `(H-mode)` is not needed for the bias

File 03 §4 proposed `(H-mode)`: realized patterns separate at least as well as typical ones,
measured at a factor 2.2. It was offered as the bridge from an ensemble bound to the data.

Measured on trained networks (48 runs, two tasks × two inits × depths 4–32 × three seeds,
8 patterns each), the bias rate `b` at realized patterns against Rademacher ones:

| task | init | `L` | realized `b` | random `b` |
|---|---|---|---|---|
| teacher–student | looks-linear | 32 | +0.853 | +0.819 |
| teacher–student | Xavier | 8 | +0.060 | +0.024 |
| MNIST | looks-linear | 16 | +1.302 | +1.242 |
| MNIST | looks-linear | 32 | +1.930 | +1.960 |
| MNIST | Xavier | 8 | +0.048 | +0.003 |

Pooled over all **408 snapshots**:

$$\operatorname{corr}\big(b_{\text{realized}},\, b_{\text{random}}\big) = +0.997, \qquad \operatorname{median}\big|b_{\text{realized}} - b_{\text{random}}\big| = 0.024$$

against a median `|b|` of `0.138`. **Realized and typical patterns bias identically** — the
disagreement is a sixth of the signal and the two track each other almost perfectly. So the
bias does not need `(H-mode)`: it is not a statement about which patterns the data picks.
That is *good* news for provability — a claim holding uniformly over the pattern ensemble
needs no transfer argument at all.

(Realized patterns run consistently a hair higher, so `(H-mode)`'s *direction* is right; its
magnitude is simply not where the bias lives.)

**And the gain exponent on trained networks confirms §2 outside the controlled sweep:**

| task | init | `L=4` | `8` | `16` | `32` | `δ` |
|---|---|---|---|---|---|---|
| MNIST | looks-linear | 1.509 | 1.753 | 1.888 | 1.922 | 0.01–0.04 |
| teacher–student | looks-linear | 1.304 | 1.725 | 1.873 | 1.937 | 0.01–0.05 |
| MNIST | Xavier | 1.035 | 1.420 | 1.603 | 1.772 | 1.04–1.19 |
| `2 − 2/L` | | 1.500 | 1.750 | 1.875 | 1.938 | |

At looks-linear the exponent matches the balanced prediction to within 0.03 for `L ≥ 8`
after 2000 training steps. At Xavier — where `δ ≈ 1.1`, far outside any perturbative regime
— it is reduced by 0.17 to 0.47 but remains large and positive. The mechanism degrades; it
does not vanish.

---

## 4. Where the outcome does vary: the drive

The second term of (1.2) is another matter. Under an arbitrary (random) `Γ`, the total rate
`b` falls monotonically with `δ` and **changes sign**:

| `δ` | 0.000 | 0.008 | 0.078 | 0.233 | 0.662 |
|---|---|---|---|---|---|
| `b`, realized | +2.45 | +1.96 | +0.65 | **−0.39** | **−0.96** |
| `b`, random | +1.93 | +1.71 | +0.79 | −0.19 | −0.64 |

and on trained networks the Xavier rows sit at `b ≈ 0.0–0.08` while the looks-linear rows
reach `+2.2`. Since §2 shows the *geometric* term is nearly unchanged across exactly this
range, all of that variation is the drive term.

So the honest summary is a separation of concerns:

> **The low-rank bias mechanism is intact and universal over gate patterns, with exponent
> `2 − 2/L`. Whether the spectrum actually separates is decided by the drive — and at Xavier
> initialization the drive very nearly cancels the mechanism.**

That reconciles the apparently contradictory measurements scattered through files 04–06: the
Xavier runs conserving separation (file 04 §5.6), the negative task exponents on synthetic
targets (§5.5), `p ≈ +11` on MNIST from an isometric start (file 06 §4). All of them are the
drive, acting on a mechanism that was never in doubt.

**Caveat, stated plainly.** `R²` of the rich-get-richer fit is low (0.03–0.30) under random
`Γ`, so `b` there is a real slope but a poor summary — the drive is not a power law in `s`.
The §2 numbers do not have this problem: no gradient enters them.

---

## 5. What can be proved, and what a proof would need

**Provable now, and easy.** At `δ = 0` (looks-linear) every pattern gives the *same*
operator `J_ε = S_L ⋯ S_1` — verified to `1e-14` — so the bias is exactly the deep-linear
one, uniformly over patterns. With a negation-closed batch this persists for all time
(Theorem 17), so on a linear teacher the whole pattern ensemble collapses to a single deep
linear network forever.

**The theorem the measurements point at.** Writing `c_k^ε = s_k(ε)^{2−2/L} K_k^ε`,

> **Conjecture.** For every pattern `ε`, `|d log K^ε / d log s| ≤ C(δ)` with `C(δ) → 0` as
> `δ → 0`, and `C` growing much more slowly than the perturbative `(1+δ)^L − 1`.
>
> **Resolved in [`08-gain-bound.md`](08-gain-bound.md), with a correction.** A bound in `δ`
> alone is *impossible*: the theorem needs an upper bound on `K`, and `K` is unbounded above
> (one layer scale to zero sends `K → ∞` at fixed `Π x_ℓ`). What is proved instead is a
> certificate, `|d log K/d log s| ≤ w/(2σ)` with `w` the width of a band containing `log K`
> and `σ` the spread of `log s` — whose floor is proved unconditionally
> (`c_k ≥ L s_k² g^{-2/L}`) and whose ceiling is one measured scalar. It certifies the
> exponent to within 0.02–0.065 of `2 − 2/L` near the linear manifold and 0.27–0.79 far from
> it, with 0 violations across the grid.

Measured `C`: `≤ 0.02` at `δ = 0.05`, `≤ 0.09` at `δ = 0.4`, `≤ 0.24` at `δ = 0.8`, roughly
flat in depth. A proof would have to exploit that the pattern change is
direction-*independent* to leading order, which is exactly the structure file 05's Theorem 19
formalizes in the aligned case — and which file 05 §"what does not transfer" shows fails
*exactly* there. So the missing step is a version of Theorem 19 with an error term rather
than an exact invariance, and that is the single most valuable thing left to prove.

**What a proof would *not* need**, given §3: any assumption about which patterns the data
realizes. That is a real simplification over the plan in file 03.

---

## 6. Status

| Claim | Status |
|---|---|
| `ṡ_k(ε) = −Σ_ℓ u_kᵀ A_ℓ^ε Γ_ℓ B_ℓ^ε v_k` | exact; finite-difference checked |
| `ṡ_k = −c_k^ε g_k^ε` | identity (definition of `g`) |
| `δ = 0` ⟹ all patterns share one operator | exact, tested |
| gain exponent `≈ 2 − 2/L` at arbitrary patterns | **measured**, 4 depths × 7 nonlinearities |
| realized ≈ random patterns | measured, 48 training runs |
| the drive decides the outcome | measured |
| `|d log K^ε/d log s| ≤ C(δ)` | **conjecture**, with measured `C` |

Widths ≤ 16, depths ≤ 32, two task families. `studies/patterns.py`.
