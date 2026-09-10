# A certificate for the gain exponent

Builds on [`05-imbalance.md`](05-imbalance.md) and closes — partly — the conjecture stated in
[`07-patterns.md`](07-patterns.md) §5. Two elementary lemmas, one theorem, and an honest
account of the part that cannot be done the way I first wanted.

---

## 0. What is being bounded

Fix a gate pattern `ε`. Its operator is `J_ε = A_ℓ W_ℓ B_ℓ` for every layer `ℓ`, with
singular triples `(s_k, u_k, v_k)`. Define, as in files 05–07,

$$q_{\ell,k} := \big\|A_\ell^\top u_k\big\|^2\,\big\|B_\ell v_k\big\|^2, \qquad c_k := \sum_\ell q_{\ell,k}, \qquad K_k := c_k\, s_k^{2/L-2}. \tag{0.1}$$

`c_k` is the **mode gain**: exactly the `(k,k)` entry of the induced step
`ΔJ = -η Σ_ℓ A_ℓ A_ℓ^\top G B_ℓ^\top B_ℓ` under `G = u_k v_k^\top`. `K_k` is a definition,
so the identity

$$\frac{d\log c}{d\log s} \;=\; \Big(2 - \frac{2}{L}\Big) \;+\; \frac{d\log K}{d\log s} \tag{0.2}$$

holds with no hypothesis whatever. File 07 measured the left side at `≈ 2 - 2/L` for
arbitrary gate patterns and conjectured a bound on the right-hand correction. This file
proves one.

---

## 1. Lemma A: a band on `y` bounds the regression slope

> **Lemma A.** Let `x, y ∈ R^n` with `x` non-constant, and let
> `b = \langle \tilde x, \tilde y\rangle / \|\tilde x\|^2` be the least-squares slope of `y`
> on `x` (tildes denote centring). If every `y_i` lies in an interval of length `w`, then
> $$|b| \;\le\; \frac{w}{2\,\mathrm{sd}(x)}, \qquad \mathrm{sd}(x) = \|\tilde x\|/\sqrt n .$$

*Proof.* Since `\langle \tilde x, \mathbb 1\rangle = 0`, we have
`\langle \tilde x, \tilde y\rangle = \langle \tilde x, y\rangle = \langle \tilde x, y - c\mathbb 1\rangle`
for **every** constant `c`. Choose `c` to be the *midpoint* of the band, so that
`|y_i - c| \le w/2` for all `i`. Cauchy–Schwarz then gives

$$|b| \;=\; \frac{|\langle \tilde x,\, y - c\mathbb 1\rangle|}{\|\tilde x\|^2} \;\le\; \frac{\|y - c\mathbb 1\|}{\|\tilde x\|} \;\le\; \frac{(w/2)\sqrt n}{\sqrt n\,\mathrm{sd}(x)} \;=\; \frac{w}{2\,\mathrm{sd}(x)}. \qquad\blacksquare$$

The midpoint is what buys the factor 2; centring `y` on its own mean would give only
`w/\mathrm{sd}(x)`.

*Tight:* `x = (-1, 1)`, `y = (0, w)` attains it. *Checked* on `2\times10^5` random instances
(`tests/test_theory_gain_bound.py`), worst ratio `0.912`, so the bound is approached and not
vacuous.

**Why this is the right tool.** It converts the question "is the exponent near `2-2/L`?" into
"is `log K` confined to a narrow band?" — one number instead of a functional form. And it
puts the spread of the log-spectrum in the *denominator*, so the bound automatically
improves on the separated spectra where the question matters.

---

## 2. Lemma B: an assumption-free floor under the gain

> **Lemma B.** With `g := \prod_\ell \|W_\ell\|_2`, for every layer count `L`, every gate
> pattern, and every direction `k`,
> $$c_k \;\ge\; L\, s_k^{2}\, g^{-2/L}, \qquad\text{equivalently}\qquad K_k \;\ge\; L\,(s_k/g)^{2/L}.$$
> No balancedness, no alignment, no diagonality, no bound on the nonlinearity.

*Proof.* Fix `ℓ` and put `a := A_\ell^\top u_k`, `b := B_\ell v_k`. Since
`J_\varepsilon = A_\ell W_\ell B_\ell`,

$$s_k \;=\; u_k^\top J_\varepsilon v_k \;=\; u_k^\top A_\ell W_\ell B_\ell v_k \;=\; \langle a,\, W_\ell b\rangle \;\le\; \|a\|\,\|W_\ell\|_2\,\|b\|,$$

so `q_{\ell,k} = \|a\|^2\|b\|^2 \ge s_k^2/\|W_\ell\|_2^2`, which is non-negative. AM–GM on the
`L` non-negative terms of `c_k` gives

$$c_k \;=\; \sum_\ell q_{\ell,k} \;\ge\; L\Big(\prod_\ell q_{\ell,k}\Big)^{1/L} \;\ge\; L\Big(\prod_\ell \frac{s_k^2}{\|W_\ell\|_2^2}\Big)^{1/L} \;=\; L\,s_k^2\, g^{-2/L}. \qquad\blacksquare$$

*Tight:* an orthogonal chain has `\|W_\ell\| = 1`, `q_{\ell,k} = 1`, `c_k = L`, `g = 1` —
equality. Measured ratio `c_k / \text{floor}` exactly `1.0000` there, and `\ge 1` in every
other case tested (deep linear and CReLU, Xavier/Haar/looks-linear, `L = 4..16`).

This is the general form of Lemma 18 of file 05, which needed the aligned diagonal picture
to get `K \ge L`. Here the price of dropping that picture is the factor `(s_k/g)^{2/L}`,
which tends to 1 as `L \to \infty` for fixed `g/s_k`.

---

## 3. The theorem

> **Theorem 20 (gain-exponent certificate).** Let `Λ := \max_k K_k` over the directions with
> `s_k > 0`, and let `σ` be the standard deviation of `\log s_k` over those directions. Then
> $$\Big|\frac{d\log c}{d\log s} - \Big(2-\frac2L\Big)\Big| \;\le\; \frac{w}{2\sigma}, \qquad w := \log Λ - \min_k \log K_k .$$
> Replacing the minimum by Lemma B's floor gives the a-priori form
> $$w \;\le\; \log\frac{Λ}{L} \;+\; \frac{2}{L}\,\log\frac{g}{s_{\min}} ,$$
> in which the only measured quantity is `Λ`.

*Proof.* By (0.2) the left-hand side is `|d\log K/d\log s|`, a least-squares slope of
`\log K` on `\log s`. Every `\log K_k` lies in `[\min_k \log K_k,\ \log Λ]`, an interval of
length `w`; Lemma A applies. For the second form, Lemma B gives
`\log K_k \ge \log L + (2/L)\log(s_k/g) \ge \log L + (2/L)\log(s_{\min}/g)`. `∎`

**Verified end to end, 0 violations** over CReLU networks at three initializations, depths
4–32, six seeds × six random gate patterns each, in both forms.

| init | `L` | `|deviation|` | bound (measured band) | tightness | bound (Lemma B floor) |
|---|---|---|---|---|---|
| looks-linear | 4 | 0.0019 | **0.065** | 0.04 | 1.99 |
| looks-linear | 16 | 0.0032 | **0.035** | 0.08 | 2.74 |
| looks-linear | 32 | 0.0025 | **0.021** | 0.12 | 3.16 |
| Haar | 4 | 0.485 | 0.793 | 0.60 | 1.69 |
| Haar | 32 | 0.193 | 0.324 | 0.62 | 0.64 |
| Xavier | 4 | 0.255 | 0.452 | 0.56 | 1.54 |
| Xavier | 32 | 0.148 | 0.271 | 0.59 | 0.80 |

Two readings. Near the linear manifold the certificate is **sharp and small**: it *proves*
the gain exponent is within `0.02`–`0.065` of `2 - 2/L` at every gate pattern — which is the
regime file 07 §2 measured and could only assert. Far from it (`δ ≈ 1`) it still proves the
exponent is within `0.27`–`0.79`, non-vacuous at `2 - 2/L ≈ 1.9`, with the measured
deviation at `0.56`–`0.62` of the bound. Lemma B's floor costs roughly a factor of four,
because `g = \prod\|W_\ell\|` overshoots the true minimum.

---

## 4. What I could not prove, and why it is not for lack of trying

The conjecture in file 07 asked for `|d\log K/d\log s| \le C(δ)` — a bound in terms of the
nonlinearity alone. **That form cannot hold**, and the obstruction is structural rather than
technical.

Theorem 20 needs an upper bound on `K`. In the aligned diagonal picture
`K = \sum_\ell x_\ell` with `\prod_\ell x_\ell = 1`, so AM–GM floors it at `L` — but nothing
caps it: sending one layer's scale to zero sends `K \to \infty` at fixed `\prod x_\ell`.
**`K` is genuinely unbounded above**, so an upper bound is a hypothesis about the network,
not a consequence of `δ`.

I tried to discharge it anyway, by relative Weyl: `M_\ell = S_\ell + Δ_\ell E` gives
`σ_j(M_\ell)/σ_j(S_\ell) \in [1 \pm δ'_\ell]` with `δ'_\ell = \|Δ_\ell\|/σ_{\min}(S_\ell)`,
which would give `Λ \le L(1-δ')^{-2}`. Measured `δ'` on real CReLU networks: **192 to 1099**.
The relative nonlinearity against the *smallest* singular value of `S_\ell` is enormous even
when `‖Δ‖/‖S‖ ≈ 1`, so the bound is vacuous. That route is dead.

What Theorem 20 does instead is make the hypothesis a **single measured scalar** with a
proved floor beneath it — which is the form the rest of this project has been aiming at
anyway: uncertainty pushed into one checkable number rather than spread through the argument.

**Also not established.** Any bound on the *drive* term of file 07 (1.2), which is where the
outcome actually varies. Theorem 20 is about the mechanism only.

---

## 5. On verification

Lemmas A and B are four lines each and can be checked by eye; I would rather they were short
than machine-checked. What the tests add is scale and protection against transcription
error: Lemma A over `2\times10^5` random instances with the worst ratio recorded (`0.912`,
so the bound is tight enough to be falsifiable), Lemma B asserted pointwise on real networks
including its equality case, and Theorem 20 checked in both forms across the grid above.

I did not formalize these in a proof assistant. Lemma A is a Cauchy–Schwarz argument that
Lean's `Mathlib` would handle, but the effort is out of proportion to a four-line proof, and
it would not cover the part that actually needed care — which was not the algebra but
noticing that `K` has no upper bound, and that the useful statement is therefore a
certificate rather than an a-priori estimate.

| Result | Statement | Status |
|---|---|---|
| Lemma A | band of width `w` ⟹ `|slope| ≤ w/(2σ)` | proved; tight; 2e5 instances |
| Lemma B | `c_k ≥ L s_k^2 g^{-2/L}` | proved; no hypotheses; tight for orthogonal chains |
| Thm 20 | `|exponent − (2−2/L)| ≤ w/(2σ)` | proved; 0 violations |
| `Λ` bounded by `δ` | — | **impossible**: `K` is unbounded above |
