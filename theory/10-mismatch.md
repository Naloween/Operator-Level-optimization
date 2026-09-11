# The implicit bias, written down explicitly

Purely mathematical. No measurements appear in this file; every claim is proved.

The question: gradient descent *on the operator* would move `J` along `−G`, where
`G := \mathbb E_x[G_x]` and `G_x := ∂\ell_x/∂J(x)`. Gradient descent *on the factors* moves it
somewhere else. The implicit bias **is** that difference. This file computes it in closed
form, says exactly when it vanishes, bounds its norm, and separates it into three
independent mechanisms.

---

## 1. Setup and the standing assumption

Operator `J(x) = A_l(x)\,W_l\,B_l(x)` for every layer `l` (contexts as in
[`00-derivation.md`](00-derivation.md) §1). Loss `\mathcal L = \mathbb E_x[\ell_x(J(x))]`, so the
weight gradient is

$$\frac{∂\mathcal L}{∂W_l} \;=\; \mathbb E_x\big[A_l(x)^\top G_x\, B_l(x)^\top\big]. \tag{1.1}$$

> **(SA) Frozen gates.** The gates are held fixed while the weights move — exactly true at a
> fixed gate pattern, and true for `J(x)` away from region boundaries. Under (SA),
> `\dot J(x_0) = \sum_l A_l(x_0)\dot W_l B_l(x_0)`.

This is the only assumption in the file.

---

## 2. The mismatch in closed form

> **Definition 2.1 (transfer superoperator).** For inputs `x_0, x` and any matrix `H` of the
> operator's shape,
> $$\mathcal T_{x_0,x}(H) \;:=\; \sum_{l=1}^{L} A_l(x_0)\,A_l(x)^\top\, H\, B_l(x)^\top B_l(x_0).$$

> **Theorem 2.2 (the bias, exactly).** Under gradient flow `\dot W_l = -∂\mathcal L/∂W_l` and
> (SA), the mismatch `\mathcal R(x_0) := \dot J(x_0) - (-G)` is
> $$\boxed{\ \mathcal R(x_0) \;=\; \mathbb E_x\Big[\big(\mathrm{Id} - \mathcal T_{x_0,x}\big)(G_x)\Big].\ }$$

*Proof.* By (SA) and (1.1),
`\dot J(x_0) = -\sum_l A_l(x_0)\,\mathbb E_x[A_l(x)^\top G_xB_l(x)^\top]\,B_l(x_0)
= -\mathbb E_x[\mathcal T_{x_0,x}(G_x)]` by linearity of the expectation. Adding
`G = \mathbb E_x[G_x]` gives the display. ∎

**Everything about the implicit bias of a factored model is in the single superoperator
`\mathcal T`.** The loss enters only through `G_x`; the architecture enters only through
`\mathcal T`.

**Vectorised form.** With `\mathrm{vec}(AHB) = (B^\top\!\otimes A)\mathrm{vec}(H)`,

$$\mathcal T_{x_0,x} \;\cong\; \sum_l \underbrace{\big(B_l(x_0)^\top B_l(x)\big)}_{=:N_l} \otimes \underbrace{\big(A_l(x_0)A_l(x)^\top\big)}_{=:M_l}, \tag{2.1}$$

a sum of `L` Kronecker products. For `x = x_0` both factors are PSD.

---

## 3. Three mechanisms

Split `\mathcal T` twice. First separate the part proportional to the identity, then separate
the input dependence.

**(i) Scale.** Let `\mathrm{tr}` denote the superoperator trace; from (2.1),
`\mathrm{tr}\,\mathcal T = \sum_l \mathrm{tr}(M_l)\mathrm{tr}(N_l)`. Put

$$c_{x_0,x} \;:=\; \frac{\mathrm{tr}\,\mathcal T_{x_0,x}}{d^2}, \qquad \widetilde{\mathcal T}_{x_0,x} := \mathcal T_{x_0,x} - c_{x_0,x}\,\mathrm{Id},$$

so `\widetilde{\mathcal T}` is traceless.

**(ii) Input dependence.** Write `\mathcal T_{x_0,x} = \mathcal T_{x_0,x_0} + \mathcal D_{x_0,x}`,
the **drift** `\mathcal D` being zero whenever the contexts do not depend on the input.

> **Theorem 3.1 (decomposition).** With `\bar c := \mathbb E_x[c_{x_0,x}]`,
> $$\mathcal R(x_0) \;=\; \underbrace{(1-\bar c)\,G}_{\textbf{scale}} \;-\; \underbrace{\widetilde{\mathcal T}_{x_0,x_0}(G)}_{\textbf{anisotropy}} \;-\; \underbrace{\mathbb E_x\big[\widetilde{\mathcal D}_{x_0,x}(G_x)\big] - \mathrm{Cov}_x\big(c_{x_0,x},\,G_x\big)}_{\textbf{input dependence}} .$$
> where `\widetilde{\mathcal D} := \mathcal D - (c_{x_0,x}-c_{x_0,x_0})\mathrm{Id}` is the traceless drift.

*Proof.* Substitute `\mathcal T_{x_0,x} = c_{x_0,x}\mathrm{Id} + \widetilde{\mathcal T}_{x_0,x}`
into Theorem 2.2:
`\mathcal R = \mathbb E_x[(1-c_{x_0,x})G_x] - \mathbb E_x[\widetilde{\mathcal T}_{x_0,x}(G_x)]`.
The first term is `(1-\bar c)G - \mathrm{Cov}_x(c_{x_0,x},G_x)`. For the second, write
`\widetilde{\mathcal T}_{x_0,x} = \widetilde{\mathcal T}_{x_0,x_0} + \widetilde{\mathcal D}_{x_0,x}`
and use linearity, noting `\widetilde{\mathcal T}_{x_0,x_0}` does not depend on `x` so it comes
out of the expectation applied to `\mathbb E_x[G_x] = G`. ∎

The three terms are qualitatively different and it is worth being explicit about why:

| term | effect on the update | vanishes when |
|---|---|---|
| **scale** `(1-\bar c)G` | parallel to `G` — changes only the step length | `\bar c = 1`, i.e. by choosing `\eta` |
| **anisotropy** `\widetilde{\mathcal T}_{x_0,x_0}(G)` | rotates the update | every context Gram is a multiple of `I` (Thm 4.1) |
| **input dependence** | rotates, and differs per input | the contexts do not depend on `x` |

**Only the last two are implicit bias in any meaningful sense.** The first is a
reparameterisation of the learning rate and cannot change what the model converges to.

---

## 4. Exactly when the bias vanishes

> **Theorem 4.1 (vanishing of the anisotropy term).** Fix `x_0` and suppose the Grams
> `M_l := A_l(x_0)A_l(x_0)^\top` and `N_l := B_l(x_0)^\top B_l(x_0)` are simultaneously
> diagonalisable, `M_l = U\,\mathrm{diag}(\mu_l)\,U^\top` and `N_l = V\,\mathrm{diag}(\nu_l)\,V^\top`.
> Then `\mathcal T_{x_0,x_0} = c\,\mathrm{Id}` if and only if
> $$\sum_{l=1}^{L}\mu_{l,i}\,\nu_{l,j} \;=\; c \qquad\text{for every pair } (i,j), \tag{4.1}$$
> equivalently `\sum_l \mu_l\nu_l^\top = c\,\mathbb 1\mathbb 1^\top`.

*Proof.* In the `(U,V)` bases `\mathcal T` acts diagonally on the matrix entries:
`\mathcal T(H)_{ij} = \big(\sum_l \mu_{l,i}\nu_{l,j}\big)H_{ij}`. It is `c\,\mathrm{Id}` iff every
multiplier equals `c`. ∎

So the bias is governed by the `d\times d` **gain matrix** `C_{ij} = \sum_l \mu_{l,i}\nu_{l,j}`,
and vanishes iff `C` is constant. Two consequences:

> **Corollary 4.2 (isometric contexts).** If `M_l = m_lI` and `N_l = n_lI` for all `l`, then
> `C_{ij} = \sum_l m_ln_l` is constant and the anisotropy term vanishes identically. In
> particular a network whose every context is a multiple of an isometry has **no implicit
> bias beyond a step-size rescaling**, at any depth.

> **Corollary 4.3 (depth one).** For `L = 1`, `A_1 = B_1 = I`, so `\mathcal T = \mathrm{Id}`,
> `c = 1`, and `\mathcal R \equiv 0`. A linear model has no implicit bias — as it must not.

> **Corollary 4.4 (a balance condition, weaker than isotropy).** (4.1) asks that
> `\sum_l \mu_l\nu_l^\top` be the constant matrix. This does **not** require each `\mu_l`,
> `\nu_l` to be constant: per-layer anisotropies may cancel across layers. Isotropy is the
> obvious solution, not the only one. The bias is therefore a property of how the layers'
> context spectra *combine*, not of any single layer.

> **Corollary 4.5 (when the input-dependence term vanishes).** `\mathcal D_{x_0,x} = 0` for
> all `x` iff `A_l(\cdot)` and `B_l(\cdot)` are constant in the input. This holds exactly for
> deep linear and fixed-gate networks, and for a CReLU network at a looks-linear
> configuration, where `W_lD(z) = O_l` for every `z` makes the contexts input-free.

Putting 4.2 and 4.5 together: **a rectifier network whose layers are orthogonal and
looks-linear has `\mathcal R = (1-L)G` exactly** — pure rescaling, no bias at all, at any
depth. That is a complete, unconditional statement, and it is the extreme point of the
picture.

---

## 5. How large is it?

Normalise away the harmless term: after choosing `\eta` so that the step length matches, the
relevant quantity is the **relative bias**

$$\beta(x_0) \;:=\; \frac{\big\|\widetilde{\mathcal T}_{x_0,x_0}\big\|}{c_{x_0,x_0}},$$

zero exactly when `\mathcal T \propto \mathrm{Id}`. Decompose each Gram into its isotropic part
and the rest: `M_l = m_lI + \widetilde M_l` with `m_l = \mathrm{tr}(M_l)/d`, likewise
`N_l = n_lI + \widetilde N_l`. Define the **per-layer anisotropies**

$$a_l := \frac{\|\widetilde M_l\|}{m_l}, \qquad b_l := \frac{\|\widetilde N_l\|}{n_l}.$$

> **Theorem 5.1 (the bias is a convex average of per-layer anisotropies).**
> $$\beta(x_0) \;\le\; \sum_{l=1}^{L} w_l\,\big(a_l + b_l + a_lb_l\big), \qquad w_l := \frac{m_ln_l}{\sum_j m_jn_j},$$
> and the `w_l` are non-negative and sum to 1.

*Proof.* Expanding (2.1) at `x = x_0`,

$$\mathcal T = \sum_l (n_lI + \widetilde N_l)\otimes(m_lI + \widetilde M_l) = \Big(\sum_l m_ln_l\Big)\mathrm{Id} \;+\; \sum_l\big[n_l\,I\otimes\widetilde M_l + m_l\,\widetilde N_l\otimes I + \widetilde N_l\otimes\widetilde M_l\big],$$

and the first bracket is `c\,\mathrm{Id}` while the second is traceless (each `\widetilde M_l`,
`\widetilde N_l` has zero trace, and `\mathrm{tr}(X\otimes Y) = \mathrm{tr}X\,\mathrm{tr}Y`).
Hence `\widetilde{\mathcal T}` is the second bracket, and by the triangle inequality and
`\|X\otimes Y\| = \|X\|\|Y\|`,

$$\|\widetilde{\mathcal T}\| \le \sum_l\big[n_l\|\widetilde M_l\| + m_l\|\widetilde N_l\| + \|\widetilde N_l\|\|\widetilde M_l\|\big] = \sum_l m_ln_l\big(a_l + b_l + a_lb_l\big).$$

Divide by `c = \sum_l m_ln_l`. ∎

Three readings.

1. **The bias is an average, not a product.** It does not compound multiplicatively with
   depth. A single badly-conditioned layer contributes at most its weight `w_l`; conversely a
   network cannot be unbiased "on average" while any heavily-weighted layer is anisotropic.
2. **It is scale-free.** Each `a_l` is a *relative* deviation, so rescaling layers changes the
   weights `w_l` but not the `a_l`. The bias is about the *shape* of the context Grams.
3. **Depth enters only through the weights.** `L` appears nowhere except in the number of
   terms and in `c`. Depth makes the bias large only insofar as it makes individual contexts
   anisotropic — which is exactly what a product of many random matrices does.

> **Corollary 5.2 (the feedback loop, explicitly).** For a balanced deep linear network whose
> operator has singular values `s_i`, the contexts satisfy
> `\mu_{l,i} = s_i^{2(L-l)/L}` and `\nu_{l,i} = s_i^{2(l-1)/L}`, so with
> `\kappa := s_{\max}/s_{\min}`,
> $$a_l \;\asymp\; \kappa^{2(L-l)/L} - 1, \qquad b_l \;\asymp\; \kappa^{2(l-1)/L} - 1 .$$
> Every `a_l, b_l` is an increasing function of `\kappa`. So a more anisotropic operator gives
> a larger bias, and a larger bias drives the operator more anisotropic: **the loop is
> `\kappa \to \beta \to \kappa`, and Theorem 5.1 is the explicit form of its forward arrow.**
> At `\kappa = 1` every `a_l = b_l = 0` and the loop has no input — consistent with
> Corollary 4.2.

*Proof.* Substituting the balanced context spectra into the definitions; the Gram `M_l` has
eigenvalues `s_i^{2(L-l)/L}`, whose relative spread about the mean is controlled by the ratio
of extremes, `\kappa^{2(L-l)/L}`. ∎

---

## 6. What this gives, and what it does not

**Gives.** An exact closed form for the implicit bias (Thm 2.2); a separation into a harmless
scale term, a factorisation term and a nonlinearity term (Thm 3.1); a complete
characterisation of when the factorisation term vanishes (Thm 4.1, with 4.2–4.5 as the
cases); an explicit, scale-free upper bound showing the bias is a **convex average of
per-layer context anisotropies** (Thm 5.1); and the forward arrow of the
conditioning-to-bias feedback loop in closed form (Cor. 5.2).

Nothing above uses balancedness, alignment of contexts with the operator's singular basis, a
power-law ansatz, or any property of the gates. Corollary 5.2 alone is stated for a balanced
network, and only as an illustration.

**Does not give.**

* A *lower* bound on `\beta`. Theorem 5.1 is one-sided; the cancellation allowed by
  Corollary 4.4 means anisotropic layers need not produce a biased network, so a matching
  lower bound needs an extra non-degeneracy condition.
* Any control of the input-dependence term beyond "it vanishes when the contexts are
  input-free". Bounding `\mathcal D` requires a bound on the context drift
  `\|A_l(x) - A_l(x_0)\|`, which for rectifier gates is governed by how far apart two inputs'
  gate patterns are — and at depth those are close to independent.
* The backward arrow of Corollary 5.2: that a biased update *increases* `\kappa`. That is a
  statement about the dynamics, not about `\mathcal R` at one instant, and it does not follow
  from anything here.
* Any claim about what the network converges to. `\mathcal R` is an instantaneous object.
