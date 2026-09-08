# Dynamics: what training does, and where the argument is still open

Builds on [`01-crelu-structure.md`](01-crelu-structure.md) and
[`02-mode-ensemble.md`](02-mode-ensemble.md). §1–§2 are unconditional. §3 states the
obstruction precisely and reports measurements. §4 lists the hypotheses that would close the
argument, each with the number that would confirm or refute it. **The low-rank bias is not
proved here**; what is proved is an exact reduction of the problem to a single object, and a
record of which candidate hypotheses about that object have already been ruled out.

---

## 1. The master equation

Fix a mode $\varepsilon$ (§4 of file 01) and hold it fixed. Then
$J_\varepsilon = M_L(\varepsilon_{L-1})\cdots M_2(\varepsilon_1) W_1$ is a polynomial in the
weights, with contexts

$$A_\ell^\varepsilon := M_L \cdots M_{\ell+1},\qquad B_\ell^\varepsilon := M_{\ell-1}\cdots M_2 W_1,\qquad J_\varepsilon = A_\ell^\varepsilon W_\ell B_\ell^\varepsilon .$$

Under gradient flow $\dot W_\ell = -\nabla_{W_\ell}\mathcal{L}$ with
$\mathcal{L} = \mathbb{E}_x[\,\ell(f(x), y)\,]$, the weight gradient is
$\nabla_{W_\ell}\mathcal{L} = \mathbb{E}_x\big[A_\ell(x)^\top G(x) B_\ell(x)^\top\big]$, where
$A_\ell(x), B_\ell(x)$ are the contexts at the *realized* mode $\varepsilon(x)$ and
$G(x) := \partial \ell / \partial J(x)$.

> **Theorem 10 (master equation).** Let $s_k = s_k(\varepsilon)$ be a **simple** singular
> value of $J_\varepsilon$ with unit singular vectors $u_k, v_k$. Then, at any time where the
> realized gate patterns $\varepsilon(x)$ are locally constant,
> $$\dot s_k(\varepsilon) \;=\; -\sum_{\ell} \mathbb{E}_x\Big[\, u_k^\top A_\ell^\varepsilon A_\ell(x)^\top\, G(x)\, B_\ell(x)^\top B_\ell^\varepsilon\, v_k \Big].$$

*Proof.* Two steps.

*(i) Derivative of a simple singular value.* If $J(t)$ is differentiable and $s_k$ is simple,
then $s_k = u_k^\top J v_k$ with $u_k, v_k$ differentiable, and

$$\dot s_k = \dot u_k^\top J v_k + u_k^\top \dot J v_k + u_k^\top J \dot v_k = s_k\,\dot u_k^\top u_k + u_k^\top \dot J v_k + s_k\, v_k^\top \dot v_k,$$

using $Jv_k = s_k u_k$ and $u_k^\top J = s_k v_k^\top$. Since $u_k, v_k$ are unit vectors,
$\dot u_k^\top u_k = \tfrac12 \frac{d}{dt}\|u_k\|^2 = 0$ and likewise for $v_k$. Hence
$\dot s_k = u_k^\top \dot J v_k$. Simplicity is needed for $u_k, v_k$ to be differentiable.

*(ii) Derivative of the product.* $\varepsilon$ is fixed, so the gate matrices in
$J_\varepsilon$ are constants and the product rule gives
$\dot J_\varepsilon = \sum_\ell A_\ell^\varepsilon \dot W_\ell B_\ell^\varepsilon$.
Substituting $\dot W_\ell = -\mathbb{E}_x[A_\ell(x)^\top G(x) B_\ell(x)^\top]$ and combining
with (i) gives the statement. $\blacksquare$

*Verified numerically* against $\sum_\ell A_\ell \dot W_\ell B_\ell$ (agreement to machine
precision, $0.0$ in float64) and against finite differences of the true singular values along
the flow (agreement to the $O(\varepsilon)$ truncation error, $\sim10^{-7}$), for CReLU, ReLU
and deep linear networks. See `tests/test_theory_nonlinear.py`.

**The two regularity caveats, stated once.** (a) Simplicity of $s_k$: at a crossing of
singular values the derivative is only one-sided, in the usual way. (b) Local constancy of
$\varepsilon(x)$: the *realized* patterns appear in $\dot W_\ell$, and they jump when an input
crosses a region boundary. Fixing $\varepsilon$ removes the discontinuity from
$J_\varepsilon$ itself but not from the driving term. The rate of such crossings is measured
by `olo.diagnostics.linearization.gate_flip_rate`; nothing here bounds it.

---

## 2. The self/cross split

Specialize to a realized mode, $\varepsilon = \varepsilon(x_0)$, and split the expectation
over a batch of $N$ inputs into the term at $x_0$ and the rest:

$$\dot s_k = \underbrace{-\tfrac1N\sum_\ell u_k^\top A_\ell(x_0) A_\ell(x_0)^\top G(x_0) B_\ell(x_0)^\top B_\ell(x_0) v_k}_{\textstyle \mathrm{self}_k} \;+\; \underbrace{-\tfrac1N\sum_{x\ne x_0}\sum_\ell u_k^\top A_\ell(x_0)A_\ell(x)^\top G(x) B_\ell(x)^\top B_\ell(x_0) v_k}_{\textstyle \mathrm{cross}_k}.$$

> **Proposition 11 (the self term is exactly the fixed-gates term).** $\mathrm{self}_k$ has the
> form $-\tfrac1N \sum_\ell u_k^\top A_\ell A_\ell^\top\, G\, B_\ell^\top B_\ell\, v_k$ with all
> contexts evaluated at one and the same mode.

*Proof.* Immediate from the display. $\blacksquare$

This matters because that is precisely the structure the linear analyses handle. For a fixed
$x$, $J(x)$ **is** a fixed-gates linear network (Prop. 4.1), so Theorem 5.3 (depth scaling)
and Theorem 6.1 (separation $\Rightarrow$ alignment) of Haas et al. (ICML 2026) apply to it
with no modification — at fixed $x$ there is no nonlinearity left to handle. Under their
assumptions the self term reduces to the balanced form
$-\tfrac{1}{N} L\, s_k^{2-2/L}\,\langle G, u_k v_k^\top\rangle$ up to their constants, which is
the rich-get-richer law.

> **Corollary 11.1.** If $N = 1$ then $\mathrm{cross}_k = 0$ identically, and the dynamics are
> exactly those of a fixed-gates linear network.

*Proof.* The sum defining $\mathrm{cross}_k$ is empty. $\blacksquare$

**So the entire nonlinear extension reduces to one object**: the cross-input context overlaps
$A_\ell(x_0)A_\ell(x)^\top$ and $B_\ell(x)^\top B_\ell(x_0)$. That is the honest location of
the difficulty — a single measurable quantity rather than a diffuse appeal to nonlinearity.

---

## 3. What has been ruled out

Each of the following would close the argument, and each was checked and **fails**. Recorded
so they are not re-attempted.

**(a) $J(x)$ is a perturbation of the linear part $S_L\cdots S_1$.** False at depth. The
per-layer nonlinearity $\|\Delta\|/\|S\|$ does shrink with depth ($0.084$ at $L=4$, $0.037$ at
$32$, $0.0145$ at $256$ on MNIST), but the product amplifies it: the bound of §3 in file 01
goes as $(1+\delta)^L - 1$ and $L\delta$ *grows* ($0.34, 1.17, 3.6$). Measured, the deviation
reaches $4.2\,\sigma_1$ and the top singular direction of $J(x)$ is **orthogonal** to the
linear part's ($\sin\theta = 1.0000$).

**(b) $\mathrm{cross}$ is negligible (H-small).** False. Normalized per input —
$|\mathrm{cross}_k| / \big((N-1)|\mathrm{self}_k|\big)$, so that $1$ means "each other input
contributes as much as the reference one" — the measured value is $0.07$–$0.7$ across depths
$4$–$64$ and $50$–$4000$ steps of plain gradient descent. It does decrease with depth and with
training ($0.37 \to 0.089$ at $L = 64$), but it is not small.

**(c) $\mathrm{cross}$ is aligned with $\mathrm{self}$ (H-aligned).** False.
$\cos(\mathrm{self}, \mathrm{cross})$ over modes scatters across $[-0.93, +0.85]$ with mean
$-0.02$, and its correlation with spectral separation is $+0.06$ — i.e. none. An earlier
reading that suggested separation drives alignment came from a *diverging* run and does not
survive a stable step size.

**(d) The mode-ensemble gap grows exponentially in depth.** Not observed. On trained networks
at depths $4$–$128$, $\log(\lambda_1/\lambda_2)$ of $\mathbb{E}_\varepsilon[J^\top J]$ per layer
is small and irregular ($10^{-4}$ to $10^{-2}$) with no clean Lyapunov rate.

---

## 4. Open hypotheses, and how to falsify each

What remains is to find a property of the gate patterns that (i) implies the bias and (ii)
holds in practice. The candidates below are ordered by how much of the literature they would
need.

**(H-mode) Realized modes separate at least as well as typical ones.**
$$\operatorname{sep}(J_{\varepsilon(x)}) \;\ge\; \mathbb{E}_\varepsilon[\operatorname{sep}(J_\varepsilon)].$$
*Status:* supported. Measured $0.215$ against $0.097$ at $L = 64$, a factor $2.2$.
*Would give:* transfer of any ensemble-level separation lower bound to the data.
*Falsified by:* finding a task/architecture where realized separation falls below the
ensemble mean. Computed by `olo.theory.modes.compare`.

**(H-fluct) The cross terms are mean-zero fluctuations across inputs.** If
$\mathrm{cross}_k = \sum_{x\ne x_0} c_k(x)$ with the $c_k(x)$ having mean zero and weak
dependence, then $\mathrm{cross}_k = O_P(\sqrt{N})$ rather than $O(N)$, and the self term
dominates in probability once $N$ is large — giving the bias with high probability rather
than deterministically.
*Status:* untested here, and the sign of the evidence is mixed — the observed mean
$\cos(\mathrm{self},\mathrm{cross}) \approx -0.02$ is consistent with mean-zero behaviour, but
per-input magnitude $0.07$–$0.7$ is large enough that a $\sqrt{N}$ cancellation needs to be
demonstrated, not assumed.
*Falsified by:* measuring $\|\mathrm{cross}\|$ as $N$ grows and finding growth linear rather
than $\sqrt{\cdot}$. This is the cheapest next experiment: `decompose` already returns the
pieces; sweep $N$.

**(H-primitive) The layerwise transfer operators are primitive.** Needed for Theorem 9.
*Status:* holds in the instances tested, with exponent $n = 2$. It is a property of
$(S_\ell,\Delta_\ell)$ and can fail — e.g. $\Delta_\ell = 0$ with $S_\ell$ singular.
*Falsified by:* exhibiting a trained network where $\mathcal{T}^n$ keeps a nonzero PSD input
singular for all $n$ up to $d$.

---

## 5. Honest summary

Proved, unconditionally: the gate is an isometry and cannot lose rank; the layer is exactly
affine in the sign pattern; the per-layer spectrum is uniformly within $\|\Delta\|_2$ of a
shared linear one; a mode is exactly a sign vector and $J(x) = J_{\varepsilon(x)}$; the
mode-averaged second moment obeys an exact transfer recursion; that recursion is a positive
map, never strictly positive for $d\ge3$, but stabilizing under primitivity; and the master
equation for $\dot s_k$.

Proved, conditionally: nothing yet about the low-rank bias.

The value of the above is the reduction — at fixed $x$ or fixed $\varepsilon$ the problem is
*exactly* the fixed-gates one already analysed, and everything genuinely new sits in the
cross-input coupling, which is one measurable object with four candidate hypotheses, three of
which are already eliminated.
