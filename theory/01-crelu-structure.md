# CReLU networks: exact structure

Everything in this file is proved unconditionally and checked numerically in
`tests/test_theory_crelu.py` and `tests/test_theory_modes.py`. Nothing here is an
approximation, and nothing here depends on the initialization, the data, or the training
algorithm. Later files build on these and do carry hypotheses; those are flagged where
they appear.

## 0. Setup and notation

A bias-free CReLU network of depth $L$ and width $d$ maps $x \in \mathbb{R}^{n_0}$ by

$$h_0 = x,\qquad z_\ell = W_\ell h_{\ell-1},\qquad h_\ell = \begin{bmatrix}\mathrm{relu}(z_\ell)\\ \mathrm{relu}(-z_\ell)\end{bmatrix} \in \mathbb{R}^{2d}$$

for $\ell = 1,\dots,L-1$, with output $f(x) = z_L$. So $W_1 \in \mathbb{R}^{d\times n_0}$ and
$W_\ell \in \mathbb{R}^{d \times 2d}$ for $\ell \ge 2$ (with $W_L$ having $n_L$ rows).

Write $\Lambda(z) := \operatorname{diag}(\mathbb{1}[z > 0])$, a $0/1$ diagonal projector, and

$$D(z) := \begin{bmatrix}\Lambda(z)\\ -(I - \Lambda(z))\end{bmatrix} \in \mathbb{R}^{2d\times d}.$$

Then $D(z)z = \big[\mathrm{relu}(z);\, \mathrm{relu}(-z)\big]$, so $h_\ell = D(z_\ell) z_\ell$ and the
input–output Jacobian is the gated product

$$J(x) = W_L D(z_{L-1}(x)) W_{L-1} \cdots D(z_1(x)) W_1. \tag{0.1}$$

**Convention at zero.** $\Lambda$ uses $\mathbb{1}[z>0]$, so $I - \Lambda$ is $\mathbb{1}[z \le 0]$.
At $z_i = 0$ both branches contribute $0$ to $D(z)z$, so (0.1) is unaffected; the choice only
fixes which of two equal values the sign takes.

---

## 1. The gate is an isometry

> **Lemma 1.** For every $z \in \mathbb{R}^d$, $\;D(z)^\top D(z) = I_d$.

*Proof.* Write $\Lambda = \Lambda(z)$. Since $\Lambda$ is diagonal with entries in $\{0,1\}$ we
have $\Lambda^\top \Lambda = \Lambda^2 = \Lambda$ and likewise $(I-\Lambda)^2 = I - \Lambda$.
Then

$$D^\top D = \Lambda^\top \Lambda + \big(-(I-\Lambda)\big)^\top\big(-(I-\Lambda)\big) = \Lambda + (I - \Lambda) = I. \qquad\blacksquare$$

Two consequences, both immediate.

> **Corollary 1.1 (norm preservation).** $\|D(z)z\|_2 = \|z\|_2$ for all $z$, i.e. the CReLU
> activation is norm preserving. Hence no gain correction is needed at initialization.

*Proof.* $\|D z\|^2 = z^\top D^\top D z = z^\top z$. $\blacksquare$

> **Corollary 1.2 (no rank loss from the nonlinearity).** $\operatorname{rank}(D(z)) = d$ for
> every $z$, and $\operatorname{rank}(W D(z)) = \operatorname{rank}(W|_{\operatorname{range} D(z)})$.
> The gate cannot annihilate any direction.

*Proof.* $D^\top D = I$ forces $D$ injective. $\blacksquare$

**Contrast with ReLU.** There the gate is $\Lambda(z)$ itself, an orthogonal *projection* with
$\Lambda^\top\Lambda = \Lambda \ne I$ whenever any unit is inactive; it annihilates the inactive
coordinates. Measured on a width-6 depth-8 network at Xavier initialization,
$\operatorname{rank} J(x)$ falls to $1$–$4$ for ReLU while remaining $6$ for CReLU.

**Why this matters for the fixed-gates analysis.** In Haas et al. (ICML 2026) the gates are
Bernoulli and can be singular; §5 notes the degeneracy that almost surely some gate becomes
the zero matrix at sufficient depth, forcing the product to vanish, and works around it with
conditioned $(r,p)$-gates (Def. 5.2). By Lemma 1 that degeneracy **cannot occur** for CReLU:
every gate is injective, so no gate can annihilate the product at any depth. All spectral
collapse in a CReLU network is attributable to the weights.

---

## 2. A CReLU layer is affine in the sign pattern

For $\ell \ge 2$ split $W_\ell = [\,P_\ell \mid Q_\ell\,]$ into its two $d\times d$ blocks and set

$$\boxed{\;S_\ell := \tfrac{1}{2}(P_\ell - Q_\ell),\qquad \Delta_\ell := \tfrac{1}{2}(P_\ell + Q_\ell).\;}$$

Define the **sign matrix** $E(z) := 2\Lambda(z) - I = \operatorname{diag}(\varepsilon)$ with
$\varepsilon_i = +1$ if $z_i > 0$ and $-1$ otherwise. Note $E(z)$ is diagonal, orthogonal, and
an involution.

> **Lemma 2 (exact layer decomposition).** For every $z$,
> $$W_\ell\, D(z) \;=\; S_\ell \;+\; \Delta_\ell\, E(z).$$

*Proof.* Using $D(z) = [\Lambda;\, -(I-\Lambda)]$ and the block split,

$$W_\ell D(z) = P_\ell \Lambda - Q_\ell (I - \Lambda) = -Q_\ell + (P_\ell + Q_\ell)\Lambda.$$

Substituting $\Lambda = \tfrac{1}{2}(I + E)$,

$$W_\ell D(z) = -Q_\ell + \tfrac{1}{2}(P_\ell+Q_\ell) + \tfrac{1}{2}(P_\ell+Q_\ell)E = \tfrac{1}{2}(P_\ell - Q_\ell) + \tfrac{1}{2}(P_\ell+Q_\ell)E,$$

which is $S_\ell + \Delta_\ell E$. $\blacksquare$

Verified symbolically with sympy (residual identically zero) and against the model's own
gates (max error $0$ in float64).

> **Corollary 2.1 (looks-linear $\iff$ $\Delta = 0$).** $W_\ell = [\,O \mid -O\,]$ if and only if
> $S_\ell = O$ and $\Delta_\ell = 0$, and in that case $W_\ell D(z) = O$ for **every** $z$.

*Proof.* $S = \tfrac12(O-(-O)) = O$ and $\Delta = \tfrac12(O + (-O)) = 0$; conversely
$P = S+\Delta$, $Q = \Delta - S$. The last claim is Lemma 2 with $\Delta = 0$. $\blacksquare$

So at looks-linear initialization the network is not *approximately* linear — it **is** linear,
and the fixed-gates picture holds with zero error. $\Delta_\ell$ is an exact, per-layer,
measurable coordinate for how far a layer has moved from linear.

---

## 3. The nonlinearity is bounded uniformly over inputs

> **Lemma 3.** For every $z$, the matrix $\Delta_\ell E(z)$ has the *same singular values* as
> $\Delta_\ell$. Consequently, for every $z$ and every $k$,
> $$\big|\sigma_k\!\big(W_\ell D(z)\big) - \sigma_k(S_\ell)\big| \;\le\; \|\Delta_\ell\|_2 .$$

*Proof.* $E(z)$ is orthogonal ($E^\top E = I$ since $E$ is diagonal with $\pm1$ entries), and
right-multiplication by an orthogonal matrix preserves singular values. For the bound, Weyl's
inequality for singular values gives
$|\sigma_k(A + B) - \sigma_k(A)| \le \|B\|_2$; apply it with $A = S_\ell$ and
$B = \Delta_\ell E(z)$, whose spectral norm is $\|\Delta_\ell\|_2$. $\blacksquare$

This is a statement *uniform in the input*: every input's layer map has its whole spectrum
within $\|\Delta_\ell\|_2$ of one shared matrix $S_\ell$. Measured on a perturbed depth-6
network: worst deviation $0.301$ against the bound $0.451$, over 200 inputs.

**Caveat, stated because it is easy to over-read.** Lemma 3 is per layer. It does **not**
imply that $J(x)$ is close to the linear product $S_L\cdots S_1$: the standard product bound
gives only

$$\Big\|\prod_\ell (S_\ell + \Delta_\ell E_\ell) - \prod_\ell S_\ell\Big\|_2 \;\le\; \prod_\ell\big(\|S_\ell\|_2 + \|\Delta_\ell\|_2\big) - \prod_\ell \|S_\ell\|_2,$$

which for $\|S\| \approx 1$ behaves as $(1+\delta)^L - 1 \approx L\delta$. Measured on MNIST,
$\delta = \|\Delta\|/\|S\|$ falls with depth ($0.084$ at $L=4$, $0.037$ at $32$, $0.0145$ at
$256$) but $L\delta$ *grows* ($0.34$, $1.17$, $3.6$), and the realized deviation reaches
$4.2\,\sigma_1$ with the top singular direction of $J(x)$ **orthogonal** to that of the linear
part. A deep CReLU network is not a perturbation of a linear one.

---

## 4. Modes

> **Definition 4.** A **mode** is a tuple $\varepsilon = (\varepsilon_1,\dots,\varepsilon_{L-1})$
> with $\varepsilon_\ell \in \{\pm 1\}^d$. Its operator is
> $$J_\varepsilon := M_L(\varepsilon_{L-1})\cdots M_2(\varepsilon_1)\, W_1, \qquad M_\ell(\varepsilon) := S_\ell + \Delta_\ell \operatorname{diag}(\varepsilon).$$

A mode need not be realized by any input.

> **Proposition 4.1.** For every input $x$, $\;J(x) = J_{\varepsilon(x)}$ where
> $\varepsilon(x)_\ell = \operatorname{sign} z_\ell(x)$ (with the convention of §0).

*Proof.* Immediate from (0.1) and Lemma 2. $\blacksquare$

Verified to $1.1\times10^{-16}$.

**Why modes rather than inputs.** Fixing an input does *not* fix the gates: as the weights
move under training, $\operatorname{sign} z_\ell(x)$ flips, so $t \mapsto J(x)$ is not a
fixed-gates product and its trajectory crosses region boundaries where the derivative jumps.
Fixing a mode does: $t \mapsto J_\varepsilon(t)$ is a smooth (indeed polynomial) function of
the weights, with no boundary crossings, because $\varepsilon$ is held fixed by fiat. This is
the technical reason to prefer modes, independently of any statistical argument.

---

## What is used downstream

| Result | Statement | Status |
|---|---|---|
| Lemma 1 | $D^\top D = I$ | unconditional |
| Cor. 1.2 | gates never lose rank | unconditional |
| Lemma 2 | $W D(z) = S + \Delta E(z)$ | unconditional, exact |
| Cor. 2.1 | looks-linear $\iff \Delta = 0$ | unconditional |
| Lemma 3 | uniform per-layer spectral bound | unconditional |
| Prop. 4.1 | $J(x) = J_{\varepsilon(x)}$ | unconditional |

Continued in [`02-mode-ensemble.md`](02-mode-ensemble.md) (what averaging over modes gives
exactly) and [`03-dynamics.md`](03-dynamics.md) (what training does, and what is still open).
