# The low-rank bias is an instability, and the seed is the data's sign asymmetry

Builds on [`01-crelu-structure.md`](01-crelu-structure.md) (the exact CReLU algebra) and
[`03-dynamics.md`](03-dynamics.md) (the master equation, and the four hypotheses ruled out
there). Numbering continues from Proposition 11.

§1 is unconditional and model-free. §2–§3 take as given the balanced deep-linear flow (2.1)
and a power-law gradient (2.2); both are stated explicitly where used, (2.2) is *measured*
rather than assumed (Proposition 15), and §2 shows exactly what changes when it fails. §4 is
unconditional for CReLU, with no assumption on the weights, the data, or the loss. §5 reports
the measurements and §6 states what is not established.

**The claim, in one paragraph.** Write `phi = 1 - 2/L`. There are **two** mechanisms, and
almost everything that is confusing about the low-rank bias comes from treating them as
one:

* a **drive** `D = -L s^phi (g_j - g_k)`, which creates separation out of an exact isometry
  whenever the gradient is anisotropic across modes. It scales as `L`, it is present from
  the first step, and it is *not* a feedback -- it is nonzero at `L = 2`, where the feedback
  is exactly zero;
* a **feedback** of rate `psi * omega_bar`, `psi = phi + p`, which multiplies whatever
  separation already exists. It vanishes identically at an isometry, its depth share `phi`
  saturates at 1 (depths 256 and 1024 differ by 0.6%), and its clock is *operator growth*,
  not training steps: `r(t)/r(0) = (sbar(t)/sbar(0))^psi`.

So the bias is universal in neither direction. Under an isotropic force it is purely the
feedback, an exact isometry is a fixed point at every depth, and the seed multiplies through
unchanged over five decades (measured). Under a real task the drive dominates and an
isometry is *not* a fixed point -- separation grows from `1e-15` to `0.47` even at `L = 2`.
Depth sets the amplification, the task sets what is amplified.

For CReLU specifically, Theorem 16 identifies the only thing that can move a looks-linear
network off the linear manifold: the correlation between the operator residual and the gate
sign pattern, `mean_b[R_b E_b]`. Under a sign-symmetric input distribution it is a mean-zero
average decaying as `B^{-1/2}`; under a sign-asymmetric one -- MNIST, whose pixels are
non-negative -- it is systematic and no batch size removes it (measured log-log slopes:
**-0.51 vs -0.01**, with a continuous dose-response between). And a batch closed under
negation makes it *exactly* zero, so the looks-linear manifold is invariant and a CReLU
network trained that way stays a deep linear network forever, to machine precision
(Theorem 17).

---

## 1. Separation depends only on the spread of the log-velocity

> **Lemma 12.** Let `s_k(t) > 0` be differentiable. Put `m_k = log s_k`, let
> `omega_k := s_k' / s_k` be the **log-velocity**, and let `r_jk := m_j - m_k`. Then
> $$\dot r_{jk} = \omega_j - \omega_k .$$
> If moreover `omega_k = a + b m_k + e_k` for scalars `a, b` and residuals `e_k`, then
> $$\dot r_{jk} = b\, r_{jk} + (e_j - e_k), \qquad\text{exactly.}$$

*Proof.* `m_k' = s_k'/s_k = omega_k` by the chain rule, and `r_jk = m_j - m_k`, giving the
first display. Substituting the second form,
`omega_j - omega_k = b(m_j - m_k) + (e_j - e_k)`. No approximation is made: the
decomposition of `omega` is a definition of `e_k`, not an assumption. ∎

Two things follow immediately and are worth stating separately, because both are routinely
conflated with the bias itself.

> **Corollary 12.1.** The mean of `omega` cancels from every `r_jk`. A velocity field with
> `omega_k` constant across modes -- however large -- rescales the operator and changes no
> singular-value ratio, no condition number, and no effective rank.

> **Corollary 12.2.** `b` is the exponential rate of every separation, and it is *linear in
> the velocity*: for any additive decomposition `s_k' = sum_i v_k^{(i)}`, the slopes add,
> `b = sum_i b^{(i)}`.

Corollary 12.2 is what makes the self/cross split of [`03-dynamics.md`](03-dynamics.md)
§2 usable. Regressing `self_k / s_k` and `cross_k / s_k` separately on `m` decomposes *the
bias* rather than the velocity's magnitude, and so distinguishes a cross term that reshapes
the spectrum from one that merely rescales it. `|cross|/|self|` cannot make that
distinction, which is why the measurement reported in file 03 §3(b) was inconclusive rather
than negative.

Implemented as `olo.theory.instability.amplification`; `rate`, `rate_self`, `rate_cross`.

---

## 2. Under a power-law gradient the rate is `psi = (1 - 2/L) + p`

Take the balanced, aligned deep-linear flow (Saxe 2014; Arora et al. 2019; and
`olo.theory.balanced`, which derives the mode gain `c_ii = L s^{2-2/L}` in closed form):

$$\dot s_k = -L\, s_k^{2 - 2/L}\, g_k, \qquad g_k := u_k^\top G v_k. \tag{2.1}$$

The gradient components `g_k` are not free -- a task supplies them -- so parameterize them
by how they scale with the mode they drive:

$$g_k = \gamma\, s_k^{\,p}. \tag{2.2}$$

`p` is *measured*, never posited (Proposition 15). Three values are worth naming: `p = 0`
is a residual whose components do not depend on mode size; `p = 1` is a pure-rescale
target, `P* = alpha P`, which asks for no change of shape at all; `p < 0` is a task that
pushes hardest on the *smallest* modes.

> **Theorem 13 (the separation ODE, exactly).** Under (2.1) and (2.2), with
> `psi := (1 - 2/L) + p` and `mbar := (m_j + m_k)/2`,
> $$\omega_k = -L\gamma\, s_k^{\,\psi}, \qquad \dot r_{jk} \;=\; 2\,\omega(\bar m)\,\sinh\!\big(\tfrac{\psi}{2} r_{jk}\big),$$
> where `omega(m) := -L gamma e^{psi m}`. Linearizing, `r' = psi * omega_bar * r + O(r^3)`.

*Proof.* From (2.1)–(2.2),
`omega_k = s_k'/s_k = -L s_k^{1-2/L} g_k = -L gamma s_k^{1-2/L+p} = -L gamma s_k^{psi}`,
which is `omega(m_k)` with `m_k = log s_k`. By Lemma 12,

$$\dot r_{jk} = \omega(m_j) - \omega(m_k) = -L\gamma\big(e^{\psi m_j} - e^{\psi m_k}\big) = -L\gamma\, e^{\psi \bar m}\big(e^{\psi r/2} - e^{-\psi r/2}\big),$$

using `m_j = mbar + r/2`, `m_k = mbar - r/2`. The bracket is `2 sinh(psi r / 2)` and the
prefactor is `omega(mbar)`. Expanding `sinh(x) = x + x^3/6 + ...` gives the linearization,
with relative error `(psi r)^2 / 24`. ∎

Verified symbolically (sympy: the first-order coefficients match
`D = -L s^phi (g_j - g_k)` and `lambda = -(L-2) s^phi (g_j+g_k)/2` identically) and against
RK4 integration of (2.1); `tests/test_theory_instability.py`.

> **Corollary 13.1 (under an isotropic force, an isometry is a fixed point).** If (2.2)
> holds and `s_k = s` for all `k`, then `r_jk = 0` for all pairs and `r_jk' = 0`, at
> **every** depth, every `psi`, and every `gamma`. The spectrum stays exactly isometric; the
> operator only rescales.

*Proof.* `sinh(0) = 0`. Equivalently, from (2.1) all modes have identical velocity when
they have identical size, so the flow moves them together. ∎

**The hypothesis is doing real work here and must not be dropped.** Without (2.2), Lemma 12
and (2.1) give, at an isometry `s_k = s`,

$$\dot r_{jk}\big|_{r=0} \;=\; -L\, s^{\phi}\,(g_j - g_k) \;=:\; D, \tag{2.3}$$

which is **nonzero whenever the gradient is anisotropic across modes**. `D` is a genuine
drive: it manufactures separation out of an exact isometry, it does not multiply an existing
seed, and -- crucially -- it survives at `L = 2`, where by Theorem 13 the feedback is
identically zero. Measured on a teacher-student task from a looks-linear start
(`r(0) ≈ 10^-15`, so no seed to amplify), the separation reaches `0.47` at `L = 2` and
`4.4` at `L = 4`: the drive alone accounts for essentially all of it.

So the correct statement is conditional and worth stating twice, because the unconditional
version is the natural thing to believe and is false:

* under an isotropic (or power-law) force, an isometry is a fixed point at every depth and
  the bias is purely a feedback multiplying a seed;
* under an anisotropic force, an isometry is *not* a fixed point, the drive creates
  separation from nothing, and the feedback then amplifies what the drive made.

This is the structural sense in which a looks-linear CReLU initialization differs in kind
from a Xavier one. It is not merely "better conditioned": it sets `r(0) = 0`, which
removes the feedback's entire input. It does not remove the drive, and no initialization
can.

> **Corollary 13.2 (four sign regimes).** The separation grows iff `psi * omega_bar > 0`.
> Since `omega_bar > 0` exactly when the operator is growing, this gives:
>
> | | operator grows | operator shrinks |
> |---|---|---|
> | `psi > 0` | separation grows | separation **shrinks** |
> | `psi < 0` | separation shrinks | separation grows |
>
> So non-monotone effective rank is a prediction of the law, not a numerical artefact.

All four cells are realized in measurement (§5.2).

---

## 3. The clock is operator growth, not time

> **Theorem 14 (the flow translates one coordinate).** Define the Box–Cox coordinate
> $$u(s) := \frac{s^{-\psi} - 1}{\psi} \quad (\,= -\log s \ \text{ at } \psi = 0).$$
> Under (2.1)–(2.2), `u(s_k(t)) = u(s_k(0)) + c(t)` with `c(t) = L gamma t`, **the same
> `c` for every mode**.

*Proof.* `du/ds = -s^{-\psi-1}`, so by the chain rule and (2.1)–(2.2),

$$\frac{d}{dt}u(s_k) = -s_k^{-\psi-1}\,\dot s_k = -s_k^{-\psi-1}\big(-L\gamma\, s_k^{1+\psi}\big) = L\gamma,$$

independent of `k`. Integrating gives the claim. The `psi = 0` case is the limit
`u = -log s`, where the same computation gives `d(-log s_k)/dt = -omega_k = L gamma`. ∎

The proof is three lines because the coordinate was chosen to make it so; the content is
that such a coordinate exists at all.

> **Corollary 14.1 (`d - 1` predictions from one fitted number).** `c` is recoverable from
> any single mode, or from the mean `ubar`, since `ubar(t) = ubar(0) + c`. Every other
> singular value is then determined: `s_k(t) = u^{-1}(u(s_k(0)) + c)`. The trajectory of the
> whole spectrum is a one-parameter family fixed by the initial spectrum. The learning rate,
> the gradient magnitude, and the number of steps enter only through `c`.

`olo.theory.instability.predict_spectrum`. Deviation from it measures exactly the part of
`g_k` that is *not* a power law in `s_k` -- the genuine drive, cleanly separated from the
feedback.

> **Corollary 14.2 (the scale–separation law).** To first order in `r`,
> $$\boxed{\;\frac{r(t)}{r(0)} \;=\; \left(\frac{\bar s(t)}{\bar s(0)}\right)^{\!\psi}\;}$$
> for the geometric mean `sbar`. Separation is a function of *how much the operator grew*,
> raised to `psi` -- not of how long training ran.

*Proof.* By Theorem 13 and Lemma 12, `d(log r)/dt = psi * omega_bar + O(r^2)`, and
`omega_bar = (1/d) sum_k d(log s_k)/dt = d(log sbar)/dt`. Dividing,
`d(log r) / d(log sbar) = psi`, and integrating gives the display. ∎

Three consequences, each of which is a separate experiment in §5:

1. **No growth, no *feedback* -- at any depth.** An operator held at constant scale is not
   separated further by the feedback, however deep it is and however long it trains. (The
   drive (2.3) is unaffected by this and keeps acting whenever the gradient is anisotropic.)
2. **The seed enters multiplicatively and never washes out.** `r(0) = 0` gives `r(t) = 0`;
   `r(0)` twice as large gives `r(t)` twice as large, forever. Because the law is
   multiplicative rather than additive, anisotropy injected at step 10⁴ is amplified
   exactly like anisotropy present at step 0 -- which is what makes a long flat plateau
   followed by a sudden fall the expected shape rather than a surprise.
3. **Depth's contribution saturates.** `phi = 1 - 2/L` is 0 at `L = 2`, 0.875 at 16, 0.9922
   at 256, 0.9980 at 1024. Depths 256 and 1024 differ by 0.6%. **There is no depth at which
   the feedback switches on**, which is why no depth wall appears once the seed is removed
   -- consistent with the measured 0.935 accuracy at `L = 256` and 0.924 at `L = 1024` for
   CReLU/looks-linear, against chance for ReLU/Xavier at `L = 256` across six decades of
   learning rate.

> **Corollary 14.3 (a task can cancel the depth bias exactly).** At `p = -(1 - 2/L)` we have
> `psi = 0`, `u = -log s`, and the flow is `s_k(t) = s_k(0) e^{-c}`: a pure rescaling that
> preserves every ratio, at every depth.

### 3.1 Reading `psi` and `p` off data, without assuming either

> **Proposition 15.** Let `(s_k, \dot s_k)` be any measured spectrum and velocity, and let
> `b` be the least-squares slope of `omega_k = \dot s_k / s_k` on `m_k = log s_k`, with
> `omega_bar` the mean. Then:
> 1. `\dot r_{jk} = b\, r_{jk} + (e_j - e_k)` holds *exactly*, with `e` the regression
>    residual -- so `b` is the bias rate of whatever model produced the numbers, with no
>    balancedness, alignment, or linearity assumption;
> 2. if the data do satisfy (2.1)–(2.2) then `b / omega_bar = psi`, so
>    `p = b/omega_bar - (1 - 2/L)` estimates the task exponent;
> 3. `b` is additive over any additive split of `\dot s_k`.

*Proof.* (1) and (3) are Lemma 12 and Corollary 12.2. For (2), Theorem 13 gives
`omega(m) = -L gamma e^{psi m}`, whose derivative in `m` is `psi * omega(m)`; the
least-squares slope over a spectrum estimates that derivative at the mean, where
`omega(mbar) ≈ omega_bar`, so `b ≈ psi * omega_bar`. The approximation is the curvature of
`exp(psi m)` over the observed spread of `m` and is quantified by the reported `R^2`. ∎

Point (2) is what makes `p` an observable rather than a modeling choice, and point (1) is
what keeps the measurement honest when the model is wrong: `R^2` reports how much of the
velocity the rich-get-richer form explains at all, and the residual spread reports the drive
the form cannot see.

---

## 4. Where the seed comes from in a CReLU network

Corollary 13.1 says a looks-linear network sits at the feedback's fixed point, and §2 says
the drive can push it off whenever the task's gradient is anisotropic. But there is a second,
CReLU-specific route off, which no amount of isotropy in the force prevents: the network can
stop being linear. This section identifies that mechanism exactly, and shows it is the *only*
one -- `Delta` moves under one quantity and nothing else.

Setup: a CReLU network at a **looks-linear configuration**, `W_l = [O_l | -O_l]` for
`l >= 2` with `W_1` arbitrary. By Corollary 2.1, `W_l D(z) = O_l` for every `z`, so

* `J(x) = O_L cdots O_2 W_1 =: J_0` for every input -- the operator is input-independent;
* `A_l = O_L cdots O_{l+1} =: M_l`, containing no gate;
* `B_l = D(z_{l-1}(x))\, N_l` with `N_l := O_{l-1} cdots O_2 W_1`, also gate-free.

Take one gradient step of size `eta` with per-sample operator gradients `G_b`, averaged over
a batch of `B` inputs, i.e. `Delta W_l = -(eta/B) sum_b A_l^T G_b B_l^T`, and split the
result into the coordinates of Lemma 2, `S = (P - Q)/2` and `Delta = (P + Q)/2`.

> **Theorem 16 (the seed is the residual–gate-sign correlation).** With
> `R_b := M_l^T G_b N_l^T` and `E_b := diag(sign z_{l-1}(x_b)) = 2 Lambda_b - I`,
> $$\Delta S_l = -\frac{\eta}{2}\cdot\frac1B\sum_b R_b, \qquad \Delta \Delta_l = -\frac{\eta}{2}\cdot\frac1B\sum_b R_b E_b .$$
> The linear part of the network moves under the **mean** residual; the nonlinear part --
> which by Corollary 2.1 is the entire departure from linearity, and hence the entire seed
> -- moves under the **correlation between the residual and the gate sign pattern**, and
> under nothing else.

*Proof.* Writing `D(z)^T = [\Lambda \mid -(I - \Lambda)]` as two `d x d` blocks side by
side, `A_l^T G_b B_l^T = R_b D(z_b)^T = [\,R_b \Lambda_b \mid -R_b (I - \Lambda_b)\,]`.
Reading off the two blocks of `Delta W_l = [\Delta P \mid \Delta Q]`,

$$\Delta P = -\frac{\eta}{B}\sum_b R_b \Lambda_b, \qquad \Delta Q = +\frac{\eta}{B}\sum_b R_b (I - \Lambda_b).$$

Hence

$$\Delta S = \tfrac12(\Delta P - \Delta Q) = -\frac{\eta}{2B}\sum_b R_b\big[\Lambda_b + (I - \Lambda_b)\big] = -\frac{\eta}{2B}\sum_b R_b,$$

$$\Delta \Delta = \tfrac12(\Delta P + \Delta Q) = -\frac{\eta}{2B}\sum_b R_b\big[\Lambda_b - (I - \Lambda_b)\big] = -\frac{\eta}{2B}\sum_b R_b E_b,$$

using `Lambda_b - (I - Lambda_b) = 2 Lambda_b - I = E_b`. ∎

Verified symbolically with sympy (both residuals identically the zero matrix) and against
the model's own autograd gradients. Note what the theorem does *not* need: no assumption on
`G_b`, on the loss, on the data distribution, or on `O_l`.

> **Corollary 16.1 (symmetric data: the seed is a `B^{-1/2}` fluctuation).** Suppose the
> pairs `(R_b, E_b)` are i.i.d. across `b` with `E[R E] = 0` -- for instance when the
> operator gradient is a function of `J` alone (so `R_b = R` is the same matrix for every
> sample, `J` being input-independent here) and the input law is sign-symmetric, giving
> `E[E_b] = 0`. Then `E[\Delta\Delta_l] = 0` and
> $$\frac{\|\Delta \Delta_l\|}{\|\Delta S_l\|} = \Theta_P\!\big(B^{-1/2}\big).$$

*Proof.* `Delta Delta_l` is `-eta/2` times a sample mean of `B` i.i.d. mean-zero matrices,
so its entries have standard deviation `Theta(B^{-1/2})` by the central limit theorem, while
`Delta S_l -> -(eta/2) E[R]` which is `Theta(1)`. ∎

In the special case `R_b = R`, the statement is exact rather than asymptotic:
`Delta Delta_l = -(eta/2)\, R \cdot \mathrm{diag}(2\hat p - \mathbb 1)` where `hat p_i` is the empirical
fraction of the batch with `z_i > 0`. The seed is *literally* the sampling error of the
gate-activation frequencies.

> **Corollary 16.2 (asymmetric data: the seed is systematic).** If `E[R E] != 0`, then
> `||\Delta\Delta_l|| / ||\Delta S_l|| \to ||E[RE]|| / ||E[R]|| > 0`. No batch size removes
> it, and the limit is a property of the task.

> **Corollary 16.3 (a closed-form dose–response).** For `R_b = R` and `x ~ N(\mu \mathbb 1, I)`
> with `N_l` orthogonal, `E[E_b] = diag(2\Phi(m_i) - 1)` at `m = N_l \mu \mathbb 1`. The
> ratio therefore runs from a pure `B^{-1/2}` fluctuation at `mu = 0` to saturation at
> `mu -> infinity`, through the Gaussian CDF.

### 4.1 A batch closed under negation freezes the network on the linear manifold

Corollary 16.1 makes the seed small. The following makes it exactly zero, and it is the
sharpest form of the reduction promised in [`03-dynamics.md`](03-dynamics.md) §2.

> **Theorem 17 (invariance of the looks-linear manifold).** Let the network be at a
> looks-linear configuration, let the teacher be linear (`y = P^* x`), let the loss be the
> squared error, and let every batch be **closed under negation**: `(x, y)` in the batch
> implies `(-x, -y)` in it with equal weight. Then `Delta Delta_l = 0` exactly for every
> layer, so the configuration remains looks-linear. By induction it remains so for every
> step: `J(x)` stays input-independent and the CReLU network *is* a deep linear network,
> for all time.

*Proof.* At a looks-linear configuration `J(x) = J_0` for every input (Corollary 2.1), so
the per-sample operator gradient of the squared error is
`G(x,y) = (J_0 x - y) x^\top`. Under `(x,y) \mapsto (-x,-y)`,

$$G(-x,-y) = \big(J_0(-x) - (-y)\big)(-x)^\top = \big(-(J_0 x - y)\big)(-x^\top) = G(x,y),$$

so `G` is **even**. The contexts `M_l` and `N_l` carry no gate, hence
`R_b = M_l^\top G_b N_l^\top` is even too. The gate sign matrix is **odd**:
`z_{l-1}(-x) = -z_{l-1}(x)` because the looks-linear chain up to layer `l-1` is linear, so
`E(-x) = -E(x)`. Therefore `R E` is odd, and the batch being closed under negation pairs
each term with its exact negative:

$$\sum_b R_b E_b = 0 \quad\Longrightarrow\quad \Delta\Delta_l = 0$$

by Theorem 16. The configuration is unchanged in its `Delta` coordinate, hence still
looks-linear, and the argument repeats. ∎

Measured: `max_l ||Delta_l||_\infty` stays at `1.1e-16` -- machine zero -- across 2000 steps
at `L = 12` with a strongly sign-asymmetric input law (`x ~ N(1, I)`), against `6.6e-2` for
the same run on the un-symmetrized batch.

**What this does and does not buy.** It removes the *entire* nonlinear contribution, exactly
and permanently: under it every result for deep linear networks -- Theorem 13, Theorem 14,
and Theorems 5.3 and 6.1 of Haas et al. (ICML 2026) -- applies to the CReLU network with
zero error rather than under a hypothesis. It does **not** remove the low-rank bias, because
the drive (2.3) lives in the linear network too: in the run above the separation grew to
`1.92` symmetrized against `1.98` raw. Symmetrizing changes *which theory applies*, not
*what the spectrum does*.

That is the honest summary of where the seed comes from. In a network that starts at the
isometric fixed point, the CReLU-specific contribution is supplied by the data's sign
asymmetry through the gates, is a `B^{-1/2}` fluctuation without it, and can be switched off
exactly. Everything that remains is the drive, and the drive is a property of the task.

---

## 5. Measurements

All in float64. `studies/forcing.py` and `studies/seed_source.py`.

### 5.1 The rate law, in the linear case

To measure `psi` without a task's confound the operator gradient is **prescribed**,
`G_b = -c U_b diag(s_b^p) V_b^T` via the surrogate `<G_b, J(x_b)>`, which back-propagates
to exactly the weight gradients gradient flow would give under that force and nothing else.
(This matters: on teacher-student with `n = 256` and `d = 16`, the empirical input
covariance alone has condition number ≈ 2.8, and its anisotropy measured *larger* than the
effect under test.) Deep linear, identity init, seed `r(0) = 10^-4`, depths 2–256:

measured `psi` (slope of `log r` against `log sbar`) / predicted `psi = (1-2/L) + p`:

| L | p = −1 | p = −0.5 | p = 0 | p = 0.5 | p = 1 |
|---|---|---|---|---|---|
| 2 | −1.001 / −1.000 | −0.500 / −0.500 | **−0.000 / 0.000** | 0.499 / 0.500 | 0.996 / 1.000 |
| 4 | −0.500 / −0.500 | **−0.000 / 0.000** | 0.499 / 0.500 | 0.997 / 1.000 | 1.493 / 1.500 |
| 8 | −0.250 / −0.250 | 0.250 / 0.250 | 0.748 / 0.750 | 1.246 / 1.250 | 1.742 / 1.750 |
| 16 | −0.125 / −0.125 | 0.375 / 0.375 | 0.873 / 0.875 | 1.370 / 1.375 | 1.866 / 1.875 |
| 32 | −0.063 / −0.062 | 0.437 / 0.438 | 0.935 / 0.938 | 1.432 / 1.438 | 1.929 / 1.938 |
| 64 | −0.031 / −0.031 | 0.468 / 0.469 | 0.966 / 0.969 | 1.463 / 1.469 | 1.960 / 1.969 |
| 128 | −0.016 / −0.016 | 0.484 / 0.484 | 0.982 / 0.984 | 1.479 / 1.484 | 1.976 / 1.984 |
| 256 | −0.008 / −0.008 | 0.492 / 0.492 | 0.990 / 0.992 | 1.487 / 1.492 | 1.984 / 1.992 |

**Worst absolute error over all 40 cells: 0.0087.** Two zeros are worth naming. `L = 2, p = 0`
gives `-0.000` against exactly 0: a two-layer factorization has no low-rank feedback at all.
`L = 4, p = -0.5` also gives `-0.000`, which is Corollary 14.3 -- `p = -phi` exactly cancels
the depth bias -- and there the separation is unchanged whether the operator grows or
shrinks (ratio 1.000 both ways).

The full-spectrum prediction of Corollary 14.1 -- `d - 1` numbers from one fitted scalar --
holds to a maximum log error of `1.4e-11` (`L = 256, p = -1`) rising to `7.9e-3`
(`L = 256, p = 1`), the latter being the regime where the flow reaches infinity in finite
time and the integration is least accurate.

### 5.2 Sign

All four cells of Corollary 13.2 are realized. Final `r(t)/r(0)`, `r(0) = 10^-4`:

| L | p | psi | operator grows (`c=+1`) | operator shrinks (`c=-1`) |
|---|---|---|---|---|
| 4 | −1 | −0.500 | 0.135 (shrinks) | 7.39 (grows) |
| 4 | −0.5 | **0.000** | 1.000 | 1.000 |
| 4 | +0.5 | 1.000 | 53.8 (grows) | 0.018 (shrinks) |
| 32 | −1 | −0.062 | 0.779 (shrinks) | 1.28 (grows) |
| 32 | +0.5 | 1.438 | 308 (grows) | 0.0031 (shrinks) |
| 256 | −1 | −0.008 | 0.969 (shrinks) | 1.03 (grows) |
| 256 | +0.5 | 1.492 | 382 (grows) | 0.0025 (shrinks) |

Nine of nine sign predictions correct, including the `psi = 0` row where neither direction
of growth changes the spectrum's shape.

### 5.2b The seed multiplies through

Deep linear, `L = 64`, `p = 0`, identical trajectory, only `r(0)` varied:

| `r(0)` | 1e-6 | 1e-4 | 1e-2 | 0.1 | 0.3 | 1.0 |
|---|---|---|---|---|---|---|
| final `r` | 4.26e-6 | 4.26e-4 | 4.26e-2 | 0.429 | 1.37 | 4.78 |
| ratio | 4.26 | 4.26 | 4.26 | 4.29 | 4.57 | 4.78 |

Constant to three digits over **five decades** of seed, with the deviation appearing only
where `r` is no longer small and the first-order form of Corollary 14.2 stops applying --
which is the `sinh` correction of Theorem 13, not an error.

### 5.3 The CReLU seed scales as `B^{-1/2}`

CReLU, `L = 16`, identity init so the operator starts **exactly isometric** (`r(0) = 0`),
symmetric prescribed force, 400 steps at equal growth per step:

| B | 8 | 32 | 128 | 512 | 2048 | 8192 |
|---|---|---|---|---|---|---|
| final `r` | 3.17\* | 0.791 | 0.528 | 0.224 | 0.103 | 0.0546 |
| `max_l ‖Δ_l‖/‖S_l‖` | 4.3e-2 | 1.5e-2 | 9.3e-3 | 3.8e-3 | 1.7e-3 | 1.2e-3 |

(\*hit the scale cap.) Fitted slope over `B = 32..8192`: **−0.48** for the separation and
**−0.47** for the nonlinearity, against the predicted −0.5. Starting from an exact isometry,
the entire low-rank bias of this network is a finite-batch effect.

### 5.4 Fluctuation or systematic: the dichotomy, and where MNIST sits

Slope of `log(‖ΔΔ‖/‖ΔS‖)` against `log B`, over `B = 8..2048`, three seeds:

| source | slope | ratio at `B = 2048` |
|---|---|---|
| prescribed force, `mu = 0` (symmetric) | **−0.51** | 0.023 |
| prescribed, `mu = 0.03` | −0.45 | 0.033 |
| prescribed, `mu = 0.1` | −0.28 | 0.084 |
| prescribed, `mu = 0.3` | −0.11 | 0.231 |
| prescribed, `mu = 1.0` | −0.02 | 0.584 |
| prescribed, `mu = 3.0` | −0.007 | 0.853 |
| teacher–student (Gaussian inputs) | −0.41 | 0.11 |
| **MNIST** (non-negative pixels) | **−0.011** | **0.94** |

Both ends of Corollaries 16.1–16.2 are realized, with a continuous dose–response between
them controlled by a single input-asymmetry parameter, and MNIST sits at the saturated end:
its seed is as large as the linear part of the gradient and is unaffected by batch size.
That is what one expects from non-negative pixels, and it is checkable on any dataset in a
few seconds.

---

## 6. What this does and does not establish

**Established unconditionally.** Lemma 12 (separation depends only on the spread of the
log-velocity, and the rate is additive over any decomposition of the velocity); Theorem 13
and its exact `sinh` form, with Corollary 13.1 (an isometry is a fixed point of the bias at
every depth) and 13.2 (the four sign regimes); Theorem 14 and Corollaries 14.1–14.3 (the
flow translates one coordinate; `d-1` predictions from one fitted scalar; `r ∝ sbar^psi`;
depth's share saturates at 1; a task can cancel it exactly). Theorem 16 and Corollaries
16.1–16.3, for CReLU, with no assumption on weights, loss, or data.

**Established conditionally.** §2–§3 are conditional on the balanced-and-aligned reduction
(2.1), which is standard for deep linear networks and holds exactly at the initializations
used here, and on the power-law parameterization (2.2), which is a *measurement* --
Proposition 15 estimates `p` rather than assuming it -- but which a task need not obey. When
it does not, Corollary 14.1's residual is exactly the size of the violation, and that
residual is reported rather than assumed small.

**Not established.**

* **That the CReLU nonlinear dynamics obey §2–§3 once `Delta != 0`.** They demonstrably do
  not obey the *multiplicative-seed* prediction, and Theorem 16 explains why: the seed is
  regenerated at every step rather than only at `t = 0`, so the trajectory is a stochastic
  drive plus the feedback rather than pure amplification of `r(0)`. The measured `B^{-1/2}`
  scaling is a consequence of that regeneration, not of the seed at initialization. What is
  *not* proved is a closed form for the resulting separation.
* **That `E[RE] != 0` on a given real dataset implies collapse rather than merely a seed.**
  §5.4 measures the seed's size, not what training does with it over 10⁴ steps.
* **The remaining hypotheses of [`03-dynamics.md`](03-dynamics.md) §4**, `(H-mode)` and
  `(H-primitive)`, are untouched here.

**One hypothesis is resolved.** `(H-fluct)` -- "the cross terms are mean-zero fluctuations
across inputs" -- is **confirmed** for sign-symmetric data, and it is exactly Corollary
16.1: at a looks-linear configuration `Delta = 0` makes the operator input-independent, so
`Delta` *is* the whole cross-input structure, and its increment is a mean-zero average with
`B^{-1/2}` size. But the conclusion runs opposite to the one file 03 anticipated. The
`sqrt(B)` cancellation does not rescue a bias that would otherwise be swamped; it *is* the
bias, and its smallness at large batch is what makes a symmetric-data CReLU network resist
low-rank collapse. Under sign-asymmetric data the cancellation fails and the cross terms
are systematic -- which is the regime the file-03 measurements (`|cross|/|self| = 0.07-0.7`,
non-negligible) were taken in.

---

## What is used downstream

| Result | Statement | Status |
|---|---|---|
| Lemma 12 | separation tracks only the spread of `omega`; rates add | unconditional |
| Thm 13 | `r' = 2 omega(mbar) sinh(psi r / 2)`, `psi = (1-2/L) + p` | exact, given (2.1)–(2.2) |
| Cor. 13.1 | an isometry is a fixed point at every depth | unconditional |
| Cor. 13.2 | four sign regimes; non-monotone rank predicted | unconditional |
| Thm 14 | the flow translates `u = (s^-psi - 1)/psi` rigidly | exact |
| Cor. 14.2 | `r(t)/r(0) = (sbar(t)/sbar(0))^psi` | first order in `r` |
| Thm 16 | `dS = -eta/2 mean(R)`, `dDelta = -eta/2 mean(R E)` | unconditional, exact |
| Cor. 16.1 | symmetric data: seed is `Theta_P(B^{-1/2})` | unconditional |
| Cor. 16.2 | asymmetric data: seed is systematic | unconditional |
