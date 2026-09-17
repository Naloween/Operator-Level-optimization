# Paper plan: the operator-level bias of coordinate gradient descent

Status: **plan under discussion, nothing implemented.** Every reference in §2 was fetched and
read on 2026-09-17; each row records what the source actually says, not what we hoped it says.

---

## 0. Verdict first

Two intended contributions were damaged by the literature check and must be demoted:

| Intended claim | What the check found | Action |
|---|---|---|
| `K_k ≥ L` (Lemma B, `theory/08`) is new | Rawal & DeWeese (arXiv:2605.01288, Jun 2026) apply AM–GM to layerwise gradient mass to get `T(W) ≳ γ^{2−2/r}`, with equality "precisely when linear-path gains are equal — the defining condition of the balanced ansatz" | **Demote to a cited lemma.** The AM–GM ⇒ `2−2/L` step is in the literature. Our version is per-direction, context-explicit and certificate-shaped, which is packaging, not insight. |
| *(correction, 2026-09-17)* earlier drafts of this plan said Lemma B proves `K_k ≥ L` "with no hypotheses" | It does **not**. Lemma B gives `K_k ≥ L (s_k/g)^{2/L}` with `g = ∏‖W_l‖₂`, and `(s_k/g)^{2/L} ≤ 1`, so it is strictly weaker. `K_k ≥ L` needs the aligned diagonal picture (`theory/05` Lemma 18). §4.5 finds `K_k ≥ L` holds on every usable direction measured, but that is evidence, not a proof. | Restate everywhere as the weaker inequality. |
| "Nobody can remove the depth preconditioner, so nobody can run the counterfactual" | Bernacchia, Lengyel & Hennequin (NeurIPS 2018) derive the exact natural gradient for deep linear networks; the follow-up literature states NGD "completely annihilates" pathological curvature in the linear case, and NGD is parameterisation-invariant *to first order* | **Reframe.** NGD already realises the counterfactual to first order. Our separation must be the *exact, finite-step, non-linearised* solve, and it has to be demonstrated where the linearisation vanishes. |

What survives is a narrower but still defensible paper. It is now an **optimisation paper with a
theory hook**, not an implicit-bias characterisation paper. §3 states exactly what we would claim;
§4 lists the go/no-go checks that must pass before we write a line of it.

---

## 1. Setup (fixed notation for everything below)

**Models.** Weights `W_l ∈ R^{n_l × n_{l−1}}`, `l = 1..L`, collected as `W`.

- **DLN** (deep linear network): `J(W) = W_L W_{L−1} ⋯ W_1`.
- **FGLN** (fixed-gate linear network): `J(W) = W_L D_{L−1} W_{L−1} ⋯ D_1 W_1`, with `D_l` a
  **fixed, untrained** diagonal matrix. `D_l ∈ {0,1}^{n_l×n_l}` is the *masked* subcase.
  This is the model class of the ICML paper (Haas et al., §"Fixed-Gates Linear Networks").

**Contexts.** `A_l := W_L D_{L−1} ⋯ D_l` and `B_l := D_{l−1} W_{l−1} ⋯ W_1`, with `A_L = I`,
`B_1 = I`, so that `J = A_l W_l B_l` for every `l`. (DLN is FGLN with all `D_l = I`.)

**Loss.** `L(J) = E_x[ℓ(Jx, y(x))]`, `G := ∂L/∂J ∈ R^{n_L × n_0}`. Everything below is stated for
a general differentiable `ℓ`; the squared loss `ℓ = ½‖Jx − y‖²` is singled out where it matters.

**Coordinate GD.** `∂L/∂W_l = A_lᵀ G B_lᵀ`, hence `W_l ← W_l − η A_lᵀ G B_lᵀ`.

**Transfer operator.** `T_W(H) := Σ_{l=1}^{L} A_l A_lᵀ H B_lᵀ B_l`, a linear map on `R^{n_L×n_0}`.

**Induced operator step.** `ΔJ_GD := J(W − η∇L) − J(W) = −η T_W(G) + O(η²)`.

**Operator-optimal step.** `ΔJ_* := −η G` (steepest descent in operator space). The **bias** is
`ΔJ_GD − ΔJ_*`; we report it as `cos(ΔJ_GD, −G)` and `sin = √(1−cos²)` as in `theory/12`.

**Mode gain.** For the SVD `J = Σ_k s_k u_k v_kᵀ`,
`c_k := ⟨u_k, T_W(u_k v_kᵀ) v_k⟩ = Σ_l ‖A_lᵀ u_k‖² ‖B_l v_k‖²`, and `K_k := c_k s_k^{2/L−2}`.

**Imbalance.** `I_l := W_{l+1}ᵀW_{l+1} − W_l W_lᵀ` (DLN). Unbalancedness magnitude
`max_l ‖I_l‖_F` — Cohen's Assumption 1.

---

## 2. State of the art, checked line by line

Every row was read at the URL given. "Hypotheses" records what the source actually requires.

### 2.1 The object we are studying is already in closed form

| Result | Source | Exact statement | Hypotheses |
|---|---|---|---|
| End-to-end dynamics | **Arora, Cohen & Hazan, ICML 2018** ([abs](https://arxiv.org/abs/1802.06509), [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.06509)) Thm 1 | `Ẇ_e = −ηλN W_e − η Σ_{j=1}^{N} [W_eW_eᵀ]^{(N−j)/N} ∇ℓ(W_e) [W_eᵀW_e]^{(j−1)/N}` | **gradient flow** (not GD); **balanced init** `W_{j+1}ᵀ(t₀)W_{j+1}(t₀) = W_j(t₀)W_jᵀ(t₀)`; deep **linear** |
| Preconditioner spectrum | same, Claim 1 | eigenvalues `Σ_j σ_r^{2(N−j)/N} σ_{r'}^{2(j−1)/N}` on `vec(u_r v_{r'}ᵀ)` | as above |
| Singular-value rate | **Cohen, lecture notes 2024** ([2408.13767](https://arxiv.org/html/2408.13767v1)) Thm 3 | `σ̇_r = n (σ_r²)^{1−1/n} ⟨−∇ℓ, u_r v_rᵀ⟩` | balanced, flow |

**Consequence.** `T_W` restricted to the diagonal is `c_k = L s_k^{2−2/L}` exactly, under
balancedness. **Our `2 − 2/L` is Arora's preconditioner.** Off-diagonal it is a divided
difference, `(s_i² − s_k²)/(s_i^{2/L} − s_k^{2/L})`. Any framing of the paper as "we characterise
the operator-level bias of GD on deep linear networks" is dead on arrival.

### 2.2 The acknowledged gaps

| Gap | Evidence it is open |
|---|---|
| **Unbalanced init** | Cohen's notes, open-problem list: "Results assume zero unbalancedness; extension to *small* unbalancedness is mentioned but not developed." |
| **Discrete GD** | same list: "Theorems focus on gradient flow; translating to practical gradient descent with finite step size requires separate analysis." Known: the invariant moves by `O(η²)` per step. |
| ~~**Gates**~~ | **Retracted 2026-09-17.** We believed no closed form existed for FGLN. One does: see §3.2. |

| Supporting result | Source | What it gives us |
|---|---|---|
| Imbalance is conserved | **Du, Hu & Lee, NeurIPS 2018** ([1806.00900](https://arxiv.org/abs/1806.00900)) | Under gradient **flow**, `‖W_{l+1}‖_F² − ‖W_l‖_F²` is invariant, for linear **and ReLU/leaky-ReLU homogeneous** nets. So `K_k` is essentially fixed at init. |
| Flow is Riemannian | **Bah, Rauhut, Terstiege & Westdickenberg**, IMAIAI 11(1) 2022 ([1910.05505](https://arxiv.org/abs/1910.05505)) | GD on the factors = Riemannian gradient flow on the rank-`r` manifold with a specific metric. **The bias *is* a metric.** Converges to a critical point; for a.e. init to a global min on rank-`k`, `k ≤ r`. |
| Orthogonal ⇒ shallow | **Bréchet, Papagiannouli, An & Montúfar** ([2011.13831](https://arxiv.org/pdf/2011.13831)) | Riemannian GD on orthogonal deep linear networks is *exactly equivalent to training one layer*. Depth has no effect in that gauge. |
| Survey of the flow | **Wendin & Altafini 2025** ([2511.10362](https://arxiv.org/html/2511.10362v1)) | Conservation law `W_iW_iᵀ − W_{i+1}ᵀW_{i+1} = C_i`, no-poor-local-minima (Kawaguchi), rank-based critical point classification, no general closed form for deep unbalanced: "the SVD does not extend well to matrix products". |

### 2.3 What the implicit bias is known to be

| Result | Source | Statement / hypotheses |
|---|---|---|
| Nuclear-norm conjecture | **Gunasekar et al., NeurIPS 2017** | Depth 2, small step, init near origin ⇒ min-nuclear-norm solution. **Conjecture** with partial proof. |
| Depth strengthens low rank | **Arora, Cohen, Hu & Luo, NeurIPS 2019** ([1905.13655](https://arxiv.org/pdf/1905.13655)) | Depth enhances the tendency to low rank in matrix completion/sensing. |
| No norm explains it | **Razin & Cohen, NeurIPS 2020** ([2005.06398](https://arxiv.org/abs/2005.06398)) | There exist problems where **all** norms and quasi-norms diverge to infinity along the trajectory. Rank, not norm, is the right lens. |
| Greedy low-rank | **Li, Luo & Lyu, ICLR 2021** ([2012.09839](https://arxiv.org/pdf/2012.09839)) | Depth-2, infinitesimal init: flow ≡ Greedy Low-Rank Learning. Refutes the nuclear-norm conjecture. |
| Saddle-to-saddle | **Jacot et al.** ([2106.15933](https://arxiv.org/pdf/2106.15933)) | Small init: flow visits a sequence of saddles of increasing rank. |
| Sequential modes | **Saxe, McClelland & Ganguli** ([1312.6120](https://arxiv.org/pdf/1312.6120)) | Exact solutions; singular values learned largest-first, with plateaus. |

**Consequence.** The *phenomenon* we would ablate (greedy, rank-incremental, largest-first
learning) is thoroughly documented. What is missing is a **controlled removal** of it.

### 2.4 The collisions — read these rows carefully

| Source | What it does | Why it does **not** subsume us |
|---|---|---|
| **Bernacchia, Lengyel & Hennequin, NeurIPS 2018** ([proceedings](https://proceedings.neurips.cc/paper/2018/hash/7f018eb7b301a66658931cb8a93fd6e8-Abstract.html)) — exact natural gradient in deep linear nets; "surprisingly simple, computationally tractable"; analytic convergence rate, loss decreases exponentially. Follow-up literature: NGD "completely annihilates" pathological curvature in the linear case; NGD is parameterisation-invariant **to first order**. | **This is the counterfactual, to first order.** NGD in a DLN cancels the depth preconditioner. | NGD solves the **linearised** system: with a singular Fisher and a pseudo-inverse it returns the min-norm `ΔW` with `M(ΔW) = Δ*`, `M(ΔW) := Σ_l A_l ΔW_l B_l`. Wherever `M ≡ 0` — the degeneracy of §3.3 — NGD returns exactly `0`. Our solve is on the exact degree-`L` polynomial, and is not pinned. **This distinction is the paper.** |
| **Rawal & DeWeese, arXiv:2605.01288 (Jun 2026)** — AM–GM on `T(W) = Σ_l‖G_l‖_F²` gives `≳ γ^{2−2/r}`; equality iff balanced ansatz; forward/backprop chains `A_ℓ`, `B_ℓ`. | Anticipates the AM–GM step of our Lemma B. | Their setting is a single-mode aligned teacher, small-signal bootstrap interval, saddle-escape exponents. Ours is per-direction and hypothesis-free. **Difference of packaging. Cite, do not headline.** |
| **Nguyen et al., arXiv:2202.02649** — implicit bias of GD on generalized gated linear networks; "captures a substantial portion of the inductive bias of ReLU networks"; frozen-gate ReLU nets are GLNs. | Covers frozen gates. | **Asymptotic** infinite-time limit, homogeneous-polynomial framing. Says nothing about finite-depth operator-step geometry or spectra. Our FGLN claims are about `T_W` at finite time. |
| **Evans & Tanner, arXiv:2601.16880 (Jan 2026)** — minimal-norm weight perturbation achieving a specified **output** change; single-layer formulas, multi-layer **Lipschitz bounds**; application to backdoor attacks. | Closest prior work on min-norm `ΔW`. | Target is an output change, not an operator step; multi-layer results are bounds, not exact solves; no dynamics, no certificate, not an optimiser. **Must cite prominently.** |
| **Dufort-Labbé et al., arXiv:2605.04230 (May 2026)** — Layerwise LQR: geometry-aware descent as an LQR, Riccati recursion, keeps cross-layer coupling, subsumes NGD/K-FAC/Shampoo. | Same "don't approximate before you solve" instinct, concurrent. | Model is the **local quadratic** `∇Lᵀδθ + ½δθᵀHδθ` over the **linearised** recurrence `δx_{i+1} = A_iδx_i + B_iδθ_i`. Order-2, hence pinned by §3.3. Optimiser design, not bias characterisation. |
| Target propagation / Proximal Backprop ([ICLR 2018](https://openreview.net/pdf?id=ByeqORgAW)) / MAC / ADMM | Layer-wise targets solved by least squares. | Targets are **activations**, set heuristically. Ours is the end-to-end operator, set by the optimality condition, with a min-norm certificate. |
| Schatten-`2/L` variational identity (Shang et al. line, e.g. [1606.01245](https://arxiv.org/pdf/1606.01245), [1803.00420](https://arxiv.org/pdf/1803.00420)) | `min Σ‖X_i‖_F² s.t. ∏X_i = X` = `L‖X‖_{S_{2/L}}^{2/L}`; proven for the 1/2 and 2/3 cases, general case folklore with a two-line proof. | We **use** this, we do not claim it. |

---

## 3. What we would claim

Each item: statement, hypotheses, proof status, and what would falsify it. **Nothing below is
written up until its go/no-go check in §4 passes.**

### 3.1 Result A — the transfer operator (bookkeeping, no hypotheses)

> `ΔJ_GD = −η T_W(G) + O(η²)`, where `T_W(H) = Σ_l A_lA_lᵀ H B_lᵀB_l` is self-adjoint and PSD
> for the Frobenius inner product. Hence `⟨G, T_W(G)⟩ ≥ 0`: the biased step never increases the
> loss to first order, whatever the gates and whatever the imbalance.

*Status:* proved, trivial, needed as scaffolding. *Hypotheses:* none (any DLN or FGLN, any loss).
*Role:* defines the object; makes `cos(ΔJ_GD, −G) ≥ 0` a theorem rather than an observation.

### 3.2 Result B — **RETRACTED.** FGLN flow is deep-linear flow in disguise

An earlier draft of this plan claimed gates break the matrix conservation law down to its
diagonal. **That claim is false**, and Proposition 4.5 of the ICML paper (Haas et al.) is correct.
The check in full, because the consequence matters more than the retraction.

**Gate absorption.** For masks (`D_l^2 = D_l`, `D_0 = D_L = I`) put `M_l := D_l W_l D_{l-1}`. Then
`M_L...M_1 = W_L D_{L-1}^2 W_{L-1} ... D_1^2 W_1 = J` — idempotency is exactly what licenses the
`M` form; for general diagonal `D_l >= 0` use `M_l := D_l^{1/2} W_l D_{l-1}^{1/2}`. The contexts
are the `M`-subproducts themselves:
```
A_l = W_L D_{L-1} ... W_{l+1} D_l = M_{L:l+1},      B_l = D_{l-1} W_{l-1} ... D_1 W_1 = M_{l-1:1}
```
so `Wdot_l = -M_{L:l+1}^T G M_{l-1:1}^T`, and since `M_{L:l+1} D_l = M_{L:l+1}` and
`D_{l-1} M_{l-1:1} = M_{l-1:1}`,
```
Mdot_l = D_l Wdot_l D_{l-1} = - M_{L:l+1}^T G M_{l-1:1}^T .            (B.1)
```
**The gates vanish from the equation of motion.**

**Conservation (Prop. 4.5, reproved).** With `Y_l := M_{l:1} G^T M_{L:l+1}`,
`Mdot_{l+1}^T M_{l+1} = -M_{l:1} G^T M_{L:l+2} M_{l+1} = -Y_l` and
`Mdot_l M_l^T = -M_{L:l+1}^T G (M_l M_{l-1:1})^T = -Y_l^T`, so both
`d/dt (M_{l+1}^T M_{l+1})` and `d/dt (M_l M_l^T)` equal `-(Y_l + Y_l^T)`, and
`Delta_l := M_{l+1}^T M_{l+1} - M_l M_l^T` is conserved. QED

**Diagnosis of the error.** The retracted claim compared the *raw* pair `W_{l+1}^T W_{l+1}` against
`W_l W_l^T`. Tracking the leftover gate exactly, with `Z := Wdot_{l+1}^T W_{l+1}` and
`V := Wdot_l W_l^T`, one finds `Z D_l = -Y_l` and `V D_l = -Y_l^T`: `Z` and `V^T` agree **only
after right-multiplication by `D_l`**, i.e. on the gate's active coordinates. That leftover `D_l`
is not evidence of broken conservation — it is the gate that belongs inside `M`. The supporting
gauge argument was invalid too: the gauge group is the *commutant* of `D_l` (for a mask with `a`
active coordinates, `GL(a) x GL(b)`, not the diagonal torus), and Noether bounds conserved
quantities from below, so symmetry counting can never establish non-conservation.

**Lemma B' (the consequence, and it is the useful part).**
> (B.1) is *exactly* the deep-linear gradient flow in the variables `M_l`, and the flow preserves
> the coordinate subspace `{X : D_l X D_{l-1} = X}`. Hence **FGLN gradient flow is deep-linear
> gradient flow restricted to a fixed coordinate subspace**, and Arora-Cohen-Hazan Thm 1 transfers
> verbatim under `M`-balancedness:
> `Jdot = -sum_j (J J^T)^{(L-j)/L} G (J^T J)^{(j-1)/L}`, fractional powers taken on the range.

**What this costs the plan.** FGLN was carrying the argument that we must prove an inequality
because no identity exists (old §3.6, old Fig. 4). That argument is gone. What remains genuinely
distinctive about FGLN is narrower and must be stated as such:
1. `M`-balancedness is a real restriction at initialisation — a Xavier-initialised FGLN does not
   satisfy it — so the *unbalanced* FGLN question is open for the same reason the unbalanced DLN
   question is (Cohen's open-problem list), not for a gate-specific reason.
2. The open object is the **multi-pattern** case: one `W`, many gate patterns, hence many
   `M`-parameterisations sharing weights, with no single `Delta_l` conserved across patterns.
   This is Remark 7.2 of the ICML paper, flagged there as lacking interpretability. It is the
   honest place for the difficulty to live, and `theory/07` already has machinery for it.

### 3.3 Result C — the degeneracy theorem (headline)

> **Definition.** A step rule is *derivative-based of order `p`* if `ΔW = Ψ({D^mF(W)}_{m≤p})`,
> and *null-consistent* if `Ψ(0,…,0) = 0`.
>
> **Theorem.** Let `Φ(ΔW) := ∏_l (W_l + ΔW_l)` and `F(ΔW) := ‖Φ(ΔW) − P_*‖_F²` with `P_* ≠ 0`.
> At `W = 0`, `Φ` is homogeneous of degree `L` in `ΔW`, so `D^mΦ(0) = 0` for all `m < L` and
> `D^mF(0) = 0` for all `1 ≤ m < L`. Hence **every null-consistent derivative-based rule of order
> `p < L` returns `ΔW = 0`.**
>
> **Corollary.** GD, momentum, Adam (`0/(√0+ε) = 0`), Muon, Shampoo, SOAP, K-FAC, Newton,
> Gauss–Newton, Levenberg–Marquardt, **natural gradient with pseudo-inverse Fisher**, and
> Layerwise LQR are all exactly pinned at the all-zero point of a depth-`L` network, for every
> `L ≥ 3`; GD/Adam/NGD/GN also for `L = 2`.
>
> **Proposition (the practically relevant version).** For `W_l = εΩ_l` with `Ω` fixed,
> `‖∂F/∂W_l‖ = Θ(ε^{2L−1})`, so the step is exponentially small in depth near the origin.

*Status:* the theorem is a two-line homogeneity argument, rigorous, and the method class is
defined so that it is a theorem rather than a list of examples. **Novelty is thin** — the `ε`
version overlaps the small-init / saddle-to-saddle escape-time literature (Jacot et al.), and
"GD is stuck at zero init" is folklore. **Its value is as the setting where our method separates,
not as a result in itself.** The paper must say this in those words.

### 3.4 Result D — min-norm realisation and its certificate

> **Problem (P).** `min Σ_l ‖ΔW_l‖_F²` subject to `J(W + ΔW) = P_*`.
>
> **Lemma (KKT).** At any regular optimum, with `A'_l, B'_l` the contexts at `W + ΔW`, there is a
> **single** multiplier `Λ ∈ R^{n_L×n_0}` with
> ```
> ΔW_l = A'^ᵀ_l Λ B'^ᵀ_l    for every l.
> ```
> *Proof:* `∂/∂ΔW_l ⟨Λ, A'_l(W_l+ΔW_l)B'_l⟩ = A'^ᵀ_lΛB'^ᵀ_l`; stationarity of the Lagrangian. ∎
>
> **Corollary (certificate).** `ρ := min_Λ Σ_l‖ΔW_l − A'^ᵀ_lΛB'^ᵀ_l‖² / Σ_l‖ΔW_l‖²` is computable
> in closed form and vanishes iff the candidate is stationary for (P). Reported per step.
>
> **Corollary (reduced solve).** Parameterising `ΔW_l = A'^ᵀ_lΛB'^ᵀ_l` reduces (P) from `L·d²` to
> `d²` unknowns — a fixed-point iteration in `Λ` alone.
>
> **Proposition (zero base point).** At `W = 0`, (P) has the closed-form solution
> `W_l = Σ^{(1)/L}`-chain from `P_* = UΣVᵀ` (balanced SVD chain), with optimal value
> `L‖P_*‖_{S_{2/L}}^{2/L}` — the Schatten identity of §2.4, which we cite.

*Status:* the KKT lemma is proved above and is elementary. The certificate is new *as an
instrument* — it is what converts the user's existing dichotomic solver from a heuristic into
something with a discharge-able proof obligation. The zero-base proposition needs a consistency
check (§4.2).

*Relation to NGD:* the **linearisation** of (P) is exactly pseudo-inverse-Fisher NGD. (P) itself
is not. This sentence must appear in the paper; hiding it would be fatal at review.

### 3.5 Result E — the honest reduction (state it ourselves, prominently)

> **Proposition.** For a DLN with squared loss, if each step exactly realises `P_* = J − ηG`,
> the operator trajectory `J(t)` is identical to plain gradient descent on the convex problem
> `min_P L(P)`. Depth has no effect on the trajectory.

*Consequence, stated in the paper:* **in the deep linear case our method is a control, not a
speedup.** Any claim of optimisation gain there is vacuous. This is also why §2.4's
"deep orthogonal linear networks are shallow" is a precursor in spirit, and we say so.

### 3.6 Result F — what the counterfactual is *for*

The deliverable is the measurement no one has run:

> Same architecture, same data, same schedule, same step budget; arm 1 takes the coordinate-GD
> step, arm 2 takes the certified min-norm realisation of `−ηG`. Report the difference in
> **(a)** singular-mode learning order, **(b)** effective rank trajectory, **(c)** endpoint, and
> **(d)** test error, across conditioning regimes.

**The claim we would like to make:** Saxe's sequential largest-first mode learning, the plateaus,
and the greedy rank-incremental trajectory are *entirely* artifacts of the factorisation, and
vanish under arm 2. **This is a hypothesis, not a result.** §4.3 is its check and it is the
single highest-risk item in the plan.

---

## 4. Go / no-go checks — run these BEFORE writing anything

Ordered by (risk × cheapness). Each has an explicit kill condition.

### 4.1 ~~Result B~~ — **settled by proof, no experiment needed**
Closed analytically in §3.2: the claim was false, Prop. 4.5 stands, and Lemma B' replaces it.
Nothing to run. The surviving FGLN question (multi-pattern, §3.2 item 2) is theory, not a check.

### 4.2 Result D: the certificate — **RUN 2026-09-17: PASSES, with a large finding about ALS**

`studies/certificate.py`, `runs/theory/certificate.json`. d=6, L∈{2,3,4}, 3 seeds. For one
prescribed step `P_* = J − ηG` we solve (P) by augmented Lagrangian and compare four candidate
realisations by `‖ΔW‖` (what (P) minimises), by the **true** residual `‖∏(W+ΔW) − P_*‖`, and by
the KKT residual `ρ`.

| init | L | ALS `‖ΔW‖` / min-norm | ρ(ALS) | NGD `‖ΔW‖` / min-norm | NGD true residual | ρ(NGD) | GD true residual |
|---|---|---|---|---|---|---|---|
| xavier | 2 | 1.55 – 3.22 | 0.56 – 0.85 | 0.87 – 0.97 | 2e-3 – 7e-3 | 0.01 – 0.25 | 0.13 – 0.16 |
| xavier | 3 | 5.7 – **447** | 0.98 – 1.00 | 1.04 – 4.63 | 8e-3 – 3e-2 | 0.23 – 0.98 | 0.38 – 0.44 |
| xavier | 4 | 47 – **175** | 1.00 | 1.22 – 8.72 | 1.6e-2 – **13** | 0.64 – 0.73 | 0.31 – 0.61 |
| tiny (‖J‖=1e-3) | 2 | 5.5 – 6.1 | 1.00 | 3.9 – 4.3 | **13 – 15** | 0.23 – 0.31 | 1.00 |
| tiny | 3 | 10.4 – 11.9 | 1.00 | 6.0 – 6.9 | **77 – 150** | 0.36 – 0.57 | 1.00 |
| tiny | 4 | 14.1 – 16.5 | 1.00 | 7.1 – 8.3 | **540 – 910** | 0.42 – 0.74 | 1.00 |

Three findings, the first of which is the most useful thing produced so far.

1. **ALS reaches the target but its realisation is wildly non-minimal, and it degrades with
   depth**: `1.5–3×` the minimal weight movement at `L=2`, `6–447×` at `L=3`, `47–175×` at `L=4`.
   `ρ(ALS) ≈ 1` almost everywhere — ALS's step is close to *orthogonal* to the optimal form, so it
   is not a noisy approximation of the min-norm realisation, it is a different object. This is a
   concrete, quantified defect of the submitted method, and it is the natural explanation for why
   it needed a warm start and behaved badly at depth.
2. **NGD is not a substitute outside the lazy regime.** At Xavier `L=2` it is within 3–13% of
   min-norm, but its *true* residual is never zero (it solves the linearised constraint), and at
   small init the linearisation fails catastrophically: true residual `13` to `910` against a
   target of unit scale, and `‖ΔW‖` 4–8× minimal. So the separation from Bernacchia et al. is not
   confined to the measure-zero degenerate point — it covers the whole small-initialisation
   regime, which is where implicit bias is studied.
3. **The GD step does not target `P_*` at all** (true residual `1.00` at small init, i.e. it moves
   essentially nowhere: `‖ΔW‖` is `0.01×` minimal at `L=2` and `0.001×` at `L=4`). That is the
   small-init stall of `theory/13` §3.3's Proposition, measured.

**The unbiased weight trajectory** (40 steps, the object §4.3 showed is the only thing here not
already in the literature):

| L | cos(min-norm step, GD step) | max imbalance at step 40 | ρ |
|---|---|---|---|
| 2 | **+0.921** | 0.038 | 1e-12 |
| 3 | **+0.893** | 0.049 | 4e-9 |
| 4 | **+0.821** | 0.062 | 4e-14 |

So the minimal realisation is **well aligned in direction with the coordinate gradient step and
differs mainly in magnitude and in how it distributes movement across layers**, with the
alignment decaying steadily in depth; and it **approximately preserves balancedness** rather than
drifting away from the balanced gauge. That is a characterisation of the unbiased weight path,
and as far as the §2 survey goes, it is not in the literature.

### 4.3 Result F: does removing the bias change the endpoint? — **RUN 2026-09-17: PASSES**

`studies/endpoint.py`, `runs/theory/endpoint.json`. Two arms from the same starting operator:
`gd` = coordinate gradient flow on the factors, `op` = the counterfactual `Jdot = -G`. For width
>= min(d_in,d_out) the reachable set is unconstrained, so `op` is exactly what the certified
solver would realise and can be integrated directly — **this check needed no solver**. Both arms
integrate the *flow* with adaptive steps (step-size-independent trajectory, and the object
Arora Thm 1 is stated for), stopping at `||grad||/||grad_0|| < 1e-11`. d=20, 3 seeds.

| task | minimiser | L | endpoint gap | gd err vs truth | op err vs truth | gd err vs min-norm | op err vs min-norm | gd stable rank | op stable rank |
|---|---|---|---|---|---|---|---|---|---|
| aniso (cond 30) | **unique** | 2 | **0.0001 ± 0.0002** | 0.0001 | 0.0000 | — | — | 5.78 | 5.78 |
| aniso | **unique** | 3 | **0.0000 ± 0.0001** | 0.0000 | 0.0000 | — | — | 5.78 | 5.78 |
| underdet (n=10<d=20) | affine | 2 | 0.0517 ± 0.0035 | 0.719 | 0.716 | 0.058 | **0.010** | 3.98 | 3.98 |
| underdet | affine | 3 | 0.0952 ± 0.0063 | 0.724 | 0.716 | 0.102 | **0.010** | 4.00 | 3.98 |
| sensing (m=228 of 400, rank 2) | affine | 2 | **0.732 ± 0.039** | **0.179** | 0.681 | 0.734 | **0.035** | **1.345** | 2.368 |
| sensing | affine | 3 | **0.864 ± 0.060** | **0.076** | 0.681 | 0.865 | **0.037** | **1.347** | 2.364 |
| deep (sensing, Xavier, ‖J₀‖=4.4) | affine | 8 | **0.959 ± 0.017** | 0.527 | 2.638 | 0.973 | 3.485 | 1.650 | 3.145 |

**Verdict: the endpoints differ, and the check passes.** Four readings, in order of importance.

1. **The control behaves exactly as the theory demands.** With a full-rank ill-conditioned input
   covariance the minimiser is unique and the two arms agree to `1e-9` on every converged seed —
   four orders below the smallest real effect. Confirms the user's own prediction, and validates
   the harness: any gap elsewhere is a genuine selection effect, not integration error.
2. **Wherever the minimiser is non-unique the arms select different points, and the gap grows
   with depth**: `underdet` 0.052 → 0.095 and `sensing` 0.73 → 0.86 from `L=2` to `L=3`.
3. **The unbiased arm converges to the minimum-Frobenius-norm solution**, to `1.0%` on
   `underdet` and `3.5%` on `sensing` — as predicted for gradient flow from a small start on a
   convex least-squares problem. The counterfactual is not merely "different", it is a
   *characterised* limit, which is what makes it usable as a control.
4. **The sign is the opposite of the one the paper wanted.** On matrix sensing the biased arm
   recovers the rank-2 ground truth (`err 0.179` at `L=2`, improving to **`0.076` at `L=3`**,
   stable rank 1.345 vs the truth's 1.36) while the unbiased arm does not (`err 0.681`). The
   depth trend independently reproduces Arora, Cohen, Hu & Luo (2019). **Correcting the mismatch
   is actively harmful here.**

**Consequence for the framing.** §6(i) as written — "the bias is removable and here is what it
was costing you" — is not supported: in the canonical implicit-bias test-bed the bias is what
makes recovery work. The defensible claim is the control framing:

> The factorisation bias is a specific, quantifiable regulariser. We can now switch it off exactly
> and at finite step size, so its contribution can be measured rather than inferred — and in the
> one setting where it has been most studied, switching it off *costs* you recovery.

This answers reviewer U6A5's "is correcting the mismatch even desirable?" with a measurement and
an honest **no, not always** — which is a better paper than a claimed universal improvement, and
is the framing to write.

*Caveat, recorded.* The `deep` row is confounded: Xavier at `L=8` gives `‖J₀‖ = 4.4`, so the
unbiased arm converges to the projection of a *large* initial operator, not to the min-norm
solution (`err 3.49`). The gap is real but reads "the unbiased arm inherits its initialisation
while the biased arm does not" rather than "min-norm vs low-rank". Worth its own figure; not
evidence for reading 3.

### 4.4 Separation from NGD — **RUN 2026-09-17: PASSES exactly**

At `W = 0` (d=6, L∈{2,3,4}, 3 seeds), with `P_* = ηA ≠ 0`:

| candidate | `‖ΔW‖` | true residual | ρ |
|---|---|---|---|
| min-norm (constructive + polish) | 0.70 – 2.13 | **1e-15 – 9e-14** | **1e-29** |
| ALS | **0** | 1.00 | — |
| NGD (pseudo-inverse Fisher) | **0** | 1.00 | — |
| GD | **0** | 1.00 | — |

Exactly as §3.3 predicts: every linearised method returns the zero step, the exact solve reaches
the target to machine precision. **`L = 2` included** — Gauss–Newton and NGD discard the very
term that carries the solution.

**A finding that strengthens §3.3 and was not anticipated.** The augmented-Lagrangian solve is
*itself* pinned at `W = 0`: L-BFGS is a first-order method on the penalty, so the same homogeneity
argument applies to it, and started from `ΔW = 0` it never leaves. For `L ≥ 3` it also *slides
back* to the origin from a feasible start when the penalty weight is moderate, and once there the
gradient is exactly zero and it is trapped. **So the balanced-SVD-chain initialiser of §3.4 is not
a convenience, it is required** — the escape from a degenerate point must be constructive, and no
amount of iterating fixes it. That is the cleanest statement of what the method contributes.

### 4.5 Unbalanced excess is measurable and meaningful — **RUN 2026-09-17: PASSES**

`studies/excess.py`, `runs/theory/excess.json`. All 27 deep-linear depth-128 checkpoints from the
grid. Directions are filtered at `s_k > 1e-12 · s_max`; without that filter the tail of the
spectrum sits at `s/s_max ~ 1e-16` with several directions sharing an identical value — a
numerically zero cluster whose singular vectors are arbitrary, and which produces spurious
"violations" of a theorem that cannot fail.

| init | task | `K/L` median | `K/L` range | unbalancedness | violations of `K ≥ L` | violations of Lemma B |
|---|---|---|---|---|---|---|
| identity | teacher_* | **1.000** | [1.000, 1.000] | 1.5e-6 – 2.5e-6 | 0 | 0 |
| orthogonal | teacher_* | **1.000** | [1.000, 1.000] | 1.5e-6 – 2.4e-6 | 0 | 0 |
| identity / orthogonal | mnist1d | 1.045 – 1.073 | [1.013, 1.208] | 4.69 | 0 | 0 |
| **xavier** | all | **38 – 230** | [9.8, **838**] | 8.9 – 9.1 | 0 | 0 |

**The excess is not a distinction without a difference — it spans two and a half orders of
magnitude.** Where the network is balanced (`unbalancedness ~1e-6`) we measure `K/L = 1.000`
exactly, which is Arora–Cohen–Hazan Thm 1 reproduced to six digits and is a strong check on the
whole `c_k` machinery. Where it is not (Xavier at depth 128, unbalancedness ≈ 9) the gain is
`38–230×` the balanced value. Any claim of the form "results extend to *small* unbalancedness"
(Cohen's open-problem list) would not reach this regime.

`K_k ≥ L` held on every usable direction across all 27 checkpoints. That is **evidence, not a
proof**: Lemma B only gives `K_k ≥ L(s_k/g)^{2/L}`, and the stronger form needs alignment.

### 4.6 Reachability floor — **RUN 2026-09-17: PASSES, and it settles the premise**

Same checkpoints. `‖(I − Π_{range M})Δ_*‖ / ‖Δ_*‖` by LSQR applied matrix-free to `M`, never
forming `M^ᵀM` (which would square a condition number that is the whole point at depth 128):

| init | reachability floor |
|---|---|
| identity / orthogonal | 5.9e-16 – 1.0e-13 |
| **xavier (depth 128)** | 7.3e-6 – **1.7e-4** |

**The prescribed operator step is reachable essentially exactly, even at Xavier depth 128 where
the operator has collapsed to stable rank ~1.1.** The first-order reachable set is not the
obstruction. Combined with §4.2 — where ALS overshoots the minimal realisation by up to `447×` —
this settles the question that opened this line of work: **the difficulty is the solver, not the
objective.**

**All checks are done. Every one passed.** §4.1 was settled by proof (and retracted the claim it
was testing); §4.2, §4.3, §4.4, §4.5 and §4.6 all ran and all passed. No section was killed. The
framing, however, has to change — see §6.

---

## 5. Experiment plan

| Fig | Content | Why |
|---|---|---|
| 1 | Zero-init separation: DLN `L∈{2,3,8}`, `W=0`. Six baselines + NGD + LLQR pinned at exactly `0` (report `‖ΔW‖ = 0` to machine precision, and the order-`<L` derivative vanishing); ours reaches the global optimum in one solve | Result C + D, and the answer to reviewer 4jZv |
| 2 | Mode-learning order, arm 1 vs arm 2, `L ∈ {2,8,32}` | Result F — the paper's reason to exist |
| 3 | Endpoint comparison: effective rank and test error vs depth, both arms | Result F(c,d) — answers reviewer U6A5 |
| 4 | FGLN multi-pattern: `cos(ΔJ_GD,−G)` and `Δ_ℓ` across gate patterns sharing one `W` — the quantity with no single conserved value | Lemma B' item 2 |
| 5 | `K_k − L` vs unbalancedness on real trajectories; certificate `ρ` per step | Results B(old)/D, honesty panel |

Conditioning held fixed by construction across arms — reviewer NGj5's objection becomes
structurally impossible, as in the earlier plan.

---

## 6. Framing, after all six checks

Every check passed, and none of them rescued the framing they were meant to support. What they
did instead was move the contribution decisively onto the **solver**.

**What the checks killed.** §4.3 showed the endpoints differ — but "deep factorisation selects
low rank, unfactored GD selects minimum Frobenius norm" is Gunasekar et al. 2017 / Arora et al.
2019 / Li–Luo–Lyu 2021, and in the deep linear case the unbiased arm's *operator* path is plain
convex GD, fully understood. The endpoint result is a **validation of the instrument**, not a
contribution. And the bias turned out to *help*, so the ablation cannot be sold as an
improvement.

**What the checks produced instead**, in descending order of strength:

1. **ALS's realisation is not minimal, by up to `447×`, and `ρ(ALS) ≈ 1`** (§4.2). The submitted
   method reaches the operator target while moving the weights almost orthogonally to the optimal
   direction, and it degrades with depth. This is a defect of the method that was never
   identified, it is measured, and the min-norm solve repairs it.
2. **The target is reachable to `1e-4` even at Xavier depth 128** (§4.6). The objective is sound;
   the solver was the problem. This is the premise of the whole direction, now established rather
   than assumed.
3. **The exact solve separates from every linearised method, and not only at a measure-zero
   point** (§4.2, §4.4). At `W=0` it is the only candidate that moves at all; across the entire
   small-initialisation regime NGD's true residual is `13`–`910` against a unit-scale target.
   And the escape must be **constructive** — a first-order solver on the exact objective is
   pinned too (§4.4).
4. **The unbiased weight path is characterised** (§4.2): `cos` with the GD step `+0.92 → +0.82`
   as `L: 2 → 4`, and it approximately preserves balancedness. This is the one object in the
   project that the §2 survey does not already contain.
5. **The unbalanced excess is real**, `K/L` from `1.000` (balanced, matching Arora to six digits)
   to `230` (Xavier depth 128) (§4.5).

**The framing to write.** Not "the bias is removable and here is what it cost you", but:

> Realising a prescribed operator step in the weights is an exactly solvable problem with a
> single-multiplier optimality certificate. The submitted ALS solver is far from its optimum, and
> every linearised alternative — natural gradient included — fails wherever the operator is small.
> The minimal realisation is computable, certifiable, and works from initialisations at which the
> whole first- and second-order family provably cannot move.

The bias ablation stays in, but demoted to the section that *validates* the instrument and
answers reviewer U6A5's "is correcting the mismatch desirable?" with an honest **no, not always**.

**Venue realism, revised.** This is now an optimisation paper with a small theory hook. The `447×`
ALS result plus the degenerate-init separation is a credible main-track contribution *if* the
method is shown to scale beyond `d=6`; without that it is a strong workshop paper. The scaling
question is the next thing to settle, and it was not part of the original check list.

**What is explicitly no longer claimed:** a new characterisation of the implicit bias (§2.1
closed that in 2018); a gated closed-form gap (§3.2 retracted); `K_k ≥ L` as a hypothesis-free
theorem (§0 correction); and any universal benefit from removing the bias (§4.3).

## 7. What we must cite and how

Unconditional citations, with the role each plays:

- **Arora–Cohen–Hazan 2018** — the closed form we generalise away from. Cite in the abstract.
- **Cohen 2024 lecture notes** — the source for "unbalanced is open" and "discrete is open".
- **Du–Hu–Lee 2018** — conservation. Prop. 4.5 of the ICML paper is its gated form; Lemma B'
  explains why: gate absorption makes the FGLN flow literally the linear one.
- **Bah et al. 2022** — "the bias is a Riemannian metric"; our arm 2 replaces it with the
  Euclidean one. This is the cleanest one-sentence description of the paper.
- **Bréchet et al.** — depth-removal precursor (constrained parameterisation).
- **Bernacchia et al. 2018** — NGD already does this to first order. Cite *early*, and state the
  distinction ourselves in §3.4.
- **Rawal–DeWeese 2026** — the AM–GM step. Cite at Lemma B.
- **Evans–Tanner 2026** — closest min-norm-perturbation work.
- **Dufort-Labbé et al. 2026** — concurrent layer-coupled solver.
- **Nguyen et al. 2022** — gated linear implicit bias (asymptotic).
- **Saxe et al.; Gunasekar et al.; Arora et al. 2019; Razin–Cohen 2020; Li–Luo–Lyu 2021;
  Jacot et al.** — the phenomenon being ablated.
- **Shang et al.** — Schatten-`2/L`, used not claimed.
- **Haas et al. (ICML 2026)** — our Prop 7.1 got `2−2/L` for FGLN without balancing but under
  "deliberately strong" assumptions that were "tested empirically but not rigorously proven".
  Result A + B is the rigorous replacement, and saying so is the honest internal narrative.
