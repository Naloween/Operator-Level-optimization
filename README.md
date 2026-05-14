# Operator-Level Optimization

Repository for reproducing operator-projection experiments and figures accompanying the NeurIPS submission *Operator Projection for Depth-Robust Optimization in Factored Networks* (anonymous code release).

## Core idea

Weight-space optimizers update parameters `W`, but the map depends on the end-to-end operator `P(W)`. This code builds operator-space targets (typically gradient targets on `P`) and solves for layer updates with ALS-style block solves, plus warm-starts that counter spectral collapse of context matrices.

## Source tree (`src/`)

- `src/operator_level_optimization/core/optim/` — core optimizers: `operator.py`, `muon.py`, `kfac.py`, `shampoo.py`, `soap.py`
- `src/operator_level_optimization/models/fgln.py` — FGLN + masked ALS
- `src/operator_level_optimization/scripts/train/` — `deep_linear_compare.py`, `fgln_compare.py`, `mnist_smoke.py`, `variant_loss_curves.py`
- `src/operator_level_optimization/scripts/figures/` — plotting helpers for deep linear / FGLN
- `src/operator_level_optimization/scripts/toy2d.py` — 2D toy trajectories and one-step figures

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

---

## Reproducing Paper Figures

### 1) 2D toy — one-step arrows and trajectories

Generates `outputs/paper/toy2d/trajectory_paper.png` and `outputs/paper/toy2d/one_step_paper.png`.

```bash
python -m operator_level_optimization.scripts.toy2d \
  --paper_depths 2 32 \
  --out_traj outputs/paper/toy2d/trajectory_paper.png \
  --out     outputs/paper/toy2d/one_step_paper.png
```

### 2) Deep linear — Xavier init ($d=16$, $L=128$)

Generates convergence and spectrum figures.

```bash
python -m operator_level_optimization.scripts.train.deep_linear_compare \
  --out_dir outputs/deep_linear/l128_xavier \
  --depth 128 --d 16 --n 64 \
  --steps 10000 \
  --init_mode xavier \
  --target_mode mse_grad \
  --methods heavyball adam muon kfac shampoo soap als_exact

python -m operator_level_optimization.scripts.figures.plot_deep_linear_figures \
  --run_dir outputs/deep_linear/l128_xavier
```

### 3) Deep linear — identity init ($d=16$, $L=128$, $\lambda=0$)

Generates convergence and spectrum figures isolating direction mismatch without spectral collapse.

```bash
python -m operator_level_optimization.scripts.train.deep_linear_compare \
  --out_dir outputs/deep_linear/l128_identity \
  --depth 128 --d 16 --n 64 \
  --steps 500 \
  --init_mode identity \
  --als_lam 0.0 \
  --target_mode mse_grad \
  --methods heavyball adam muon kfac shampoo soap als_exact

python -m operator_level_optimization.scripts.figures.plot_deep_linear_figures \
  --run_dir outputs/deep_linear/l128_identity
```

### 4) FGLN ($L=128$, $p=0.9$)

Generates FGLN convergence and spectrum figures.

```bash
python -m operator_level_optimization.scripts.train.fgln_compare \
  --out_dir outputs/fgln/fgln_compare_p09_L128 \
  --p 0.9 --depth 128 --steps 10000 \
  --als_lam 1e-4 --als_sweeps 4 \
  --als_gateperm_warmstart_once \
  --als_lam_anchor_post_warmstart \
  --spec_every 50

python -m operator_level_optimization.scripts.figures.plot_fgln_spectrum_snapshots \
  --run_dir outputs/fgln/fgln_compare_p09_L128
```

### 5) MNIST MLP depth sweep

Trains bias-free ReLU MLP at depths $L \in \{1, 2, 4, 8, 16, 32\}$ and plots best val loss vs depth.

```bash
python -m operator_level_optimization.scripts.train.mnist_smoke \
  --out_dir outputs/mnist/depth_sweep \
  --depths 1 2 4 8 16 32 \
  --epochs 50 --batch_size 128 \
  --seeds 0 1 2

python -m operator_level_optimization.scripts.figures.plot_mnist_depth \
  --run_dir outputs/mnist/depth_sweep
```

### 6) Appendix 2D toy — $L=256$ trajectories (identity and Haar init)

```bash
python -m operator_level_optimization.scripts.toy2d \
  --trajectory --depth 256 --hidden 2 \
  --init_mode identity --n_steps 300 \
  --out_traj outputs/paper/toy2d/identity_traj_L256.png

python -m operator_level_optimization.scripts.toy2d \
  --trajectory --depth 256 --hidden 2 \
  --init_mode haar --n_steps 300 \
  --out_traj outputs/paper/toy2d/haar_traj_L256.png
```

### 7) Appendix solver smoke (CI-friendly)

Runs tiny deep-linear, MLP, and FGLN for each implemented variant; writes `outputs/smoke/variants/smoke_variants.json` and `config.json`.

```bash
python -m operator_level_optimization.scripts.smoke_variants
```

### 8) Variant training curves (appendix-style optimizers)

Generates per-variant loss plots and overlays (deep linear D&C, MLP approximations, FGLN adaptive-λ) plus `variant_curves.json` and `config.json`.

```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/run01
```

For a more convincing demonstration closer to paper scale:

```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/canonical \
  --deeplinear_depth 32 --deeplinear_d 8 --deeplinear_n 64 --steps_deeplinear 2000 \
  --fgln_depth 32 --fgln_d 8 --fgln_n 64 --steps_fgln 500 \
  --mlp_batch 128 --steps_mlp 400
```

---

## Variant Experiments — Evidence for Appendix Claims

For every approximation and variant described in the paper's Appendix B, the sub-sections below specify the **claim** being evidenced, the **generate command**, and the **expected result / interpretation**. Figures will be added here as they are generated.

---

### B.1 Mean-field approximation (App. B.1)

**Claim:** Per-sample ALS solved independently then averaged is not accurate enough in the MLP case — the expectation–minimization swap introduces bias under gate heterogeneity. In the deep linear case (no ReLU), all samples share the same operator context, so per-sample and batch solves are mathematically equivalent.

**Experiment:** Two-panel controlled comparison. Same optimizer (ALS, 3 sweeps, warmstart, lr=0.15), same architecture (4 layers, d=32, batch=64) — only the activation function differs.
- *Left panel — deep linear (no ReLU):* `als_layer_solve="per_sample"` must overlap `als_layer_solve="mn"` because D_l=1 for every sample → gate heterogeneity is zero.
- *Right panel — ReLU MLP:* per-sample solves push shared weights in incompatible directions (each sample activates a different ~50% of neurons per layer); their average satisfies nobody's equations.

**Generate:**
```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/mean_field \
  --steps_deeplinear 2 --steps_mlp 2 --steps_fgln 2 \
  --steps_kfac_depth 2 --kfac_depths 2 \
  --steps_mean_field 300 \
  --mf_depth 4 --mf_hidden 32 --mf_d_in 32 --mf_d_out 8 \
  --mf_batch 64 --mf_lr 0.15 --mf_lam 1e-3 --mf_n_sweeps 3
# Output: outputs/variant_curves/mean_field/mean_field_comparison.png
```

![Mean-field vs batch ALS — deep linear (overlap) and ReLU MLP (diverge)](images/ablation_mean_field_mlp.png)

**Result:** Deep linear: both curves are near-identical (final CE ≈ 1.82 for both), confirming the theoretical equivalence. ReLU MLP: batch ALS-MN reaches CE ≈ 1.02 while mean-field stalls at ≈ 1.73 (barely below the random-init level of ≈ 2.08 for 8 classes). The gap is caused entirely by gate heterogeneity — the same optimizer, same data, same initialization, only the activation function changed.

---

### B.2 Linearized operator objective (App. B.2)

**Claim:** The linearized operator objective is a good approximation at small learning rates but breaks down at large ones, where only ALS-exact (which solves the true nonlinear objective) remains stable.

**Experiment:** Bias-free ReLU MLP (depth=3, hidden=16, d_in=16, d_out=8) trained to match a random linear teacher (P*x) via MSE. Three solvers compared across two learning rate regimes:
- **ALS-exact** — iterative (5 sweeps, no gate-permutation warmstart), solves the true nonlinear operator objective.
- **Linearized-exact** — one-shot coupled solve via first-order Taylor expansion of the operator objective.
- **Block-diagonal** — one-shot per-layer solve, decouples inter-layer coupling entirely.

**Generate:**
```bash
venv/bin/python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/linobj \
  --linobj_lr_small 0.05 --linobj_steps_small 200 \
  --linobj_lr_large 1.0  --linobj_steps_large 60
# Output: outputs/variant_curves/linobj/linearized_obj_comparison.png
```

**Results (seed=0):**

| Panel | Solver | Final MSE (step) |
|---|---|---|
| Small lr=0.05 (200 steps) | ALS-exact | 1.183 |
| Small lr=0.05 (200 steps) | Linearized-exact | 0.064 |
| Small lr=0.05 (200 steps) | Block-diagonal | 0.046 |
| Large lr=1.0  (60 steps)  | ALS-exact | 0.620 |
| Large lr=1.0  (60 steps)  | Linearized-exact | NaN (diverged) |
| Large lr=1.0  (60 steps)  | Block-diagonal | 1.334 (stalled) |

At small lr all three are stable; block-diagonal converges fastest (efficient per-layer Newton step), ALS-exact slowest (constrained to factored manifold, no warmstart). At large lr, the first-order Taylor approximation collapses immediately for linearized-exact; block-diagonal stalls at a poor local minimum; ALS-exact, operating on the exact nonlinear objective, stays monotonically convergent.

![Linearized objective: approximation hierarchy](images/ablation_linobj.png)

---

### B.3 Operator-KFAC depth sweep (App. B.3)

**Claim:** Operator-KFAC uses better (operator-aligned) statistics than classical K-FAC but avoids the full Sylvester solve. This should give an advantage over classical K-FAC at depth while still trailing ALS-exact.

**Experiment:** Depth sweep comparing `operator_kfac` vs `block_diagonal` vs `als_mn` across $L \in \{2, 4, 8\}$ on a synthetic cross-entropy task with a purely linear MLP (no ReLU), plotting final loss vs depth.

**Generate:**
```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/canonical \
  --steps_kfac_depth 200 \
  --kfac_depths 2,4,8 \
  --kfac_d_in 32 --kfac_hidden 32 --kfac_d_out 16 --kfac_batch 64
# Relevant outputs:
#   outputs/variant_curves/canonical/operator_kfac_depth_summary.png
#   outputs/variant_curves/canonical/operator_kfac_depth_2.png
#   outputs/variant_curves/canonical/operator_kfac_depth_4.png
#   outputs/variant_curves/canonical/operator_kfac_depth_8.png
```

**Expected result:** At each depth, `operator_kfac` outperforms `block_diagonal` (due to operator-aligned geometry) but trails `als_mn` (which solves Sylvester exactly). The gap between `operator_kfac` and `block_diagonal` should grow with depth.

<!-- ![Operator-KFAC depth sweep](images/ablation_operator_kfac_depth.png) -->

---

### B.4 Divide-and-Conquer (D&C) solver

#### B.4a D&C on deep linear / FGLN — valid case (App. B.4)

**Claim:** D&C works correctly for deep linear and FGLN networks. Sub-operators at every node are deterministic, so node targets propagate cleanly down the tree.

**Experiment:** D&C (`dc`, both `identity_delta` and `plain` init modes) vs ALS-exact on the deep linear teacher-student task.

**Generate:**
```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/canonical \
  --deeplinear_depth 32 --deeplinear_d 8 --deeplinear_n 64 --steps_deeplinear 2000
# Relevant outputs:
#   outputs/variant_curves/canonical/deep_linear_rel_operator_error.png  (overlay)
#   outputs/variant_curves/canonical/deep_linear_mse_als_exact.png
#   outputs/variant_curves/canonical/deep_linear_mse_dc_identity_delta.png
#   outputs/variant_curves/canonical/deep_linear_mse_dc_plain.png
```

**Expected result:** D&C and ALS-exact both converge. Node targets propagate cleanly in the deterministic (linear) setting.

<!-- ![D&C vs ALS-exact, deep linear](images/ablation_dc_linear_vs_als.png) -->

---

#### B.4b D&C on MLP — expected failure (App. B.4)

**Claim:** Because gate patterns differ across samples, per-sample solutions push shared factors in incompatible directions. The averaged update satisfies no individual sample's equations, and the resulting operator residual does not decrease reliably.

**Experiment:** `dc_mlp` (per-sample ALS → batch average) vs `mlp_als_mn` on the synthetic cross-entropy task.

**Generate:**
```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/canonical \
  --steps_mlp 400 --mlp_batch 128
# Relevant outputs:
#   outputs/variant_curves/canonical/loss_mlp_dc_mlp.png
#   outputs/variant_curves/canonical/loss_mlp_als_mn.png
```

**Expected result:** `dc_mlp` stalls or diverges relative to `mlp_als_mn`, illustrating that gate heterogeneity makes per-sample decomposition incoherent when the results are averaged back to shared weights.

<!-- ![D&C MLP failure](images/ablation_dc_mlp_fail.png) -->

---

### B.5 Secant gate linearization

#### B.5a Rank-1 secant vs ALS-MN (App. B.5)

**Claim:** Rank-1 secant is much cheaper than full ALS context systems but introduces bias because only rank-1 secant directions are retained.

**Experiment:** `mlp_secant` (rank-1) vs `mlp_als_mn` (full batch ALS) on the synthetic cross-entropy task.

**Generate:**
```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/canonical \
  --steps_mlp 400 --mlp_batch 128
# Relevant output:
#   outputs/variant_curves/canonical/loss_mlp_secant.png
```

**Expected result:** `mlp_secant` converges more slowly than `mlp_als_mn`. The rank-1 bias reflects the cost-accuracy trade-off: cheaper context (no frozen gate storage) at the cost of convergence quality.

<!-- ![Secant rank-1 vs ALS-MN](images/ablation_secant_r1_vs_als.png) -->

---

#### B.5b Rank-1 vs rank-$r$ secant (App. B.5)

**Claim:** Higher-rank secant interpolates between rank-1 secant and richer context representations. Larger $r$ improves fidelity but removes most of the computational advantage since the full frozen gate pattern must be stored.

**Experiment:** `mlp_secant` (rank-1) vs `mlp_secant_r3` (rank-3) vs `mlp_als_mn` (full Sylvester).

**Generate:**
```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/canonical \
  --steps_mlp 400
# Relevant outputs:
#   outputs/variant_curves/canonical/loss_mlp_secant.png
#   outputs/variant_curves/canonical/loss_mlp_secant_r3.png
#   outputs/variant_curves/canonical/mlp_cross_entropy_logy.png  (all variants overlay)
```

**Expected result:** `mlp_secant_r3` outperforms `mlp_secant` and approaches `mlp_als_mn` as rank grows, showing the monotone fidelity–cost trade-off.

<!-- ![Rank-1 vs rank-r secant](images/ablation_secant_rankr.png) -->

---

#### B.5c Secant exact-gradient variant (App. B.5)

**Claim:** Using exact backprop gradient direction while only approximating curvature with rank-1 secant statistics reduces one source of bias but is still insufficient to correctly correct operator mismatch at depth.

**Experiment:** `mlp_secant_grad_exact` (rank-1 curvature, exact gradient) vs `mlp_secant` (both approximate) vs `mlp_als_mn` (exact Sylvester) on the synthetic cross-entropy task.

**Generate:**
```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/canonical \
  --steps_mlp 400 --mlp_batch 128
# Relevant outputs:
#   outputs/variant_curves/canonical/loss_mlp_secant_grad_exact.png
#   outputs/variant_curves/canonical/loss_mlp_secant.png
#   outputs/variant_curves/canonical/loss_mlp_als_mn.png
```

**Expected result:** `mlp_secant_grad_exact` improves over `mlp_secant` (exact gradient removes one source of bias) but still falls short of `mlp_als_mn`, since curvature approximation alone is insufficient at depth.

<!-- ![Secant exact-gradient vs rank-1 vs ALS](images/ablation_secant_exact_grad.png) -->

---

### B.6 Adaptive regularization

#### B.6a Adaptive $\lambda$ failure (MLP) (App. B.6)

**Claim:** Setting $\lambda_k \propto \sigma_{\max}(M_k) + \sigma_{\max}(N_k)$ equalizes per-layer update magnitudes. For middle layers in deep networks, contexts are nearly random and carry no useful direction signal — amplifying those updates injects noise that compounds over depth and ALS sweeps, preventing convergence.

**Experiment:** `mlp_block_diagonal_adaptive_lam` / `mlp_als_adaptive_lam` (adaptive $\lambda$) vs `mlp_block_diagonal` / `mlp_als_mn` (fixed $\lambda$) on the synthetic cross-entropy task.

**Generate:**
```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/canonical \
  --steps_mlp 400 --mlp_batch 128
# Relevant outputs:
#   outputs/variant_curves/canonical/loss_mlp_block_diagonal.png
#   outputs/variant_curves/canonical/loss_mlp_block_diagonal_adaptive_lam.png
#   outputs/variant_curves/canonical/loss_mlp_als_adaptive_lam.png
```

**Expected result:** Adaptive-$\lambda$ variants converge more slowly or less reliably than their fixed-$\lambda$ counterparts.

<!-- ![Adaptive lambda failure (MLP)](images/ablation_adaptive_lambda_mlp.png) -->

---

#### B.6b Adaptive $\lambda$ vs fixed $\lambda$ — FGLN (App. B.6)

**Claim:** The warm-start resolves spectral collapse at its source by restoring bounded context singular values. Adaptive $\lambda$ only rescales the update without restoring the geometry that makes the update meaningful.

**Experiment:** FGLN fixed $\lambda$ (`als_fixed_lam`) vs FGLN adaptive-$\lambda$ step (`als_adaptive_lambda_step`) on the teacher-student task, both starting from Xavier initialization.

**Generate:**
```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \
  --out_dir outputs/variant_curves/canonical \
  --fgln_depth 32 --fgln_d 8 --fgln_n 64 --steps_fgln 500
# Relevant outputs:
#   outputs/variant_curves/canonical/fgln_mse_als_fixed_lam.png
#   outputs/variant_curves/canonical/fgln_mse_als_adaptive_lam.png
#   outputs/variant_curves/canonical/fgln_train_mse.png  (overlay)
```

**Expected result:** `als_fixed_lam` converges while `als_adaptive_lambda_step` stalls or diverges. Fixed $\lambda$ correctly suppresses zero singular-value directions; adaptive $\lambda$ tries to amplify corrections along them, producing instability.

<!-- ![Adaptive lambda vs fixed (FGLN)](images/ablation_adaptive_lambda_fgln.png) -->

---

## Notes

- Deep linear and FGLN training defaults favor `float64` for deep settings.
- Three D&C heuristics (cross-sample target penalty, node-linearized schedule, multi-pass tree) have no code path; the smoke test documents this explicitly.
- All variant figures are generated by `variant_loss_curves.py`. Run the canonical command (depth 32, $d=8$, longer steps) to produce figures suitable for the README.
