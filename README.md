# Operator-Level Optimization

Repository for reproducing operator-projection experiments and figures accompanying the NeurIPS submission *Operator Projection for Depth-Robust Optimization in Factored Networks* (anonymous code release).

## Artifacts layout

- **`outputs/`** — run logs, JSON metrics, and intermediate plots from training scripts. This directory is **gitignored**; regenerate locally after cloning.
- **`images/`** — **tracked** PNG (or PDF) exports meant for this README and for syncing with the LaTeX paper’s `images/` includes. Add generated files here; filenames below match `notes/operator_optimizer/neurips26_als.tex` in the companion paper repo where applicable.

## Core idea

Weight-space optimizers update parameters `W`, but the map depends on the end-to-end operator `P(W)`. This code builds operator-space targets (typically gradient targets on `P`) and solves for layer updates with ALS-style block solves, plus warm-starts that counter spectral collapse of context matrices.

## Source tree (`src/`)

- `src/operator_level_optimization/core/optim/` — core optimizers: `operator.py`, `muon.py`, `kfac.py`, `shampoo.py`, `soap.py`
- `src/operator_level_optimization/models/fgln.py` — FGLN + masked ALS
- `src/operator_level_optimization/scripts/train/` — `deep_linear_compare.py`, `fgln_compare.py`, `mnist_smoke.py`
- `src/operator_level_optimization/scripts/figures/` — plotting helpers for deep linear / FGLN
- `src/operator_level_optimization/scripts/toy2d.py` — 2D toy trajectories and one-step figures

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

---

## Paper figures (`images/`)

Regenerate from `outputs/...` runs, then copy or symlink exports into `images/` with the names below so README and LaTeX stay aligned.

| File | Paper | What to plot | Status |
|------|--------|----------------|--------|
| `toy_2d_trajectories.png` | Fig. `fig:2d-traj` | Operator-space trajectories, $L=2$ and $L=32$, same $P_0$, all methods | *placeholder — generate* |
| `toy_2d_L2_h2.png` | Fig. `fig:2d` (appendix) | One-step $\Delta P$ arrows in $\mathbb{R}^{1\times2}$, $L=2$, $h=2$, Xavier | *placeholder — generate* |
| `deeplinear_convergence.png` | Fig. `fig:deeplinear-convergence` | Rel.\ error $\|P-P^\star\|_F/\|P^\star\|_F$ vs step, Xavier, $d=16$, $L=128$ | *placeholder — generate* |
| `deeplinear_spectrum.png` | Fig. `fig:deeplinear-spectrum` | SVD spectrum of $P(t)$, Xavier, $L=128$ | *placeholder — generate* |
| `deeplinear_convergence_identity.png` | Fig. `fig:deeplinear-convergence-identity` | Same as convergence panel, identity init, $\lambda=0$, 500 steps | *placeholder — generate* |
| `deeplinear_spectrum_identity.png` | Fig. `fig:deeplinear-spectrum-identity` | Spectrum snapshots at steps $\{0,50,500\}$, identity init | *placeholder — generate* |
| `mnist_mlp_depth.png` | Fig. `fig:mnist-depth` | Best val loss vs depth $L\in\{1,2,4,8,16,32\}$, MLP $784\to[32]^L\to10$ | *placeholder — generate* |

### Appendix-only figures (`images/`)

| File | Paper | What to plot | Status |
|------|--------|----------------|--------|
| `toy_2d_identity_traj_L256.png` | Fig. `fig:2d-id-traj-L256` | Trajectories at $L=256$, $h=2$, identity init — all optimizers converge | *placeholder — generate* |
| `toy_2d_haar_traj_L256.png` | Fig. `fig:2d-haar-traj-L256` | Same at $L=256$, Haar orthogonal init — GD methods fail, ALS-Exact ok | *placeholder — generate* |
| `loss_fgln.png` | Fig. `fig:fgln-convergence` | FGLN rel.\ operator error vs step, $d=16$, $L=128$, Xavier, Bernoulli gates $p=0.9$ | *placeholder — generate* |
| `spectrum_fgln.png` | Fig. `fig:fgln-spectrum` | SVD spectrum of $P(t)$ for same FGLN setting | *placeholder — generate* |

### Embedding in the README (after generation)

Uncomment or add standard markdown once the corresponding files exist under `images/`:

```markdown
![2D trajectories (L=2 and L=32)](images/toy_2d_trajectories.png)
![Deep linear convergence (Xavier, L=128)](images/deeplinear_convergence.png)
```

---

## Main-body tables (export to `images/` or document in JSON)

| Artifact | Paper | Content | Status |
|----------|--------|---------|--------|
| One-step 2D cosine / norm table | Tab. `tab:2d-cos` | Methods vs cosine w.r.t.\ ideal $\Delta P^\star$ and $\|\Delta P\|_F$ | *placeholder — paste from run summary* |
| Per-step cost table | Tab. `tab:cost` | Asymptotic costs (GD, Muon, ALS-exact linear/FGLN, ALS-MN, ALS-exact MLP) | *optional figure: typeset table as PNG* |

---

## Appendix B: solver variants and approximations

Cross-reference: Appendix `\ref{app:variants}` and subsections in the paper. Below is a checklist of **what to benchmark or ablate**, suggested metrics, and where results should land. Implementation may be partial in this repo; placeholders record intent.

### B.1 Mean-field (per-sample solve, then batch mean)

- **Idea:** $\bar{\Delta\theta} = \frac{1}{B}\sum_b \arg\min_{\Delta\theta} \mathcal{L}_b(\Delta\theta)$ vs exact shared $\Delta\theta$.
- **Metrics:** operator alignment (cosine / rel.\ Frobenius error on $\Delta P$), train MSE, steps to threshold.
- **Regimes:** MLP frozen-gate batches (paper: not accurate enough); deep linear/FGLN (paper: collapses to exact equivalence in the trivial sense).
- **Output:** `outputs/ablations/mean_field_mlp/results.json` (+ optional plot in `images/ablation_mean_field_mlp.png`).

### B.2 Linearized operator objective

| Variant | Description | Metrics | Suggested output |
|---------|-------------|---------|------------------|
| **Linearized exact (coupled)** | Single coupled normal system over all $\Delta W_\ell$ | Same as one-step / short-horizon operator error vs ALS-exact nonlinear objective | `outputs/ablations/linobj_coupled/` |
| **Block-diagonal (one-shot)** | Drop cross-layer blocks; per-layer Sylvester with batch Grams | Stress cross-layer coupling: deep $L$, large steps | `outputs/ablations/linobj_block_diag/` |

### B.3 Operator-KFAC

- **Idea:** Operator-context statistics + K-FAC-style factored inverse (not full Sylvester solve).
- **Metrics:** vs classical K-FAC and ALS-MN on depth sweep or operator cosine.
- **Output:** `outputs/ablations/operator_kfac/` + `images/ablation_operator_kfac_depth.png` *placeholder*.

### B.4 Divide-and-conquer (D\&C) on operator projection

| Variant | Notes | Output dir *placeholder* |
|---------|--------|--------------------------|
| **D\&C baseline** | Recursive two-factor solves; valid deep linear / FGLN | `outputs/ablations/dnc_linear/` |
| **D\&C + cross-sample target penalty** | Node-level penalty to align targets across samples | `outputs/ablations/dnc_penalty/` |
| **D\&C + node linearized schedule** | Strengthen batch coupling root→leaves | `outputs/ablations/dnc_schedule/` |
| **D\&C + multi-pass tree** | Several tree passes per outer step | `outputs/ablations/dnc_multipass/` |
| **D\&C on MLP (expected fail)** | Per-sample subtree solves + average — documents incoherence | `outputs/ablations/dnc_mlp_fail/` + short note / plot |

### B.5 Secant gate linearization (`secant`, `secant_r`)

| Variant | Description | Metrics | Output *placeholder* |
|---------|-------------|---------|----------------------|
| **Rank-1 secant** | $\widehat A_\ell$, $\widehat B_\ell$ rank-1 surrogates | Val loss / operator error vs full frozen-gate ALS | `outputs/ablations/secant_r1/` |
| **Rank-$r$ secant (`secant_r`)** | Richer subspace per layer | Cost vs accuracy tradeoff | `outputs/ablations/secant_r/` |
| **Approx vs exact gradient** | Forward-derived vs exact backprop direction + secant curvature | Stability at depth | `outputs/ablations/secant_grad_exact/` |

### B.6 Adaptive per-layer $\lambda_k$ (Appendix adaptive regularization)

- **Idea:** $\lambda_k \propto \sigma_{\max}(M_k)+\sigma_{\max}(N_k)$ on MLP ALS-MN blocks.
- **Metrics:** per-layer step norms, residual decay, failure mode diagnostics (noise amplification in middle layers).
- **Output:** `outputs/ablations/adaptive_lambda/` + learning curves in `images/ablation_adaptive_lambda.png` *placeholder*.

### B.7 Core methods already in main experiments (reference)

| Method | Role in paper | Typical script knob |
|--------|----------------|---------------------|
| ALS-exact (deep linear / FGLN) | Exact modewise solve, $O(Ld^3)$ | `als_exact`, masked ALS |
| ALS-exact (MLP) | Sum of Kronecker normal eq., $O(Ld^6)$ | `operator.py` exact path |
| ALS-MN | Kronecker approx.\ of batch normal eq. | MLP approximate solve |
| Heavy Ball, Adam, Muon, K-FAC, Shampoo, SOAP | Baselines | `deep_linear_compare`, `fgln_compare`, `mnist_smoke` |

---

## Reproduce paper runs (commands)

### 1) Deep linear ($L=128$)

```bash
python -m operator_level_optimization.scripts.train.deep_linear_compare \
  --out_dir outputs/deep_linear/l128_ginibre_msegrad \
  --depth 128 \
  --steps 2000 \
  --target_mode mse_grad \
  --methods heavyball adam muon kfac shampoo soap als_exact
```

Optional plotting:

```bash
python -m operator_level_optimization.scripts.figures.plot_deep_linear_figures \
  --run_dir outputs/deep_linear/l128_ginibre_msegrad
```

### 2) FGLN ($L=128$, e.g.\ $p=0.9$ / $p=0.5$)

```bash
python -m operator_level_optimization.scripts.train.fgln_compare \
  --out_dir outputs/fgln/fgln_compare_p09_L128 \
  --p 0.9 \
  --depth 128 \
  --steps 10000 \
  --als_lam 1e-4 \
  --als_sweeps 4 \
  --als_gateperm_warmstart_once \
  --als_lam_anchor_post_warmstart \
  --spec_every 50
```

Spectrum snapshots:

```bash
python -m operator_level_optimization.scripts.figures.plot_fgln_spectrum_snapshots \
  --run_dir outputs/fgln/fgln_compare_p09_L128
```

### 3) Toy 2D

```bash
python -m operator_level_optimization.scripts.toy2d
```

### 4) MNIST MLP smoke / depth sweep

```bash
python -m operator_level_optimization.scripts.train.mnist_smoke --help
```

*(Extend or add a dedicated depth-sweep driver to match paper protocol: seeds, LR grid, best checkpoint per depth.)*

### 5) Appendix-style solver smoke (CI-friendly)

Runs tiny deep-linear, MLP (`OperatorLevelMLP`), and FGLN configurations for each implemented variant; writes `outputs/smoke/variants/smoke_variants.json` and `outputs/smoke/variants/config.json` (CLI, environment, summary defaults). Three D\&C heuristics from the paper text are **skipped** (not implemented).

```bash
python -m operator_level_optimization.scripts.smoke_variants
```

### 6) Variant training curves (appendix-style optimizers)

Generates per-variant loss plots and overlays (deep linear, tiny gated MLP on synthetic CE, FGLN), plus ``variant_curves.json`` under ``--out_dir``.

```bash
python -m operator_level_optimization.scripts.train.variant_loss_curves \\
  --out_dir outputs/variant_curves/run01
```

Use shorter runs while iterating on layout or hyperparameters, for example
``--steps_deeplinear 80 --steps_mlp 60 --steps_fgln 40``.

Diverging traces are **clipped for display only** at ``--plot_y_max`` (default ``10``) so overlays stay readable; raw metrics remain in ``variant_curves.json``. Disable with ``--plot_y_max 0``.

Canonical hyperparameters for the run are in ``config.json`` (same directory as the figures).

---

## Notes

- Deep linear and FGLN training defaults favor `float64` for deep settings.
- After generating files under `outputs/`, export publication figures into `images/` using the table filenames so this README and the LaTeX `\\includegraphics{images/...}` paths stay consistent.
