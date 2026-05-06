# Operator-Level Optimization

Clean paper-focused repository for reproducing the main operator-level optimization figures.

## Core idea

Weight-space optimizers update parameters `W`, but the prediction map depends on the end-to-end operator `P(W)`.
This project builds operator-space targets (typically MSE-gradient targets) and solves for layer updates with ALS-exact style block solves.

## Source tree (`src/`)

- `src/operator_level_optimization/core/optim/`
  - Core optimizer implementations: `operator.py`, `muon.py`, `kfac.py`, `shampoo.py`, `soap.py`
- `src/operator_level_optimization/models/`
  - `fgln.py`: Fixed-Gates Linear Network model + native masked ALS solver
- `src/operator_level_optimization/scripts/train/`
  - `deep_linear_compare.py`: deep linear benchmark runs
  - `fgln_compare.py`: FGLN benchmark runs (includes warmstart-once + lambda-anchor options)
- `src/operator_level_optimization/scripts/figures/`
  - `plot_deep_linear_figures.py`
  - `plot_fgln_spectrum_snapshots.py` (3 snapshots: init/mid/final)
- `src/operator_level_optimization/scripts/toy2d.py`
  - 2D toy visualization and trajectory experiments

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Reproduce paper runs

### 1) Deep linear (L=128)

```bash
python -m operator_level_optimization.scripts.train.deep_linear_compare \
  --out_dir outputs/deep_linear/l128_ginibre_msegrad \
  --depth 128 \
  --steps 2000 \
  --target_mode mse_grad \
  --methods heavyball adam muon kfac shampoo soap als_exact
```

Optional deep linear plotting:

```bash
python -m operator_level_optimization.scripts.figures.plot_deep_linear_figures \
  --run_dir outputs/deep_linear/l128_ginibre_msegrad
```

### 2) FGLN (L=128, p=0.9 / p=0.5)

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

```bash
python -m operator_level_optimization.scripts.train.fgln_compare \
  --out_dir outputs/fgln/fgln_compare_p05_L128 \
  --p 0.5 \
  --depth 128 \
  --steps 10000 \
  --als_lam 1e-6 \
  --als_sweeps 4 \
  --als_gateperm_warmstart_once \
  --als_lam_anchor_post_warmstart \
  --spec_every 50
```

3-snapshot spectrum panels:

```bash
python -m operator_level_optimization.scripts.figures.plot_fgln_spectrum_snapshots \
  --run_dir outputs/fgln/fgln_compare_p09_L128
```

```bash
python -m operator_level_optimization.scripts.figures.plot_fgln_spectrum_snapshots \
  --run_dir outputs/fgln/fgln_compare_p05_L128
```

### 3) Toy 2D

```bash
python -m operator_level_optimization.scripts.toy2d
```

## Notes

- Outputs are intended under `outputs/` (ignored by git).
- Deep linear and FGLN scripts default to `float64` for numerical stability in deep settings.
