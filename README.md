# Operator-Level Optimization

A factored network's behaviour is governed by its input-output operator

```
J(x) = W_L D_{L-1}(x) W_{L-1} ... D_1(x) W_1
```

but training updates the factors `W_l`. The step taken in parameter space does not realize
the step intended in operator space, and the two drift further apart with depth. This
repository is the instrument for measuring that drift and for separating it from the
other thing depth does -- destroying the conditioning of the operator.

Those two effects are usually entangled. The design here keeps them apart:

- **Direction mismatch** is measured directly, for any optimizer, by
  `olo.diagnostics.mismatch` -- two operator evaluations around a step, no solver needed.
- **Conditioning** is removed at the source by the CReLU parameterization with a
  looks-linear initialization, whose Jacobian is *exactly orthogonal at every depth for
  every input*. Nothing has to be warm-started or repaired.

One operator-space method is implemented, exactly: alternating least squares on the
operator-projection objective. It is a control condition, not a proposed optimizer.

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
pytest                     # 83 tests
```

## Run something

```bash
python -m olo.run configs/experiments/e01_deep_linear_identity.yaml \
  --sweep optim.type=als,adam,heavyball,muon,kfac,shampoo,soap \
  --sweep optim.lr=1.0,0.1,0.01,0.001,0.0001 --seeds 0,1,2
```

Each run writes `runs/<name>/seed<k>/` containing `config.yaml`, `meta.json` (git commit,
device, resolved components), `metrics.jsonl`, `arrays.npz` (spectra, per-layer profiles)
and checkpoints. Notebooks in `notebooks/` read those back and write `figures/`.

A single config holds every method's hyperparameters in a block named after it, so
`--sweep optim.type=...` produces a real comparison from one file -- same batches, same
budget, same diagnostics, each method at its own tuned learning rate. Fairness is a
property of the harness rather than a protocol to remember.

## What is here

```
src/olo/
  models/       deep_linear, fgln, relu_mlp, crelu_mlp  -- one gate protocol, four models
  init/         xavier, haar, identity, looks_linear
  optim/
    als.py      exact operator projection (the method)
    targets.py  gradient | shadow | fixed  -- what displacement of P to ask for
    baselines/  heavyball, adam, muon, kfac, shampoo, soap
  tasks/        teacher_student, matrix_sensing, mnist, cifar10
  diagnostics/  mismatch, conditioning, spectrum, linearization, cost
  viz/          plotting library (notebooks stay thin)
configs/experiments/    one file per experiment, fully self-describing
neurips26/              the submitted paper and its reviews, for reference
```

### The four models

All four are the same chain with a different gate `D_l`, which is why one solver covers
them: identity (deep linear), fixed diagonal (FGLN), data-dependent diagonal (ReLU), and
data-dependent *rectangular* (CReLU, where `D(z) = [diag(1[z>0]); -diag(1[z<0])]` maps
`R^d -> R^2d`).

CReLU is written as a single `W in R^{d x 2d}` acting on `c = [relu(z); relu(-z)]`. Since
`||c|| = ||z||` it is norm-preserving, and with `W = [O | -O]`:

```
W D(z) = O diag(1[z>0]) + O diag(1[z<0]) = O      for every z
```

so `J(x) = O_L ... O_1` is orthogonal regardless of depth or input.

| | deep linear | FGLN | ReLU MLP | CReLU MLP |
|---|---|---|---|---|
| `xavier` | yes | yes | yes | yes |
| `haar` | yes | yes | yes | yes |
| `identity` | yes | yes | yes | yes (`[I \| -I]`) |
| `looks_linear` | rejected | rejected | mirror blocks, even width | `[O \| -O]`, exact |

### The solver

`OperatorALS` minimizes `|| prod_l (W_l + dW_l) - P_tgt ||^2 + lam sum_l ||dW_l||^2` by
cycling layers, exactly per block, via two routes:

- **shared contexts** (deep linear, FGLN): the modewise filter
  `dW_ij = G_ij / (sigma_A,i^2 sigma_B,j^2 + lam)`, `O(L d^3)`.
- **per-sample contexts** (ReLU, CReLU): the dense sum-of-Kronecker normal equations,
  `O((n m)^3)`. No separable approximation is made -- that approximation *is* K-FAC, and
  keeping it out is what makes this a control.

Layers too large for the exact per-sample solve are refused rather than silently
downgraded; configure an `outer` optimizer to take them (this is how a 784-wide input
layer is handled on MNIST), and the split is recorded in the run's metadata.

Sweeps run downward, `k = L..1`, so the left contexts update by one matmul per layer and
the right contexts are a single prefix scan computed once -- `O(L)` matmuls per sweep
rather than `O(L^2)`. That is what makes depth 1024 reachable.

### Diagnostics, logged for every optimizer

| | what it answers |
|---|---|
| `cos_target` | did the realized `dP` go where `-eta G` asked? |
| `cos_gd_first_order` | Proposition 3.1's predicted distortion, from the contexts alone -- no step needed |
| `cos_linear` | is the first-order expansion still describing what happened? |
| `rho_k` | has this layer's context collapsed so far that the solve carries no geometry? |
| `singular_values`, `effective_rank` | is the operator spectrum being destroyed and rebuilt? |
| `gate_flip_rate`, `frozen_gate_error` | is the frozen-gate linearization still valid? |
| `step_seconds`, `peak_memory_mb` | what the exact solve actually costs |

## Findings that differ from the submitted paper

Recorded here because they are the reason for the rewrite, and each is pinned by a test in
`tests/`.

1. **The mismatch is second-order, and depth and step size collapse into one variable.**
   At an identity (or orthogonal) initialization every context is the identity, so the
   *first-order* term is `-L eta G` -- exactly parallel to `-G` at every depth. The
   realized step is still misaligned, because factors multiply rather than add:

   ```
   dP = (I - eta G)^L - I = sum_k C(L,k) (-eta G)^k        (exact, identity init)
   ```

   `G^2` is not parallel to `G`, so the mismatch is real, second order, and matrix-valued
   -- a scalar `G` would show none. The ratio of second to first order is
   `((L-1)/2) eta ||G||`, so depth and learning rate enter only through

   ```
   tau = L * eta * ||G||
   ```

   Measured: `tau = 0.15` gives cosine 0.999906 at `L=4`, and `tau = 0.12` gives 0.999901
   at `L=32` -- different depths, matched `tau`, same alignment. `tau = 4.7` gives 0.893.
   **The usable learning rate scales as `1/L`: the depth wall for gradient descent on a
   perfectly conditioned network is a step-size wall, not a conditioning one.** This also
   explains finding 2 -- a tuned baseline is one that has driven `tau` small.

   The closed form is in `olo/theory/deep_linear.py` and agrees with measured steps to
   machine precision (`tests/test_theory.py`). Corollary: the appendix claim that Haar
   initialization "reintroduces the Gram-matrix mismatch" is false as stated -- orthogonal
   layers compose to orthogonal contexts, so there is nothing to reintroduce at first
   order, and what does the damage is `tau`.

2. **Tuned baselines do far better than reported.** On the identity-init deep linear
   problem the paper reports baselines stalling above `3.5e-1`. Over a proper per-method
   learning-rate grid, SOAP reaches `6.6e-6` against ALS's `4.6e-6`, and Adam `4.7e-3`.
   The reported gap was largely a step-size artifact. `cos_target` still rank-orders the
   methods almost perfectly, which is the more interesting result.

3. **Solving the projection exactly does not imply improving the network.** At depth 128
   with an aggressive step, ALS drives its own residual to `~1e-8` while the loss diverges:
   the increments leave the pre-activation margins the gates were frozen at, so the
   objective stops describing the network. `gate_flip_rate` and `frozen_gate_error` make
   this visible instead of leaving it to look like a solver bug.

4. **The factorization bias can be beneficial.** On underdetermined matrix sensing,
   factored Adam reaches held-out MSE `4.8e-3` at effective rank 5.6, while projection --
   which removes the bias while holding parameterization and initialization fixed --
   tracks direct-`P` Adam to `3.9e-2` at effective rank 15.6. Correcting the mismatch is
   not automatically desirable, and this is the setting that shows when it is not.

## History

The previous version of this repository (the NeurIPS 2026 submission, its ~15 solver
variants, warm-starts, and one-off experiment scripts) is at the tag
`neurips26-submission`. The paper and reviews remain in `neurips26/`.
