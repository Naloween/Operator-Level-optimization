# Theory

Proofs for the low-rank bias question in CReLU networks, written so each claim can be
checked. Every numbered result has a proof here and a numerical test in `tests/`; where a
result carries a hypothesis, the hypothesis is stated separately from the proof and comes
with the measurement that would confirm or refute it.

| file | contents | status |
|---|---|---|
| [`01-crelu-structure.md`](01-crelu-structure.md) | gate isometry, exact sign decomposition, uniform per-layer bound, modes | all unconditional |
| [`02-mode-ensemble.md`](02-mode-ensemble.md) | exact transfer operator for mode averages, positivity, stabilization | unconditional except Thm 9 (primitivity) |
| [`03-dynamics.md`](03-dynamics.md) | master equation for singular-value dynamics, self/cross split, four hypotheses ruled out | reduction is unconditional |
| [`04-instability.md`](04-instability.md) | drive vs feedback, the rate `psi = (1-2/L)+p`, growth as the clock, where the CReLU seed comes from, exact invariance under symmetrized batches | §1 and §4 unconditional; §2–§3 need (2.1)–(2.2) |
| [`05-imbalance.md`](05-imbalance.md) | balancedness weakened to mode-independence; `K >= L`; what random init actually does to the exponent | unconditional; the modal reduction still assumed |
| [`06-alignment.md`](06-alignment.md) | the modal reduction measured: what it costs, when separation helps, and the task exponent on real data | measurement only; no new theorems |

## Reading order

01 (structure) → 02 (statics at frozen weights) → 03 (dynamics, and what failed) →
04 (the mechanism). A reader who wants only the answer can start at 04: it is
self-contained apart from Corollary 2.1 and Lemma 2 of file 01.

## The one-line summary

The low-rank bias is two mechanisms, not one. A **drive** — gradient anisotropy across modes
— manufactures separation out of an exact isometry and is present even at depth 2, where the
feedback is identically zero. A **feedback** of rate `psi = (1 - 2/L) + p` multiplies
whatever separation exists, with operator *growth* as its clock rather than training steps,
and with depth's share saturating at 1. Neither is universal: an isotropic force at an
isometry produces no bias at any depth, and a task with `p = -(1 - 2/L)` cancels the depth
bias exactly.

For CReLU specifically, a looks-linear network starts on the linear manifold and the only
thing that can move it off is the correlation between the operator residual and the gate
sign pattern (Theorem 16). That correlation is a `B^{-1/2}` fluctuation under sign-symmetric
data, is systematic under asymmetric data such as MNIST's non-negative pixels, and is
**exactly zero** when the batch is closed under negation — in which case the network remains
a deep linear network for all time (Theorem 17), so the reduction of file 03 becomes exact
rather than hypothetical.

## Measured, in one table

| claim | prediction | measured |
|---|---|---|
| amplification exponent | `psi = (1-2/L) + p` | 40/40 cells, `L = 2..256`, `p = -1..1`, worst error 0.0087 |
| whole spectrum from one scalar | Cor. 14.1 | max log error `1.4e-11` to `7.9e-3` |
| seed is multiplicative | `r(t) ∝ r(0)` | ratio constant to 3 digits over 5 decades of `r(0)` |
| sign of the effect | Cor. 13.2 | 40/40 correct, including `psi = 0` |
| CReLU seed, symmetric data | `B^{-1/2}` | slope −0.507 ± 0.010 (seed), −0.48 (realized separation) |
| CReLU seed, MNIST | systematic | slope −0.032 ± 0.036, ratio 0.83 at `B = 2048` |
| symmetrized batch | `Delta = 0` exactly | `2.2e-16` at every depth, both input laws |
| direction of every separation change | Lemma 12: `sign(dr) = sign(b)` | 100% where the form fits (`R^2 >= 0.9`), 89% overall |
| Xavier vs looks-linear, all else equal | seed decides | eff. rank 1.50 vs 7.04 at `L = 16` |
| mode-independent imbalance | Thm 19: exponent unchanged | exact to 2e-15 at `K/L = 24` |
| random init | mode-dependent, weakens the bias | `d log K/d log s < 0` in every cell tested |
| modal reduction cost | assumed free in file 04 | 3% from looks-linear, **43% from Xavier** |
| separation buys alignment (Thm 6.1) | error falls with separation | holds above a threshold that grows with depth |
| task exponent on MNIST | unknown | `p ≈ +11` from an isometric start: the task *reinforces* the bias |

## Tests and studies

```
pytest tests/test_theory_crelu.py        # file 01
pytest tests/test_theory_modes.py        # file 01 §4
pytest tests/test_theory_transfer.py     # file 02
pytest tests/test_theory_nonlinear.py    # file 03
pytest tests/test_theory_instability.py  # file 04
pytest tests/test_theory_imbalance.py    # file 05
pytest tests/test_theory_alignment.py    # file 06
pytest tests/test_theory_balanced.py     # closed-form mode gains (deep linear reference)

python studies/forcing.py       # the rate law, with the operator force prescribed
python studies/seed_source.py   # fluctuation vs systematic seed, across tasks
python studies/symmetrize.py    # switching the nonlinearity off exactly
python studies/report.py        # regenerates every table in file 04 section 5
python studies/imbalance.py     # dropping balancedness: what the exponent does
python studies/alignment.py     # what the modal reduction costs, and p on real data
```
