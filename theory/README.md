# Theory

Proofs for the low-rank bias question in CReLU networks, written so each claim can be
checked. Every numbered result has a proof here and a numerical test in `tests/`; where a
result carries a hypothesis, the hypothesis is stated separately from the proof and comes
with the measurement that would confirm or refute it.

| file | contents | status |
|---|---|---|
| [`01-crelu-structure.md`](01-crelu-structure.md) | gate isometry, exact sign decomposition, uniform per-layer bound, modes | all unconditional |
| [`02-mode-ensemble.md`](02-mode-ensemble.md) | exact transfer operator for mode averages, positivity, stabilization | unconditional except Thm 9 (primitivity) |
| [`03-dynamics.md`](03-dynamics.md) | master equation for singular-value dynamics, self/cross split, what is ruled out, open hypotheses | reduction is unconditional; the bias is **not** proved |

## Reading order

Start with 01 (structure), then 02 (statics at frozen weights), then 03 (dynamics, and the
open problem). §3 of 03 lists four candidate hypotheses that were checked and failed; §4
lists the three that remain, each with the experiment that would settle it.

## Tests

```
pytest tests/test_theory_crelu.py      # file 01
pytest tests/test_theory_modes.py      # file 01 §4
pytest tests/test_theory_transfer.py   # file 02
pytest tests/test_theory_nonlinear.py  # file 03
pytest tests/test_theory_balanced.py   # closed-form mode gains (deep linear reference)
```

## The one-line summary

At fixed input, or at fixed mode, a CReLU network *is* a fixed-gates linear network, so the
existing analysis applies unchanged. Everything genuinely new sits in the cross-input
coupling of the weight gradient — one measurable object. That is a reduction, not a proof of
the bias.
