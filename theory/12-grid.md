# What the grid says

`studies/grid.py`: 3 architectures x 3 initialisations x 3 tasks x 3 seeds, depth **128**,
width 32, full batch (n = 256), lr 1e-4, up to 30k steps. Tables in
`runs/theory/grid_report.txt`; raw traces in `runs/theory/grid.json`; per-run checkpoints in
`runs/theory/grid_ckpt/`.

**Caveat, first.** 72 of 81 runs hit the step cap. Every "final" number below is
end-of-budget, not end-of-training. Checkpoints exist, so extending is one command.

---

## 1. Trainability at depth is decided by the initialisation, and skip connections make it
## initialisation-proof

Cells that train (final loss below half the initial), out of 9:

| | identity | orthogonal | xavier |
|---|---|---|---|
| ReLU MLP | **9/9** | 1/9 | 0/9 |
| CReLU MLP | **9/9** | **9/9** | 0/9 |
| residual ReLU MLP | **9/9** | **9/9** | **9/9** |

A strict ordering, and the operator explains it. At depth 128 the Jacobian's effective rank
*at initialisation* is 1.0 for ReLU/xavier, 1.1 for CReLU/xavier, and 10.2 for
residual/xavier. Nothing trains from rank 1. The residual net starts at rank 10 because
`prod_l (I + W_l D_l)` with `1/L` branch scaling stays near the identity whatever the weights
are — which is what a skip connection is for, here measured on the operator rather than
inferred from accuracy.

CReLU sits between: its gate is an isometry, so looks-linear initialisation gives an exactly
orthogonal operator, but Xavier weights still destroy it.

---

## 2. The final effective rank is set by the **task**, not by the architecture or the init

Final `PR` over every cell that trains:

| task | target rank | final PR (mean ± sd) | n |
|---|---|---|---|
| teacher, low rank | 4 | **3.73 ± 0.88** | 19 |
| MNIST-1D | 10 classes | **2.56 ± 0.37** | 18 |
| teacher, isotropic | 32 | 22.82 ± 6.61 | 18 |

Nineteen cells spanning three architectures, two-to-three initialisations and three seeds land
on `3.73 ± 0.88` for a rank-4 target. The architecture and the starting point change *whether*
a network trains; they do not change *where its spectrum ends up* once it does.

Note this is not "PR equals the target's rank": MNIST-1D has ten classes and the operator
settles at 2.6. It is the rank the *loss* needs, which at ~10^-3 cross-entropy is well below
the class count.

---

## 3. The collapse is non-monotone; the minimum is not the endpoint

**20 of the 55 training cells** recover their effective rank by more than 10% after a dip —
e.g. CReLU/orthogonal on the isotropic teacher goes `32.0 -> 21.8 -> 27.4`, and ReLU/identity
`15.4 -> 4.8 -> 8.2`. Reading start-and-end alone would report a collapse that partly
reverses. Any measurement of "the low-rank bias" from endpoints is therefore an
overstatement of a quantity that is not monotone in the first place.

---

## 4. The factorised update loses its direction almost everywhere — with one exception

`sin(Delta J, -G)` is the scale-free mismatch: 0 means the realised operator step reproduces
operator-space descent up to a step size, 1 means it is orthogonal to it. It ends at
**0.97–1.00 in 49 of the 55 training cells**.

The exception is sharp and reproducible:

| arch | init | task | sin | scale | PR |
|---|---|---|---|---|---|
| CReLU | orthogonal | teacher low-rank | 0.483–0.510 | 0.221–0.223 | 4.21–4.22 |
| CReLU | identity | teacher low-rank | 0.557–0.577 | 0.229–0.235 | 4.21–4.22 |

Six cells, three seeds each of two initialisations, all at `sin ~ 0.5` while everything else is
at 1.0. The common factor: the target is genuinely low-rank *and* the architecture keeps its
contexts isometric. That is the one configuration in the grid where the factorisation still
transmits half the operator gradient's direction.

Note also the scale: 0.22, so the realised step is 4.5x **shorter** than ideal there, while
elsewhere it is 10^3–10^7 times **longer**. Long and orthogonal is the generic behaviour.

---

## 5. The conditioning-to-bias feedback appears only where the operator stays conditioned

Rank correlation of `sin` with spectral separation along each trajectory:

| | identity | orthogonal | xavier |
|---|---|---|---|
| residual, MNIST-1D | **+0.86** | **+0.62** | +0.35 |
| CReLU (all tasks) | +0.25…+0.34 | +0.28…+0.40 | — |
| ReLU (all tasks) | −0.11…+0.23 | −0.09…+0.28 | +0.03…+0.20 |

`theory/10` Cor. 5.2 predicts a positive arrow from conditioning to bias. It is clearly
present in the residual net, moderately in CReLU, and absent in ReLU — i.e. present exactly
where the contexts are well conditioned enough for the quantity to mean anything.

---

## 6. What to try to prove

In order of how sharply the data pins them:

1. **Trainability at depth is a statement about the operator's rank at initialisation.** The
   3x3 table above is a clean monotone ordering and each row has a structural explanation
   (`I + W D` near identity; CReLU gate an isometry; ReLU gate a projector). `theory/10` §6
   already contains most of the pieces.
2. **The terminal rank is task-determined.** This is the most surprising result and the least
   explained by anything in `theory/`. A proof would have to show the spectrum's fixed point
   depends on the loss and not on the factorisation — which is close to saying the implicit
   bias has no effect on the *endpoint*, only on the path. That would be a substantial claim.
3. **`sin -> 1` generically, and the low-rank exception.** The exception is the informative
   half: it says direction is preserved when the target's rank matches what the isometric
   architecture can hold.

Nothing here should be proved from the endpoints alone until the runs are actually converged
(§0 caveat) — extending the 72 capped runs is the first thing to do.
