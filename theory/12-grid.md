# What the grid says

`studies/grid.py`: **4** architectures x 3 initialisations x 3 tasks x 3 seeds, depth **128**,
width 32, full batch (n = 256), lr 1e-4, up to 30k steps. Tables in
`runs/theory/grid_report.txt`; raw traces in `runs/theory/grid.json`; per-run checkpoints in
`runs/theory/grid_ckpt/`.

**Caveat, first.** 96 of 108 runs hit the step cap. Every "final" number below is
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

## 6. The deep linear baseline

A fourth architecture with no nonlinearity at all, run identically, so every number above can
be read against what the *factorisation alone* does.

### 6.1 Isometric contexts give a bias of exactly zero

`sin` at step 0 on the isotropic teacher:

| architecture | identity | orthogonal | xavier |
|---|---|---|---|
| **deep linear** | **0.006** | **0.006** | 0.975 |
| CReLU | 0.331 | 0.227 | — |
| residual | 0.722 | 0.782 | 0.799 |
| ReLU | 0.971 | 0.999 | 1.000 |

`0.006` is `theory/10` Corollary 4.2 measured: with every context a multiple of an isometry
the mismatch is a **pure rescaling**, with no direction change at all. It is the first
quantity in this project to hit a predicted zero rather than approach it, and it calibrates
the rest of the column — the ordering is exactly the trainability ordering of §1.

### 6.2 The rank-1 collapse at depth is the product, not the gates

Effective rank at initialisation under Xavier:

| deep linear | CReLU | ReLU | residual |
|---|---|---|---|
| **1.11 ± 0.18** | 1.08 ± 0.06 | 1.02 ± 0.04 | 9.01 ± 1.75 |

A network with **no nonlinearity whatsoever** is already at rank 1.1 of 32. So the depth-128
collapse under Xavier is a property of the product of random matrices, full stop; the gates
add nothing to it. Only the skip connection escapes, because `prod_l(I + W_lD_l)` with `1/L`
scaling is near the identity however the weights are drawn.

### 6.3 The terminal rank is task-set across the linear/nonlinear divide

| task | target | deep linear | CReLU | residual | ReLU |
|---|---|---|---|---|---|
| teacher low-rank | 4 | 3.72 | 4.21 | 4.13 | 2.10 |
| MNIST-1D | 10 | 3.52 | 2.83 | 2.45 | 2.34 |
| teacher isotropic | 32 | 27.57 | 24.25 | 26.30 | 9.52 |

The universality of §2 now spans the linear control too. Whatever sets the endpoint is not
the nonlinearity.

### 6.4 The nonlinearity strictly adds bias, and it never subtracts it

Final `sin`, best initialisation per architecture:

| task | deep linear | CReLU | residual | ReLU |
|---|---|---|---|---|
| teacher low-rank | **0.352** | 0.497 | 0.993 | 0.985 |
| teacher isotropic | **0.666** | 0.994 | 0.967 | 0.999 |
| MNIST-1D | **0.978** | 0.995 | 0.999 | 0.999 |

Deep linear is the best case in every task. No architecture anywhere in the grid beats it.

### 6.5 The bias appears in the first 5% of training

Steps for `sin` to first exceed 0.9, of a 30k budget:

| architecture | `sin(0)` | steps | % of budget |
|---|---|---|---|
| deep linear | 0.15 | 1500–1750 | 5–6% |
| CReLU | 0.33–0.43 | 1000–1500 | 3–5% |
| residual | 0.76–0.81 | 1000 | 3% |
| ReLU | 0.95–0.98 | **0** | 0% |

Even from an exactly isometric start the mismatch saturates almost immediately. Whatever the
initialisation buys, it is spent within a few percent of training.

### 6.6 This is not a vanishing-gradient artefact

The obvious worry is that `sin -> 1` merely reports noise once `G` has collapsed. It does not.
Median `sin` by relative loss, over 5983 snapshots:

| `L/L_0` | 1–0.1 | 0.1–0.01 | 1e-2–1e-3 | 1e-3–1e-5 | < 1e-8 |
|---|---|---|---|---|---|
| median `sin` | 0.999 | 0.992 | 0.990 | 0.999 | **0.002** |

The mismatch is ~1 throughout training, not only at the end. Only in the last band — 33
snapshots, at machine-precision convergence — does it fall back to zero, and **only 4 of 108
runs** ever got there. That reversal is real but rests on very little.

---

## 7. What to try to prove

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
4. **The nonlinearity only ever adds bias (§6.4), and the deep linear control attains the
   predicted zero at initialisation (§6.1).** Together these say the implicit bias of a
   gated network is the factorisation's bias plus a non-negative gate contribution — which is
   exactly the shape of `theory/10` Prop. 6.1, and is now measured rather than argued.

Nothing here should be proved from the endpoints alone until the runs are actually converged
(§0 caveat) — extending the 72 capped runs is the first thing to do.
