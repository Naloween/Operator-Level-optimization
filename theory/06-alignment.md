# Does separation buy alignment? Yes — above a threshold that grows with depth

Builds on [`04-instability.md`](04-instability.md) §2 and [`05-imbalance.md`](05-imbalance.md).
This file tests the one hypothesis those rest on, and it is a measurement file: there are no
new theorems here.

---

## 1. The hypothesis, stated as a number

Everything in file 04 §2–§3 replaces the exact per-mode velocity

$$\dot s_k \;=\; -\sum_\ell u_k^\top A_\ell A_\ell^\top\, G\, B_\ell^\top B_\ell\, v_k \tag{exact}$$

by its **diagonal** part `-c_k g_k`, with `c_k = \sum_\ell \|A_\ell^\top u_k\|^2 \|B_\ell v_k\|^2`
and `g_k = u_k^\top G v_k`. The two agree exactly when `A_\ell A_\ell^\top` and
`B_\ell^\top B_\ell` are diagonal in the operator's own singular bases — each subproduct
aligned with the whole product. That is Theorem 6.1 of Haas et al. (ICML 2026), proved for
fixed-gates linear networks and conditional on the subproduct being large enough.

Three measurable quantities (`olo.theory.alignment`), none of which assumes anything:

* **`reduction_error`** `= \|\text{exact} - \text{diagonal}\| / \|\text{exact}\|` — what the
  replacement actually costs. This is the quantity that matters; alignment only matters
  through it.
* **`alignment`** — worst per-layer `\|\mathrm{diag}(M)\| / \|M\|_F` for
  `M = U^\top A_\ell A_\ell^\top U` and `V^\top B_\ell^\top B_\ell V`. 1 iff diagonal.
* **`separation`** `= \log(s_1/s_d)` — the variable the theorem says drives alignment.

One structural fact worth recording, because it explains why the reduction is *ever* exact:

> In an aligned network an off-diagonal gradient moves the singular *vectors* and not the
> singular *values*, so the diagonal reduction discards exactly nothing. Verified: a purely
> off-diagonal `G` gives `exact = 0` and `diagonal = 0` alike.

---

## 2. The controlled test

Training cannot answer the question, because separation, misalignment and training time all
move together — any correlation between two of them is uninterpretable. So separation and
misalignment are set independently:

$$W_\ell \;=\; R(\theta)\,\mathrm{diag}\!\big(e^{a/L}\big), \qquad R(\theta) = \exp\!\Big(\tfrac{\theta}{\sqrt d}(A - A^\top)\Big),$$

with the spread of `a` fixing `r(0)` and `\theta` fixing how far each subproduct is rotated
out of the operator's basis. Reduction error, `L = 8`, `d = 12`:

| `\theta` | `r0=0.01` | `0.1` | `0.5` | `2.0` | `8.0` | `20.0` |
|---|---|---|---|---|---|---|
| 0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.05 | 0.0003 | 0.0029 | 0.0128 | **0.0301** | 0.0055 | 0.0004 |
| 0.20 | 0.0010 | 0.0093 | 0.0412 | **0.1037** | 0.0199 | 0.0023 |
| 0.50 | 0.0013 | 0.0131 | 0.0613 | **0.1979** | 0.0877 | 0.0135 |
| 1.00 | 0.0014 | 0.0138 | 0.0682 | 0.2595 | **0.3773** | 0.1027 |

**The error is non-monotone in separation.** It rises to a peak and then falls, so:

* **above the peak, Theorem 6.1's prediction holds** — more separation makes the modal
  reduction better, and at `\theta = 0.2` the error falls by a factor of 45 from `r0 = 2` to
  `r0 = 20`. The asymptotic content of the theorem transfers.
* **below the peak it is reversed** — more separation makes the reduction *worse*. A network
  with a nearly isometric spectrum is aligned for a trivial reason (there is no preferred
  basis to be misaligned with), and the first anisotropy breaks that before the theorem's
  mechanism can take over.

The peak is the theorem's "large enough" condition, made visible. It moves right with both
misalignment and depth:

| `L` | `\theta=0.05` | `0.2` | `0.5` | `1.0` |
|---|---|---|---|---|
| 2 | 2 | 2 | 2 | 8 |
| 8 | 2 | 2 | 2 | 8 |
| 16 | 2 | 2 | 8 | 20 |
| 32 | 2 | 2 | 20 | 20 |

(`r0` at which the error peaks.) **The separation needed before alignment starts helping
grows with depth.** That is a caveat the original theorem's setting does not surface, and it
matters here because deep networks are exactly where the reduction is being relied on.

---

## 3. What trained networks actually do

`studies/alignment.py`: 90 runs, 1890 snapshots, two models × two inits × two tasks ×
depths 2–32 × three seeds, `diag_batch = 1` so that cross-input coupling is excluded and the
only error source is misalignment.

**Pooled, the correlation has the wrong sign** — rank correlation of `reduction_error` with
`separation` is **+0.741** (n = 1860, separation spanning 0.07–45.4), and 4 of 6
model×init×task cells are individually positive. Read naively that refutes Theorem 6.1.

It does not, and the reason is the confound §2 was built to remove: looks-linear runs have
both low separation and low error, Xavier runs have both high separation and high error, so
the pooled correlation is reporting the initialization. Within a run, separation and
misalignment grow together, so that correlation reports training time. **Neither is a test of
the theorem.** The controlled sweep is, and it supports it above threshold.

What the training runs *do* establish is the size of the error, which is what file 04 needs:

| init | n | median error | 90th pct | median alignment |
|---|---|---|---|---|
| looks-linear | 630 | **0.029** | 0.074 | 0.934 |
| Xavier | 1260 | **0.425** | 1.014 | 0.790 |

**The modal reduction costs 3% from a looks-linear start and 43% from Xavier.** So file 04
§2–§3 is a decent approximation in the looks-linear regime and badly violated in the Xavier
one — which is precisely the regime where the depth wall occurs. Any claim that the rate law
explains the depth wall has to reckon with a 43% error in the step that derives it. This is
the sharpest available limit on how far the framework reaches.

---

## 4. The task exponent on real data

Measured by Proposition 15 over snapshots where the rich-get-richer form fits at all
(`R^2 >= 0.1`):

| task | init | model | n | median `p` | IQR | |
|---|---|---|---|---|---|---|
| **MNIST** | **looks-linear** | CReLU | 158 | **+10.95** | [6.02, 20.54] | **reinforces** |
| MNIST | Xavier | CReLU | 150 | −1.03 | [−1.96, 1.09] | opposes |
| MNIST | Xavier | deep linear | 113 | −1.31 | [−2.91, 0.22] | opposes |
| teacher–student | looks-linear | CReLU | 230 | −4.74 | [−8.11, 0.92] | opposes |
| teacher–student | Xavier | CReLU | 168 | −1.40 | [−2.06, −1.11] | opposes |
| teacher–student | Xavier | deep linear | 160 | −1.33 | [−1.76, −1.15] | opposes |

**From an isometric start on real data, the task strongly reinforces the low-rank bias**
(`p ≈ +11`, so `psi = phi + p` is an order of magnitude above the architectural part), while
a well-conditioned synthetic target opposes it at every init. The mechanism is not subtle:
ten MNIST classes through width 16 make the operator gradient effectively rank ≤ 10, so it
pushes hardest on exactly the directions that are already largest.

This closes a loop with the earlier depth sweeps, where CReLU/looks-linear reached 0.935 at
`L = 256` and 0.924 at `L = 1024` on MNIST *despite* the operator's effective rank falling
from 9.98 to 3.17. **The bias was not being resisted; it was doing what the task wanted.** A
depth-robust architecture and a strong low-rank bias are not in conflict when `p > 0`.

Note the init dependence: the same task gives `p ≈ +11` from looks-linear and `p ≈ −1` from
Xavier. `p` is not a property of the task alone — it is a property of how the operator's
current spectrum sits relative to what the loss wants, which is why Proposition 15 measures
it rather than tabulating it.

---

## 5. Status

| Claim | Status |
|---|---|
| off-diagonal `G` moves vectors, not values | exact, tested |
| reduction error is non-monotone in separation | measured, controlled sweep |
| above threshold, separation improves the reduction (Thm 6.1) | **supported** |
| the threshold grows with depth and misalignment | measured |
| pooled training correlation is confounded | shown, not a refutation |
| reduction error: 3% (looks-linear) vs 43% (Xavier) | measured, n = 1890 |
| `p > 0` on MNIST from an isometric start | measured, n = 158 |

**Not established.** That the controlled sweep's threshold is the same object as Theorem
6.1's subproduct condition — the shapes agree, the identification is not proved. That a 43%
reduction error invalidates file 04's conclusions rather than merely bounding them; the
error's *direction* may still be right, and `cosine` is recorded for that but not analysed.
Widths ≤ 16, depths ≤ 32, two task families throughout.
