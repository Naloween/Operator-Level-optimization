"""Exact modewise gain of the gradient step on factors, for balanced deep networks.

This is the closed form behind "why does the operator concentrate but never fall apart".

Setup. Let `P = U S V^T` and let the factors be balanced -- each carrying singular values
`s^{1/L}` in aligned bases, which is exactly what a looks-linear or identity initialization
gives, and which gradient flow conserves (`W_{l+1}^T W_{l+1} - W_l W_l^T` is an invariant,
Du et al. 2018). Then

    A_k = W_L...W_{k+1}  has singular values  s^{(L-k)/L}
    B_k = W_{k-1}...W_1  has singular values  s^{(k-1)/L}

so Proposition 3.1's induced step, `dP_GD = -eta sum_k A_k A_k^T G B_k^T B_k`, is diagonal
in the (U, V) bases and its mode (i, j) is multiplied by

    c_ij = sum_{k=1}^{L} s_i^{2(L-k)/L} s_j^{2(k-1)/L}
         = (s_i^2 - s_j^2) / (s_i^{2/L} - s_j^{2/L})            [geometric series]
    c_ii = L * s_i^{2 - 2/L}                                     [the i = j limit]

So the weight-space step realizes `-eta c_ij G_ij` where operator-space descent asks for
`-eta G_ij`. Every difference between the two is in `c`. ALS sets `c_ij = 1` by
construction: that is what solving the projection means.

Three consequences, and they answer three separate questions:

1. **No mismatch at an isometry.** At `s = 1` every `c_ij = L`, so the realized step is
   exactly `L` times the desired one: right direction, `L` times too long. This is why the
   first-order alignment is exactly 1 at looks-linear init at any depth, and why the
   usable learning rate goes as `1/L`.

2. **The low-rank bias, and why it saturates.** On the diagonal the gain is
   `L s^{2 - 2/L}`, so relative to a unit direction a mode of size `s` is amplified by
   `s^{2-2/L}` -- the rich-get-richer exponent. It is 0 at `L = 1` (no bias: a linear model),
   1 at `L = 2`, and rises to **2** as `L -> infinity`. It is 1.75 at L=8, 1.94 at L=32,
   1.992 at L=256, 1.998 at L=1024. **The bias saturates.** There is no depth at which it
   suddenly becomes catastrophic, and the difference between depth 256 and depth 1024 is
   under a percent -- which is why pushing depth further does not produce a wall.

3. **Why the operator never collapses entirely.** `s = 0` makes the gain zero, so a mode
   that reaches zero is a fixed point and cannot come back -- collapse is one-way, which is
   why effective rank falls monotonically. But the same `s^{2-2/L}` that suppresses small
   modes *protects the largest one*: the top mode has the largest gain, the loss actively
   drives it, and its growth self-limits only as the residual shrinks. So the spectrum
   concentrates onto the few directions the loss needs rather than vanishing or exploding.
"""
from __future__ import annotations

import numpy as np


def bias_exponent(depth: int) -> float:
    """`2 - 2/L`: how strongly a mode's own size amplifies its growth.

    0 at depth 1 (a linear model has no such bias), 1 at depth 2, and saturating at 2.
    """
    return 2.0 - 2.0 / depth


def mode_gain(s_i: float, s_j: float, depth: int) -> float:
    """`c_ij`: the factor by which weight-space GD scales operator mode (i, j).

    Operator-space descent wants 1. Deviation from a constant across modes is the
    direction mismatch; deviation from 1 overall is a step-size rescaling.
    """
    if depth == 1:
        return 1.0
    a, b = s_i ** (2.0 / depth), s_j ** (2.0 / depth)
    if np.isclose(a, b):
        return depth * s_i ** bias_exponent(depth)
    return (s_i**2 - s_j**2) / (a - b)


def gain_matrix(s: np.ndarray, depth: int) -> np.ndarray:
    """`c_ij` for every mode pair of a spectrum."""
    s = np.asarray(s, dtype=float)
    return np.array([[mode_gain(si, sj, depth) for sj in s] for si in s])


def induced_step(s: np.ndarray, G_modal: np.ndarray, depth: int, eta: float) -> np.ndarray:
    """The operator increment weight-space GD actually produces, in the (U, V) basis."""
    return -eta * gain_matrix(s, depth) * np.asarray(G_modal, dtype=float)


def alignment(s: np.ndarray, G_modal: np.ndarray, depth: int) -> float:
    """cos(realized dP, -eta G): 1 exactly when `c` is constant across the modes present."""
    c = gain_matrix(s, depth) * np.asarray(G_modal, dtype=float)
    g = np.asarray(G_modal, dtype=float)
    denom = np.linalg.norm(c) * np.linalg.norm(g)
    return float((c * g).sum() / denom) if denom > 0 else float("nan")


def condition_amplification(s: np.ndarray, depth: int) -> float:
    """Ratio of the largest to smallest diagonal gain: how unevenly modes are driven.

    `(s_max / s_min)^(2 - 2/L)` -- the per-step factor by which an already-anisotropic
    spectrum is made more so.
    """
    s = np.asarray(s, dtype=float)
    lo = s[s > 0].min() if np.any(s > 0) else 0.0
    return float((s.max() / lo) ** bias_exponent(depth)) if lo > 0 else float("inf")
