"""Balancedness is not needed. What matters is whether the imbalance depends on the mode.

`olo.theory.instability` derives the amplification exponent from the *balanced* deep-linear
flow -- every layer carrying `s^{1/L}` in aligned bases. That hypothesis is unrealistic, it
is not what a Xavier network satisfies, and a rate law resting on it would say nothing about
the regime where the depth wall actually happens. This module removes it, and the answer is
sharper than "the exponent degrades".

**The exact decomposition.** The mode gain of Proposition 3.1 is
`c_k = sum_l prod_{j != l} a_{j,k}^2 = s_k^2 sum_l a_{l,k}^{-2}`. Factor out the balanced
part by writing `a_{l,k} = s_k^{1/L} x_{l,k}^{-1/2}`:

    c_k = s_k^{2 - 2/L} K_k,      K_k := sum_l x_{l,k} = sum_l (a_{l,k} s_k^{-1/L})^{-2}.

`K_k` is a pure, scale-free imbalance factor -- the layer scales measured against the
balanced reference `s^{1/L}`. Since `prod_l x_{l,k} = 1` by construction, AM-GM gives

    K_k >= L,   with equality exactly at balance.

The **cross-mode bias exponent** -- the one that governs separation, because Lemma 12's rate
is a regression *across modes* at fixed time -- is therefore

    d log c / d log s  =  (2 - 2/L)  +  d log K / d log s.

Verified to machine precision (`|err| <= 1.3e-15`) including at extreme imbalance.

**The consequence that matters, and it is not what one expects.** A layerwise imbalance that
is the *same for every mode* makes `K` constant across modes, so `d log K/d log s = 0` and
**the balanced exponent survives untouched**. Measured: a 20x bottleneck layer, or a
smooth 4-nat spread of layer scales, leaves `d log c/d log s` at exactly `2 - 2/L`. So

* balancedness was never the real hypothesis -- *mode-independence of the imbalance* is, and
  it is a far weaker and more plausible condition;
* only imbalance that varies with the mode moves the exponent, and then by exactly
  `d log K/d log s`, which can be large and can flip the sign (measured `-1.95` against a
  balanced `1.75` when the bottleneck sits on the largest modes).

Xavier initialization is mode-dependent, and its measured exponent sits *below* the balanced
value at every depth (1.18 vs 1.50 at `L = 4`, 1.80 vs 1.94 at `L = 32`) -- a weaker bias
than the balanced theory predicts, in the regime where collapse actually happens.

**A separate effect of the level.** `K >= L` means an unbalanced network takes *larger*
operator steps for the same gradient: imbalance speeds the dynamics up while leaving the
bias exponent alone. Rate and bias are independent knobs.

**Measuring it with no alignment assumption.** The per-layer contribution to the mode gain is
exactly `q_{l,k} = ||A_l^T u_k||^2 ||B_l v_k||^2`, the `(k,k)` entry of the induced step under
`G = u_k v_k^T`. So `c_k = sum_l q_{l,k}` and `K_k = c_k s_k^{2/L - 2}` are computable in any
network, and the exponent is read off by regression. Only the interpretation of `K` in terms
of layer scales needs the aligned diagonal picture.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def imbalance_factor_from_scales(a: np.ndarray) -> float:
    """`K = sum_l (a_l s^{-1/L})^{-2}` for one mode, from its layer scales. `K >= L`."""
    a = np.clip(np.asarray(a, dtype=float), 1e-300, None)
    s = float(np.prod(a))
    return float(((a * s ** (-1.0 / a.size)) ** -2).sum())


def balanced_exponent(depth: int) -> float:
    """`2 - 2/L`: the cross-mode exponent of the mode gain when `K` does not vary."""
    return 2.0 - 2.0 / depth


def trajectory_exponent_from_scales(a: np.ndarray) -> float:
    """`2 - 2/L_eff`, `L_eff = H^2/H_2` -- a *different* derivative, kept to avoid confusion.

    Under gradient flow the invariant `W_{l+1}^T W_{l+1} - W_l W_l^T` is conserved, which
    pins a mode's layer scales to `a_l^2 = t + C_l` and makes it a one-dimensional system.
    Differentiating the gain along *that* path gives `2 - 2 H_2/H^2` with `H = sum a_l^-2`,
    `H_2 = sum a_l^-4`, i.e. `L` replaced by the participation ratio `L_eff = H^2/H_2`.

    This is correct, and it is **not** the bias exponent. It describes how one mode's own
    gain changes as that mode grows over time; separation is governed by the comparison
    *between* modes at fixed time, which is `balanced_exponent` plus the mode-dependence of
    `K`. Reported for completeness because the two are easy to conflate -- I did conflate
    them, and the numbers above are what caught it.
    """
    x = np.clip(np.asarray(a, dtype=float), 1e-300, None) ** -2
    return 2.0 - 2.0 * float((x**2).sum()) / float(x.sum() ** 2)


def imbalance_invariants(net) -> list[torch.Tensor]:
    """`C_l = W_{l+1}^T W_{l+1} - W_l W_l^T`, conserved by gradient flow; zero iff balanced.

    Boundaries whose shapes disagree (a non-square first or last layer) are skipped.
    """
    out, Ws = [], net.weights
    for l in range(len(Ws) - 1):
        A, B = Ws[l + 1], Ws[l]
        if A.shape[1] == B.shape[0]:
            out.append((A.T @ A - B @ B.T).detach())
    return out


def imbalance_norm(net) -> float:
    """`max_l ||C_l||_F / ||W_l||_F^2`: a scale-free summary. 0 iff balanced.

    **nan when no boundary is comparable**, which is the case for every CReLU network:
    `W_l` there is `d x 2d`, so `W_{l+1}^T W_{l+1}` and `W_l W_l^T` have different shapes
    and Du's invariant does not apply as written (the gate sits between them, and it is not
    a fixed matrix). Returning 0.0 in that case would report "perfectly balanced" for a
    network the quantity says nothing about.
    """
    Ws, vals = net.weights, []
    for l, C in enumerate(imbalance_invariants(net)):
        vals.append(float(C.norm()) / max(float(Ws[l].detach().norm() ** 2), 1e-300))
    return max(vals) if vals else float("nan")


@dataclass
class ModeGain:
    """Per-mode decomposition of the induced step's diagonal gain, measured."""

    s: np.ndarray            # singular values
    q: np.ndarray            # (modes, layers) per-layer contributions
    L: int

    @property
    def gain(self) -> np.ndarray:
        """`c_k = sum_l q_{l,k}`: what weight-space GD multiplies mode k's gradient by."""
        return self.q.sum(axis=1)

    @property
    def imbalance_factor(self) -> np.ndarray:
        """`K_k = c_k s_k^{2/L - 2}`. Equal to `L` at balance, `>= L` always."""
        s = np.clip(self.s, 1e-300, None)
        return self.gain * s ** (2.0 / self.L - 2.0)

    @property
    def balanced_gain(self) -> np.ndarray:
        return self.L * self.s ** balanced_exponent(self.L)

    def exponent(self, floor: float = 1e-10) -> tuple[float, float]:
        """(measured `d log c/d log s`, its balanced-law residual `d log K/d log s`).

        Both by least squares across the surviving modes. The second is the entire
        correction: it is 0 exactly when the imbalance does not depend on the mode.
        """
        m = self.s > floor
        if int(m.sum()) < 3:
            return float("nan"), float("nan")
        ls = np.log(self.s[m])
        # An isometric spectrum has no spread in log s, so no slope is identifiable. That
        # is exactly the looks-linear case (J orthogonal, every s_k = 1); without this the
        # regression happily fits rounding noise and returns a plausible-looking number.
        if float(((ls - ls.mean()) ** 2).sum()) < 1e-12:
            return float("nan"), float("nan")
        fit = lambda y: float(np.polyfit(ls, y, 1)[0])
        return fit(np.log(np.clip(self.gain[m], 1e-300, None))), fit(
            np.log(np.clip(self.imbalance_factor[m], 1e-300, None))
        )


@torch.no_grad()
def mode_gains(net, x: torch.Tensor, sample: int = 0) -> ModeGain:
    """Measure `q_{l,k} = ||A_l^T u_k||^2 ||B_l v_k||^2` -- no alignment assumed.

    Exactly the `(k,k)` entry of the induced operator step under `G = u_k v_k^T`, so it is
    the mode gain whatever the network's imbalance or basis structure.
    """
    gates = net.gates(x)
    J = net.operator(x, gates)
    J_k = (J[sample] if J.shape[0] > 1 else J[0]).double()
    U, S, Vh = torch.linalg.svd(J_k)
    ctx = net.all_contexts(x, gates)
    pick = lambda T: (T[sample] if T.shape[0] > 1 else T[0]).double()

    k = S.shape[0]
    q = np.zeros((k, net.depth))
    for l in range(net.depth):
        A, B = pick(ctx[l][0]), pick(ctx[l][1])
        q[:, l] = ((A.T @ U).pow(2).sum(0)[:k] * (B @ Vh.T).pow(2).sum(0)[:k]).numpy()
    return ModeGain(s=S.numpy(), q=q, L=net.depth)


# -- a rigorous certificate for the gain exponent ---------------------------


def slope_bound(x: np.ndarray, lo: float, hi: float) -> float:
    """Lemma A: `|least-squares slope| <= (hi - lo) / (2 sd(x))` when every `y` is in [lo,hi].

    Proof. Let `xc = x - mean(x)`, so `<xc, 1> = 0` and hence `<xc, y> = <xc, y - c1>` for
    every constant `c`. Take `c` the midpoint of the band, so `|y_i - c| <= w/2` with
    `w = hi - lo`. Cauchy-Schwarz gives

        |b| = |<xc, y - c1>| / ||xc||^2 <= ||y - c1|| / ||xc|| <= (w/2) sqrt(n) / ||xc||,

    and `||xc|| = sqrt(n) sd(x)`. The midpoint is what buys the factor 2; centring `y` on
    its own mean instead would only give `w / sd(x)`.
    """
    sd = float(np.std(np.asarray(x, dtype=float)))
    return (hi - lo) / (2.0 * sd) if sd > 0 else float("inf")


def gain_floor(s: np.ndarray, weight_norms, depth: int) -> np.ndarray:
    """Lemma B: `c_k >= L s_k^2 g^{-2/L}` with `g = prod_l ||W_l||_2`. No hypotheses.

    Proof. `s_k = u_k^T A_l W_l B_l v_k = <A_l^T u_k, W_l B_l v_k>`, so Cauchy-Schwarz gives
    `s_k <= ||A_l^T u_k|| ||W_l|| ||B_l v_k||`, i.e. `q_{l,k} >= s_k^2 / ||W_l||^2` for every
    layer. AM-GM on `c_k = sum_l q_{l,k}` then gives
    `c_k >= L (prod_l q_{l,k})^{1/L} >= L s_k^2 g^{-2/L}`.

    Tight: an orthogonal chain has every `q_{l,k} = 1`, `g = 1`, `c_k = L`.
    """
    g = float(np.prod(np.asarray(weight_norms, dtype=float)))
    return depth * np.asarray(s, dtype=float) ** 2 * g ** (-2.0 / depth)


@dataclass
class GainCertificate:
    """A rigorous bound on how far the gain exponent can be from `2 - 2/L`."""

    exponent: float          # the measured d log c / d log s
    deviation: float         # |exponent - (2 - 2/L)|
    bound: float             # the certified bound on that deviation
    band: float              # width of the band containing log K
    sd_log_s: float          # spread of the log-spectrum: the bound's denominator
    floor_is_proved: bool    # whether the band's lower end came from Lemma B

    @property
    def holds(self) -> bool:
        return self.deviation <= self.bound + 1e-9

    @property
    def tightness(self) -> float:
        return self.deviation / self.bound if self.bound > 0 else float("nan")


def gain_certificate(s: np.ndarray, c: np.ndarray, depth: int,
                     weight_norms=None, floor: float = 1e-11) -> GainCertificate:
    """Certify `|d log c/d log s - (2 - 2/L)|` from the measured spectrum and gain.

    `K_k := c_k s_k^{2/L - 2}` by definition, so the exponent is exactly
    `(2 - 2/L) + d log K/d log s` and Lemma A bounds the second term by the width of any
    band containing `log K`, over twice the spread of `log s`.

    With `weight_norms` supplied the band's lower end is Lemma B's *proved* floor, making
    the whole statement a priori except for the single measured number `max_k K_k`. Without
    them the observed minimum is used, which is still a rigorous certificate for the data
    at hand -- and roughly four times tighter, because Lemma B's `g = prod ||W_l||`
    overshoots.
    """
    s = np.asarray(s, dtype=float)
    c = np.asarray(c, dtype=float)
    m = (s > floor) & (c > 0)
    if int(m.sum()) < 3:
        return GainCertificate(*(float("nan"),) * 5, floor_is_proved=False)
    ls, lK = np.log(s[m]), np.log(c[m] * s[m] ** (2.0 / depth - 2.0))
    if weight_norms is not None:
        lo = float(np.min(np.log(gain_floor(s[m], weight_norms, depth)
                                 * s[m] ** (2.0 / depth - 2.0))))
        proved = True
    else:
        lo, proved = float(lK.min()), False
    hi = float(lK.max())
    exponent = float(np.polyfit(ls, lK, 1)[0]) + balanced_exponent(depth)
    return GainCertificate(
        exponent=exponent,
        deviation=abs(exponent - balanced_exponent(depth)),
        bound=slope_bound(ls, lo, hi),
        band=hi - lo,
        sd_log_s=float(ls.std()),
        floor_is_proved=proved,
    )
