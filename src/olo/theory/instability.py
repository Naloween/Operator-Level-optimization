"""The low-rank bias as an instability: a seed, a clock, and an exponent.

The bias is usually stated as "big modes grow faster", via the exponent `2 - 2/L` in
`s_k' = -L s_k^{2-2/L} g_k`. That framing hides what actually decides whether a network
collapses, because it conflates the architecture's contribution with the task's, and it
measures progress in training steps when the right clock is something else entirely.

**Move to logarithms.** With `m_k = log s_k` and the **log-velocity**

    omega_k := s_k' / s_k = -L g_k s_k^{phi},        phi := 1 - 2/L

the separation between two modes, `r_jk = m_j - m_k`, obeys `r_jk' = omega_j - omega_k`.
So only the *spread* of omega across modes matters. Its mean is a uniform rescaling of the
whole operator: it changes no singular-value ratio, no condition number, no effective rank.

**Suppose the gradient follows a power of the mode size,** `g_k = gamma s_k^p`. This is not
an extra assumption so much as a parameterization -- `p` is measured, not posited, and it
is what a task supplies:

* `p = 0` -- gradient components independent of mode size (an isotropic residual).
* `p = 1` -- a pure-rescale target, `P* = alpha P`: the task asks for no change of shape.
* `p < 0` -- the task actively opposes the bias, pushing hardest on the smallest modes.

Then `omega_k = -L gamma s_k^{psi}` with

    psi := phi + p = (1 - 2/L) + p

and since `d omega / d m = psi * omega`, every separation obeys `r' = psi * omega_bar * r`.
Two independent contributions, cleanly split:

* **`phi = 1 - 2/L` is the architecture's.** It is 0 at `L = 2` and saturates at 1. Depth
  contributes a bounded amount and nothing more; 256 and 1024 differ by 0.6%.
* **`p` is the task's.** A task can add to the bias, or cancel it (`p = -phi`), or invert it.

**And the clock is scale, not time.** `omega_bar = d(log sbar)/dt` for the geometric mean
`sbar`, so `d(log r)/d(log sbar) = psi` and, integrating,

    r(t) / r(0) = ( sbar(t) / sbar(0) ) ** psi.

Separation is not a function of how long you trained. It is a function of **how much the
operator grew**, raised to `psi`. Three consequences, each falsifiable:

1. **An operator that does not grow develops no bias, at any depth.** Growth is the clock.
   A phase in which the operator shrinks *un*-separates the spectrum -- so non-monotone
   effective rank is predicted, not an artefact.
2. **The law is multiplicative in the seed, so `r(0)` never washes out.** A Xavier start
   with `r(0) ~ 20` nats and a looks-linear start with `r(0) = 0` are the two ends of the
   mechanism, not two runs of one experiment. At `r(0) = 0` the right-hand side is zero at
   every depth and for every task: **an exact isometry is a fixed point of the bias.**
   Something else must supply the seed, and because the growth is exponential in `log sbar`
   the seed enters only logarithmically -- noise injected mid-training is amplified exactly
   like anisotropy present at initialization, which is what makes a long flat plateau
   followed by a sudden fall the expected shape rather than a surprise.
3. **The bias is an instability, not a drive.** The rate multiplies whatever anisotropy is
   already there.

**Measuring it without assuming anything.** Regress the log-velocity on the log-spectrum:

    omega_k = a + b m_k + residual.

Then `r_jk' = b r_jk + (residual difference)`, exactly -- no balancedness, alignment, or
linearity assumption. So `b` (`Amplification.rate`) *is* the bias rate for whatever model
produced the numbers, `b / omega_bar` estimates `psi`, and `psi - phi` recovers the task's
exponent `p`. Because `s_k' = self_k + cross_k` splits linearly, `b` splits too: `b_self`
is what the fixed-gates theory covers and `b_cross` is everything the nonlinearity adds.
That is a sharper question than `|cross| / |self|`, which measures magnitude and therefore
cannot distinguish a term that reshapes the spectrum from one that merely rescales it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def log_velocity_exponent(depth: int) -> float:
    """`phi = 1 - 2/L`: the architecture's share of the amplification exponent.

    0 at depth 2 (log-velocity independent of mode size: no feedback), saturating at 1.
    """
    return 1.0 - 2.0 / depth


def amplification_exponent(depth: int, task_exponent: float = 0.0) -> float:
    """`psi = (1 - 2/L) + p`: architecture plus task.

    `r` grows as `sbar ** psi`. `psi <= 0` means growth does not separate the spectrum;
    `p = -(1 - 2/L)` is the task that exactly cancels the depth bias.
    """
    return log_velocity_exponent(depth) + float(task_exponent)


def feedback_rate(s: float, gbar: float, depth: int, task_exponent: float = 0.0) -> float:
    """`lambda = -psi L s^phi gbar`: the exponential rate at which separation self-amplifies.

    Positive when `gbar < 0`, i.e. while the loss is growing the modes. At `p = 0` this is
    `-(L-2) s^phi gbar`, and it is exactly zero at depth 2.
    """
    psi = amplification_exponent(depth, task_exponent)
    return -psi * depth * s ** log_velocity_exponent(depth) * gbar


def drive(s: float, dg: float, depth: int) -> float:
    """`D = -L s^phi (g_j - g_k)`: separation created directly by gradient anisotropy.

    The part of the dynamics that does not vanish at an isometry -- and so the only part
    that can supply a seed from nothing.
    """
    return -depth * s ** log_velocity_exponent(depth) * dg


def separation_trajectory(
    r0: float, s: float, gj: float, gk: float, depth: int, t: np.ndarray
) -> np.ndarray:
    """`r(t)` under the linearized law `r' = D + lambda r`, for comparison with measurement."""
    lam = feedback_rate(s, 0.5 * (gj + gk), depth)
    d = drive(s, gj - gk, depth)
    t = np.asarray(t, dtype=float)
    if abs(lam) < 1e-300:
        return r0 + d * t
    return (r0 + d / lam) * np.exp(lam * t) - d / lam


def balanced_flow(
    s0: np.ndarray, g: np.ndarray, depth: int, dt: float, steps: int
) -> np.ndarray:
    """Exact integration of `s_k' = -L s_k^{2-2/L} g_k` (RK4), for validating the algebra.

    Returns (steps+1, k) trajectories. `g` is held fixed, which is the regime the
    expansion describes; a real loss moves it.
    """
    exp = 2.0 - 2.0 / depth
    g = np.asarray(g, dtype=float)
    f = lambda s: -depth * np.clip(s, 0.0, None) ** exp * g
    out = [np.asarray(s0, dtype=float)]
    for _ in range(steps):
        s = out[-1]
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        out.append(np.clip(s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, None))
    return np.array(out)


# -- the integrated law -----------------------------------------------------


def _boxcox(s, psi: float):
    """`u = (s^-psi - 1)/psi`, the coordinate the flow translates rigidly.

    Box-Cox rather than plain `s^-psi` so that `psi = 0` -- the task that exactly cancels
    the depth bias -- is the continuous limit `u = -log s` rather than a division by zero.
    """
    s = np.clip(np.asarray(s, dtype=float), 1e-300, None)
    return -np.log(s) if abs(psi) < 1e-12 else (s ** (-psi) - 1.0) / psi


def _unboxcox(u, psi: float):
    """Inverse of `_boxcox`: `s = (1 + psi u)^(-1/psi)`, and `e^-u` at `psi = 0`."""
    u = np.asarray(u, dtype=float)
    if abs(psi) < 1e-12:
        return np.exp(-u)
    return np.clip(1.0 + psi * u, 1e-300, None) ** (-1.0 / psi)


def spectral_flow(s0, c: float, depth: int, task_exponent: float = 0.0):
    """Where the flow takes a whole spectrum: `u_k(t) = u_k(0) + c` in the coordinate above.

    `s_k' = -L gamma s_k^{1+psi}` is separable and integrates to a *translation* of every
    `u_k` by one and the same `c = L gamma t`. So the spectrum at any later time is a
    one-parameter function of the spectrum at time zero. Nothing else about the trajectory
    enters: not the learning rate, not the gradient magnitude, not the number of steps.

    That is what makes the law falsifiable with essentially no fitting -- read `c` off the
    spectrum's scale and the other `d-1` modes are *predicted*. What deviation measures is
    the part of `g` that is not a power law in `s`, i.e. the genuine drive.
    """
    psi = amplification_exponent(depth, task_exponent)
    return _unboxcox(_boxcox(s0, psi) + c, psi)


def _scale_coord(s, depth: int, task_exponent: float = 0.0) -> float:
    """`mean_k u(s_k)` -- the one summary of a spectrum this flow moves rigidly.

    Because the flow translates every `u_k` by a *shared* `c`, averaging over modes is
    exact: this mean translates by `c` and does nothing else. It is not the geometric
    mean; that is its first-order approximation near an isometry.
    """
    return float(_boxcox(s, amplification_exponent(depth, task_exponent)).mean())


def fit_scale(s0, s1, depth: int, task_exponent: float = 0.0) -> float:
    """The `c` carrying `s0` to `s1`, from one scalar summary -- no per-mode fitting."""
    return _scale_coord(s1, depth, task_exponent) - _scale_coord(s0, depth, task_exponent)


def scale_separation_law(s0, s1, depth: int, task_exponent: float = 0.0) -> float:
    """Predicted `r(t)/r(0)`: the separation growth implied by the scale growth.

    To first order in the separation this is `(sbar(t)/sbar(0)) ** psi`. It is exactly 1
    when the scale is unchanged, and exactly 1 for every scale change when `psi = 0`.
    """
    psi = amplification_exponent(depth, task_exponent)
    num = 1.0 + psi * _scale_coord(s0, depth, task_exponent)
    den = 1.0 + psi * _scale_coord(s1, depth, task_exponent)
    return num / den if den != 0 else float("inf")


def predict_spectrum(s0, s1, depth: int, task_exponent: float = 0.0):
    """The full spectrum the law predicts at time t, given `s0` and only the scale of `s1`.

    `d-1` predicted numbers against 1 fitted one. Comparing this with the measured `s1` is
    the cleanest available test of the whole picture.
    """
    return spectral_flow(s0, fit_scale(s0, s1, depth, task_exponent), depth, task_exponent)


# -- measuring the rate on data ---------------------------------------------


@dataclass
class Amplification:
    """The regression `omega_k = a + b m_k` on one spectrum, and its split."""

    rate: float          # b: exponential rate at which any separation grows. >0 = bias.
    uniform: float       # omega_bar: mean log-velocity, i.e. pure rescaling
    spread: float        # std of the residual: separation change not explained by size
    r2: float            # how well the rich-get-richer form fits at all
    n_modes: int

    #: same regression run on the self and cross parts, when they are supplied
    rate_self: float = float("nan")
    rate_cross: float = float("nan")

    @property
    def psi(self) -> float:
        """`b / omega_bar`: separation growth per unit of scale growth."""
        return self.rate / self.uniform if self.uniform != 0 else float("nan")

    def task_exponent(self, depth: int) -> float:
        """`p = psi - (1 - 2/L)`: the share of the bias the task supplies, not the depth."""
        return self.psi - log_velocity_exponent(depth)

    @property
    def cross_share(self) -> float:
        """`b_cross / b`: how much of the measured bias the nonlinear coupling supplies.

        Near 0 means the fixed-gates theory accounts for the bias and the cross-input
        terms, whatever their magnitude, are spectrally neutral.
        """
        return self.rate_cross / self.rate if abs(self.rate) > 0 else float("nan")


def _slope(m: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """(slope, R^2) of `w` on `m`; nan when `m` has no spread (an isometric spectrum)."""
    mc = m - m.mean()
    var = float(mc @ mc)
    if var < 1e-12:
        return float("nan"), float("nan")
    b = float(mc @ (w - w.mean()) / var)
    resid = w - w.mean() - b * mc
    tot = float(((w - w.mean()) ** 2).sum())
    return b, (1.0 - float((resid**2).sum()) / tot if tot > 0 else float("nan"))


def amplification(
    s: np.ndarray,
    ds: np.ndarray,
    self_term: np.ndarray | None = None,
    cross_term: np.ndarray | None = None,
    floor: float = 1e-10,
) -> Amplification:
    """Fit the rich-get-richer rate to a measured spectrum and its velocity.

    `s` are singular values, `ds` their time derivatives (`self_term + cross_term` when
    those are given). Modes below `floor` are dropped: their log-velocity is dominated by
    rounding, and a mode that has already collapsed carries no information about the rate
    at which modes collapse.
    """
    s = np.asarray(s, dtype=float)
    ds = np.asarray(ds, dtype=float)
    keep = s > floor
    if int(keep.sum()) < 3:
        return Amplification(float("nan"), float("nan"), float("nan"), float("nan"),
                             int(keep.sum()))
    m = np.log(s[keep])
    w = ds[keep] / s[keep]
    b, r2 = _slope(m, w)
    resid = w - w.mean() - (0.0 if np.isnan(b) else b) * (m - m.mean())
    out = Amplification(rate=b, uniform=float(w.mean()), spread=float(resid.std()),
                        r2=r2, n_modes=int(keep.sum()))
    if self_term is not None:
        out.rate_self = _slope(m, np.asarray(self_term, float)[keep] / s[keep])[0]
    if cross_term is not None:
        out.rate_cross = _slope(m, np.asarray(cross_term, float)[keep] / s[keep])[0]
    return out


def separation(s: np.ndarray) -> float:
    """`log(s_max/s_min)` over the nonzero modes: the quantity the rate acts on."""
    s = np.asarray(s, dtype=float)
    nz = s[s > 0]
    return float(np.log(nz.max() / nz.min())) if nz.size > 1 else 0.0
