"""Spectral dynamics without a basis, and therefore without the alignment assumption.

Everything in `instability.py`, `imbalance.py` and `reduction.py` tracks individual singular
values `s_k`, which requires the singular *vectors* — and collapsing the velocity to
`-c_k g_k` then needs the context Grams diagonal in those vectors. That assumption
(`00-derivation.md` (A2)) is measured to fail qualitatively away from near-isometric
networks: the diagonal surrogate reports a low-rank bias where the exact dynamics have none.

This module drops it, by tracking **basis-free** functionals instead. For `M = J^T J`,

    d/dt tr(M^p) = 2p * tr(M^{p-1} J^T Jdot)                                    [EXACT]

involves no eigenvectors at all, and

    PR := tr(M)^2 / tr(M^2)                                                     (participation ratio)

is a smooth effective rank: `PR = d` for an isometry, `PR = 1` for a rank-one operator.
`d/dt log PR` is then two traces over two traces, exact and computable in any network.

**The criterion.** Writing `mu_k` for the eigenvalues of `M` and `x_k` for the diagonal of
`J^T Jdot` in `M`'s own eigenbasis — which the traces pick out automatically, with nothing
assumed —

    d/dt log PR  =  -2 * Cheb / (tr M * tr M^2),
    Cheb := sum_{k,j} mu_k mu_j (x_k/mu_k - x_j/mu_j)(mu_k - mu_j).

Verified symbolically. So **the effective rank falls if and only if the relative growth rate
`x_k/mu_k` is positively correlated with `mu_k`** — the rich-get-richer statement, as an
exact equivalence with no hypothesis. Note `x_k/mu_k = omega_k`, the log-velocity of file 00.

**Why this is progress.** The *condition* was always assumption-free; what needed (A2) was
predicting it from the architecture. Here the prediction can be attempted on the trace
moments, which are boundable without ever choosing a basis.

**One case where that works.** If the relative rate follows a power law, `x_k = C mu_k^theta`,
then `Cheb >= 0` reduces to `f(1+theta) f(1) >= f(theta) f(2)` for the spectral moment
function `f(p) = sum_k mu_k^p`. Both exponent pairs sum to `2 + theta`, and `f` is
log-convex (Holder), so the more *spread* pair dominates: the condition is exactly
`theta >= 1`. For a balanced depth-`L` network under pure growth (`G = -cJ`) each layer
contributes `mu^{1 + (L-l)/L + (l-1)/L}`, so `theta = 2 - 1/L` and the rank falls for every
`L >= 2`, with equality at `L = 1`. Verified over 2e5 random spectra, 0 violations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RankFlow:
    """Basis-free effective rank and its exact velocity."""

    pr: float                # tr(M)^2 / tr(M^2)
    d_log_pr: float          # its logarithmic derivative; < 0 means the rank is falling
    d_log_scale: float       # d/dt log tr(M): pure growth, orthogonal to the rank question
    trM: float
    trM2: float

    @property
    def falls(self) -> bool:
        return self.d_log_pr < 0


@torch.no_grad()
def rank_flow(J: torch.Tensor, J_dot: torch.Tensor) -> RankFlow:
    """`PR` and `d/dt log PR` from four traces. No eigenvectors, no alignment.

    `d/dt tr(M) = 2 tr(J^T Jdot)` and `d/dt tr(M^2) = 4 tr(M J^T Jdot)`, so
    `d/dt log PR = 2 d(log tr M) - d(log tr M^2)`. Matches finite differences of `log PR`
    to `1e-7` on trained networks.
    """
    J, J_dot = J.double(), J_dot.double()
    M = J.T @ J
    X = J.T @ J_dot
    trM = float(torch.trace(M))
    trM2 = float(torch.trace(M @ M))
    d1 = 2.0 * float(torch.trace(X))
    d2 = 4.0 * float(torch.trace(M @ X))
    return RankFlow(
        pr=trM**2 / trM2 if trM2 > 0 else float("nan"),
        d_log_pr=(2.0 * d1 / trM - d2 / trM2) if trM > 0 and trM2 > 0 else float("nan"),
        d_log_scale=d1 / trM if trM > 0 else float("nan"),
        trM=trM, trM2=trM2,
    )


def chebyshev_correlation(mu: np.ndarray, x: np.ndarray) -> float:
    """`sum_{k,j} mu_k mu_j (x_k/mu_k - x_j/mu_j)(mu_k - mu_j)`; `>= 0` iff the rank falls.

    Equal to `2 (tr(MX) tr(M) - tr(X) tr(M^2))` in the notation above, so it is exactly the
    numerator of `-d/dt log PR` up to the positive factor `tr M * tr M^2`.
    """
    mu = np.asarray(mu, dtype=float)
    x = np.asarray(x, dtype=float)
    return 2.0 * float((mu * x).sum() * mu.sum() - x.sum() * (mu**2).sum())


def power_law_criterion(mu: np.ndarray, theta: float) -> float:
    """`f(1+theta) f(1) - f(theta) f(2)` for `f(p) = sum mu^p`; `>= 0` iff `theta >= 1`.

    The log-convexity form of the criterion when `x_k = C mu_k^theta`. Its sign is that of
    `theta - 1` whenever the spectrum is non-degenerate, by Holder.
    """
    mu = np.clip(np.asarray(mu, dtype=float), 1e-300, None)
    f = lambda p: float((mu**p).sum())
    return f(1.0 + theta) * f(1.0) - f(theta) * f(2.0)


def balanced_theta(depth: int) -> float:
    """`theta = 2 - 1/L` for a balanced depth-L network under pure growth.

    Each layer contributes `mu^{1 + (L-l)/L + (l-1)/L}` to `x`, independent of `l`. `> 1`
    for every `L >= 2`, and exactly 1 at `L = 1` — a single matrix rescaled uniformly does
    not change its effective rank.
    """
    return 2.0 - 1.0 / depth
