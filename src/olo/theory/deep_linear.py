"""Closed forms for the operator step in deep linear networks.

The first-order account (Proposition 3.1) says the induced operator motion is
`-eta sum_k A_k A_k^T G B_k^T B_k`. At an identity initialization every context is the
identity, so that term is `-L eta G` -- exactly parallel to `-G`. Read literally, this says
there is *no* mismatch, which is wrong: the factors do not act one at a time, and their
product carries every higher-order term too.

For identity init the whole thing is available in closed form. Every layer receives the
same increment `dW_l = -eta G`, so the updated factors all equal `I - eta G`, they commute,
and the product telescopes:

    P' = (I - eta G)^L,        dP = (I - eta G)^L - I = sum_{k>=1} C(L,k) (-eta G)^k

The first-order term `-L eta G` is parallel to `-G`; the k-th term carries `G^k`, which is
not. So the mismatch is real, second-order, and *matrix-valued* -- for a scalar G every
term would be parallel and there would be no mismatch at all.

The ratio of the second-order term to the first is `((L-1)/2) eta ||G||`, so depth and step
size enter only through the product

    tau = L * eta * ||G||

Two consequences worth stating as predictions rather than observations:

  - the usable learning rate scales as `1/L`. The depth wall for gradient descent on a
    perfectly conditioned network is a *step-size* wall, not a conditioning one.
  - runs with the same `tau` have the same alignment regardless of how it was reached.
    L=4 at eta=1e-2 and L=32 at eta=1e-3 give tau = 0.146 and 0.117, and cosines 0.999906
    and 0.999901.

`alignment` is exact, not asymptotic, so it can be checked against a measured step to
machine precision -- which `tests/test_theory.py` does.
"""
from __future__ import annotations

import torch


def gd_operator_step(G: torch.Tensor, eta: float, depth: int) -> torch.Tensor:
    """Exact `dP` from one simultaneous gradient step at an identity initialization.

    `(I - eta G)^L - I`, computed by binary powering rather than term by term.
    """
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    if G.shape[-2] != G.shape[-1]:
        raise ValueError(
            f"the closed form needs a square operator (identity init implies n_L = n_0); "
            f"got G of shape {tuple(G.shape)}"
        )
    return torch.linalg.matrix_power(eye - eta * G, depth) - eye


def alignment(G: torch.Tensor, eta: float, depth: int) -> float:
    """cos(dP, -G) for that step: the realized operator alignment, exactly."""
    dP = gd_operator_step(G, eta, depth)
    want = -G
    denom = dP.norm() * want.norm()
    if denom == 0:
        return float("nan")
    return float((dP.flatten() @ want.flatten()) / denom)


def mismatch_parameter(G: torch.Tensor, eta: float, depth: int) -> float:
    """`tau = L eta ||G||_2`: the single variable depth and step size collapse into.

    The spectral norm is the right scale here because the expansion is in powers of
    `eta G`, and it is `||eta G||_2 < 1` that controls whether the series is dominated by
    its first term.
    """
    return float(depth * eta * torch.linalg.matrix_norm(G.to(torch.float64), ord=2))


def max_stable_lr(G: torch.Tensor, depth: int, tol: float = 1.0) -> float:
    """Largest `eta` with `tau <= tol` -- the `1/L` scaling made explicit."""
    scale = float(torch.linalg.matrix_norm(G.to(torch.float64), ord=2))
    return tol / (depth * scale) if scale > 0 else float("inf")


def alignment_curve(
    G: torch.Tensor, depth: int, etas: list[float]
) -> tuple[list[float], list[float]]:
    """(tau, cos) over a range of step sizes, for plotting the collapse onto one curve."""
    return (
        [mismatch_parameter(G, e, depth) for e in etas],
        [alignment(G, e, depth) for e in etas],
    )
