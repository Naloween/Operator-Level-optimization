"""Initialization schemes.

Four schemes span the conditioning axis this project studies:

    xavier        i.i.d. N(0, gain^2 / fan_in) -- the standard regime, where products of
                  random matrices separate exponentially with depth and the operator
                  spectrum collapses.
    haar          (semi-)orthogonal -- every layer has unit singular values, so collapse
                  is absent layer-wise, but random orientations still make the *context
                  Grams* anisotropic.
    identity      W = I -- the perfectly conditioned control: every context singular value
                  is 1, isolating update-direction mismatch from conditioning.
    looks_linear  block structure making the network exactly linear at init, so J(x) is
                  orthogonal for every input at any depth (see `olo.models.crelu_mlp` and
                  `olo.models.relu_mlp`). Only meaningful for rectifier models, which
                  build it themselves; the generic path rejects it.

`gain` is a multiplier on the weight *standard deviation* (not variance): a model whose
activation halves the signal energy passes gain = sqrt(2).
"""
from __future__ import annotations

import math

import torch


def make_weight(
    shape: tuple[int, int],
    scheme: str,
    gen: torch.Generator,
    gain: float = 1.0,
) -> torch.Tensor:
    """A single layer's initial weight of shape (n_out, n_in)."""
    n_out, n_in = shape
    if scheme == "xavier":
        return torch.randn(n_out, n_in, generator=gen) * (gain / math.sqrt(n_in))
    if scheme == "haar":
        return gain * haar(n_out, n_in, gen)
    if scheme == "identity":
        return torch.eye(n_out, n_in)
    if scheme == "looks_linear":
        raise ValueError(
            "looks_linear has no generic form: it is a block structure specific to the "
            "rectifier models (relu_mlp, crelu_mlp), which construct it themselves. "
            "Use 'identity' or 'haar' for linear and fixed-gate models."
        )
    raise ValueError(
        f"unknown init scheme {scheme!r}; known: xavier, haar, identity, looks_linear"
    )


def haar(n_out: int, n_in: int, gen: torch.Generator) -> torch.Tensor:
    """Haar-distributed (semi-)orthogonal matrix of shape (n_out, n_in).

    Built from the SVD of a Gaussian rather than QR: QR's sign convention is not
    Haar-uniform without an explicit diagonal correction, and at the depths used here a
    systematic bias in orientation would be indistinguishable from a real effect.

    Returned in float64 and cast by the caller when the parameter is narrower. Rounding
    the factor to float32 here would cost about 1e-7 of orthogonality per layer, which
    compounds through a deep product -- at depth 16 it already shows up as a condition
    number of 1.0000002 for an operator that is supposed to be exactly orthogonal. The
    whole point of these initializations is that the product is *exact*, so the precision
    is kept until the parameter forces otherwise.
    """
    a = torch.randn(max(n_out, n_in), min(n_out, n_in), generator=gen, dtype=torch.float64)
    u, _, vh = torch.linalg.svd(a, full_matrices=False)
    q = u @ vh
    return q if n_out >= n_in else q.T
