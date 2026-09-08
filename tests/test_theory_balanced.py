"""The closed-form mode gain must equal the sum it claims to summarize.

`c_ij = (s_i^2 - s_j^2)/(s_i^{2/L} - s_j^{2/L})` is asserted to be exactly
`sum_k A_k A_k^T G B_k^T B_k` for balanced factors. That is checkable against an explicitly
constructed balanced network, and against the framework's own Proposition 3.1 diagnostic,
so the analysis is pinned to the code rather than sitting beside it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from olo.diagnostics.mismatch import gd_first_order_alignment
from olo.models import DeepLinear
from olo.theory import balanced


def _balanced_net(s: np.ndarray, depth: int, seed: int = 0):
    """A deep linear net whose product is U diag(s) V^T with every factor balanced."""
    d = len(s)
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))[0]
    root = torch.diag(torch.tensor(s, dtype=torch.float64) ** (1.0 / depth))

    net = DeepLinear(d_in=d, d_out=d, width=d, depth=depth).to(torch.float64)
    with torch.no_grad():
        for k in range(depth):
            W = root.clone()
            if k == 0:
                W = W @ V.T
            if k == depth - 1:
                W = U @ W
            net.weights[k].copy_(W)
    return net, U, V


@pytest.mark.parametrize("depth", [2, 4, 8, 32])
def test_closed_form_matches_the_context_sum(depth):
    """c_ij must reproduce sum_k A_k A_k^T G B_k^T B_k exactly."""
    s = np.array([1.7, 1.0, 0.6, 0.2])
    net, U, V = _balanced_net(s, depth)
    x = torch.eye(len(s), dtype=torch.float64)

    G = torch.randn(len(s), len(s), generator=torch.Generator().manual_seed(1),
                    dtype=torch.float64)
    summed = torch.zeros_like(G)
    for A, B in net.all_contexts(x):
        A, B = A[0], B[0]
        summed += (A @ A.T) @ G @ (B.T @ B)

    modal = U.T @ G @ V                                  # into the (U, V) basis
    predicted = U @ (torch.tensor(balanced.gain_matrix(s, depth)) * modal) @ V.T
    assert torch.allclose(predicted, summed, rtol=1e-8, atol=1e-8), \
        float((predicted - summed).abs().max())


@pytest.mark.parametrize("depth", [1, 2, 8, 64, 256])
def test_an_isometry_has_gain_L_in_every_mode(depth):
    """At s = 1 the step is exactly L times the desired one: aligned, L times too long."""
    s = np.ones(5)
    C = balanced.gain_matrix(s, depth)
    assert np.allclose(C, depth), C
    assert balanced.alignment(s, np.random.default_rng(0).normal(size=(5, 5)), depth) \
        == pytest.approx(1.0)


def test_isometric_alignment_agrees_with_the_measured_diagnostic():
    """The prediction has to match what the runner's own Prop 3.1 diagnostic reports."""
    depth = 16
    s = np.ones(6)
    net, _, _ = _balanced_net(s, depth)
    x = torch.eye(6, dtype=torch.float64)
    G = torch.randn(6, 6, generator=torch.Generator().manual_seed(2),
                    dtype=torch.float64).unsqueeze(0)

    measured = gd_first_order_alignment(net.all_contexts(x), G)
    assert measured["cos_gd_first_order"] == pytest.approx(1.0, abs=1e-10)
    assert measured["gd_gain"] == pytest.approx(float(depth), rel=1e-8)


def test_the_bias_exponent_saturates_at_two():
    """This is why more depth stops making the bias worse -- and why there is no wall from it."""
    assert balanced.bias_exponent(1) == pytest.approx(0.0)      # linear model: no bias
    assert balanced.bias_exponent(2) == pytest.approx(1.0)
    assert balanced.bias_exponent(8) == pytest.approx(1.75)
    assert balanced.bias_exponent(256) == pytest.approx(1.9921875)
    assert balanced.bias_exponent(1024) == pytest.approx(1.998046875)
    # depth 256 -> 1024 is a change of under a percent
    assert abs(balanced.bias_exponent(1024) - balanced.bias_exponent(256)) < 0.01
    assert balanced.bias_exponent(10**9) < 2.0


def test_a_collapsed_mode_is_a_fixed_point():
    """Zero gain at s = 0: collapse is one-way, which is why effective rank only falls."""
    assert balanced.mode_gain(0.0, 0.0, 32) == pytest.approx(0.0)
    assert balanced.gain_matrix(np.array([1.0, 0.0]), 32)[1, 1] == pytest.approx(0.0)


def test_anisotropy_is_amplified_more_at_depth_but_boundedly():
    s = np.array([1.0, 0.1])
    amps = [balanced.condition_amplification(s, L) for L in (2, 8, 64, 1024)]
    assert amps == sorted(amps), amps                    # deeper amplifies more
    assert amps[-1] < 10 ** 2.0 * 1.01                   # but bounded by the s^2 limit
