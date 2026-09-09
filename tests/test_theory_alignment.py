"""The modal reduction: exact when aligned, and measurably wrong when not."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from olo.models.crelu_mlp import CReLUMLP
from olo.models.deep_linear import DeepLinear
from olo.theory.alignment import modal_reduction
from olo.theory.nonlinear import decompose


def _diag_net(scales, width=6):
    net = DeepLinear(d_in=width, d_out=width, width=width, depth=len(scales)).double()
    with torch.no_grad():
        for W, a in zip(net.weights, scales):
            W.copy_(torch.diag(torch.as_tensor(a, dtype=torch.float64)))
    return net


def test_reduction_is_exact_on_an_aligned_network():
    """Diagonal weights and a diagonal gradient: the reduction costs nothing."""
    rng = np.random.default_rng(0)
    scales = [np.exp(rng.normal(0, 0.3, size=6)) for _ in range(4)]
    net = _diag_net(scales)
    X = torch.eye(6, dtype=torch.float64)
    G = torch.diag(torch.as_tensor(rng.normal(size=6), dtype=torch.float64)).unsqueeze(0)
    mc = modal_reduction(net, X, G)
    assert mc.reduction_error < 1e-12
    assert mc.alignment == pytest.approx(1.0, abs=1e-12)


def test_exact_velocity_matches_the_master_equation():
    """`exact` is the same quantity `decompose` computes, by an independent route."""
    torch.manual_seed(0)
    net = DeepLinear(d_in=5, d_out=5, width=5, depth=4).double()
    net.initialize("xavier", seed=2)
    X = torch.randn(3, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    G = torch.randn(1, 5, 5, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    mc = modal_reduction(net, X[:1], G)
    terms = decompose(net, X[:1], G)
    assert np.allclose(mc.exact, terms.total.numpy(), atol=1e-10)


def test_aligned_network_ignores_off_diagonal_gradients_entirely():
    """In an aligned network an off-diagonal G moves the bases, not the singular values."""
    net = _diag_net([np.full(6, 1.1)] * 3)
    X = torch.eye(6, dtype=torch.float64)
    G = torch.zeros(1, 6, 6, dtype=torch.float64)
    G[0, 0, 1] = 1.0
    mc = modal_reduction(net, X, G)
    assert np.allclose(mc.diagonal, 0.0, atol=1e-12)
    assert np.allclose(mc.exact, 0.0, atol=1e-12)      # ...so the reduction loses nothing


def test_misalignment_is_what_breaks_the_reduction():
    """Rotate one layer out of the operator's basis: the diagonal prediction now misses."""
    scales = [np.array([2.0, 1.5, 1.0, 0.7, 0.5, 0.3])] * 3
    net = _diag_net(scales)
    th = 0.7
    R = torch.eye(6, dtype=torch.float64)
    R[0, 0] = R[1, 1] = np.cos(th)
    R[0, 1], R[1, 0] = -np.sin(th), np.sin(th)
    with torch.no_grad():
        net.weights[1].copy_(R @ net.weights[1])       # breaks alignment, not diagonality of P
    X = torch.eye(6, dtype=torch.float64)
    G = torch.randn(1, 6, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    mc = modal_reduction(net, X, G)
    assert mc.alignment < 0.99
    assert mc.reduction_error > 1e-3


def test_diagonality_is_between_zero_and_one():
    net = DeepLinear(d_in=6, d_out=6, width=6, depth=5).double()
    net.initialize("xavier", seed=0)
    X = torch.randn(4, 6, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    G = torch.randn(1, 6, 6, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    mc = modal_reduction(net, X, G)
    for d in (mc.diag_A, mc.diag_B):
        assert np.all(d >= 0) and np.all(d <= 1 + 1e-12)
    assert 0.0 <= mc.alignment <= 1.0


def test_crelu_network_is_measurable_and_not_perfectly_aligned():
    """The nonlinear setting the reduction has never been checked in."""
    net = CReLUMLP(d_in=8, d_out=8, width=8, depth=6).double()
    net.initialize("xavier", seed=0)
    g = torch.Generator().manual_seed(3)
    X = torch.randn(6, 8, generator=g, dtype=torch.float64)
    G = torch.randn(6, 8, 8, generator=g, dtype=torch.float64)
    mc = modal_reduction(net, X, G)
    assert np.isfinite(mc.reduction_error)
    assert mc.alignment < 0.999
    assert mc.separation > 0
