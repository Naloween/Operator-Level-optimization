"""Dynamics at a fixed gate pattern: the exact identities the measurements rest on."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from olo.models.crelu_mlp import CReLUMLP
from olo.theory.modes import (
    gates_of_pattern,
    hamming_mix,
    pattern_dynamics,
    random_modes,
    realized_modes,
)


def _net(depth=5, width=8, init="xavier", seed=0):
    net = CReLUMLP(d_in=width, d_out=width, width=width, depth=depth).double()
    net.initialize(init, seed=seed)
    return net


def test_realized_pattern_reproduces_the_true_jacobian():
    """`J_eps = J(x)` when eps is the pattern x actually realizes."""
    net = _net()
    X = torch.randn(4, 8, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    eps = realized_modes(net, X)[0]
    with torch.no_grad():
        a = net.operator(X[:1], gates_of_pattern(net, eps))[0]
        b = net.operator(X[:1])[0]
    assert torch.allclose(a, b, atol=1e-14)


def test_pattern_operator_does_not_depend_on_the_probe_input():
    """The whole point: at a prescribed pattern the operator is input-free."""
    net = _net()
    eps = random_modes(net, 1, torch.Generator().manual_seed(2))[0]
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        gates = gates_of_pattern(net, eps)
        a = net.operator(torch.randn(1, 8, generator=g, dtype=torch.float64), gates)[0]
        b = net.operator(torch.randn(1, 8, generator=g, dtype=torch.float64), gates)[0]
    assert torch.allclose(a, b, atol=1e-15)


def test_velocity_matches_a_finite_difference_of_the_pattern_spectrum():
    """`sdot_k(eps)` is the real derivative of `J_eps`'s singular values under `-Gamma`."""
    net = _net(depth=4, width=6)
    g = torch.Generator().manual_seed(5)
    eps = random_modes(net, 1, g)[0]
    grads = [torch.randn(W.shape, generator=g, dtype=torch.float64) for W in net.weights]
    probe = torch.zeros(1, 6, dtype=torch.float64)

    pred = pattern_dynamics(net, eps, grads)["s"], pattern_dynamics(net, eps, grads)["sdot"]
    h = 1e-7
    with torch.no_grad():
        for W, G in zip(net.weights, grads):
            W.sub_(h * G)
        s2 = torch.linalg.svdvals(net.operator(probe, gates_of_pattern(net, eps))[0]).numpy()
    fd = (s2 - pred[0]) / h
    assert np.allclose(fd, pred[1], rtol=1e-4, atol=1e-6)


def test_gain_drive_split_is_an_identity():
    """`sdot_k = -c_k g_k` holds by construction, so the split assumes nothing."""
    net = _net()
    g = torch.Generator().manual_seed(7)
    eps = random_modes(net, 1, g)[0]
    grads = [torch.randn(W.shape, generator=g, dtype=torch.float64) for W in net.weights]
    d = pattern_dynamics(net, eps, grads)
    assert np.allclose(d["sdot"], -d["gain"] * d["drive"], atol=1e-12)


def test_looks_linear_makes_every_pattern_identical():
    """Delta = 0 collapses the whole pattern ensemble to one operator."""
    net = _net(init="looks_linear")
    g = torch.Generator().manual_seed(11)
    eps = random_modes(net, 3, g)
    probe = torch.zeros(1, 8, dtype=torch.float64)
    with torch.no_grad():
        Js = [net.operator(probe, gates_of_pattern(net, eps[i]))[0] for i in range(3)]
    assert torch.allclose(Js[0], Js[1], atol=1e-14)
    assert torch.allclose(Js[0], Js[2], atol=1e-14)


def test_hamming_mix_interpolates_between_the_two_endpoints():
    g = torch.Generator().manual_seed(13)
    net = _net()
    real = realized_modes(net, torch.randn(1, 8, generator=g, dtype=torch.float64))[0]
    rand = random_modes(net, 1, g)[0]
    assert torch.equal(hamming_mix(real, rand, 0.0, g), real)
    assert torch.equal(hamming_mix(real, rand, 1.0, g), rand)
    mid = hamming_mix(real, rand, 0.5, g)
    agree = float((mid == real).double().mean())
    assert 0.3 < agree < 1.0


def test_gates_of_pattern_rejects_a_wrong_length():
    net = _net(depth=5)
    with pytest.raises(ValueError, match="need 4 sign vectors"):
        gates_of_pattern(net, torch.ones(2, 8, dtype=torch.float64))
