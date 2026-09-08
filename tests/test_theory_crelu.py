"""Every algebraic claim in `olo.theory.crelu`, checked rather than asserted.

The claims are what would let the fixed-gates analysis of Haas et al. (ICML 2026) reach
data-dependent gates, so they carry weight only if they are exact. Each is tested against
the model's own gates, not against a re-derivation of them.
"""
from __future__ import annotations

import pytest
import torch

from olo.models import CReLUMLP, ReLUMLP
from olo.theory import crelu


def _net(depth, d=6, init="looks_linear", seed=0):
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=depth).to(torch.float64)
    net.initialize(init, seed)
    return net


def _perturb(net, eps, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for W in net.weights:
            W += eps * torch.randn(W.shape, generator=g, dtype=torch.float64)
    return net


# -- 1. the gate is an isometry --------------------------------------------


def test_crelu_gate_is_an_isometry_and_relu_is_not():
    torch.manual_seed(0)
    d = 6
    x = torch.randn(9, d, dtype=torch.float64)

    c = _net(4, d, "xavier")
    D = c.gates(x)[1]
    eye = torch.eye(d, dtype=torch.float64)
    assert torch.allclose(D.transpose(1, 2) @ D, eye.expand_as(D.transpose(1, 2) @ D),
                          atol=1e-12), "CReLU gate is not an isometry"

    r = ReLUMLP(d_in=d, d_out=d, width=d, depth=4).to(torch.float64)
    r.initialize("xavier", 0)
    Dr = r.gates(x)[1]
    assert not torch.allclose(Dr.transpose(1, 2) @ Dr, eye.expand_as(Dr), atol=1e-6)


def test_the_isometry_means_gates_never_cost_rank():
    """A ReLU gate is a projection and destroys rank; a CReLU gate provably cannot."""
    torch.manual_seed(0)
    d = 6
    x = torch.randn(32, d, dtype=torch.float64)

    c = _net(8, d, "xavier")
    Jc = c.operator(x)
    assert min(int(torch.linalg.matrix_rank(Jc[b], atol=1e-10)) for b in range(32)) == d

    r = ReLUMLP(d_in=d, d_out=d, width=d, depth=8).to(torch.float64)
    r.initialize("xavier", 0)
    Jr = r.operator(x)
    assert min(int(torch.linalg.matrix_rank(Jr[b], atol=1e-10)) for b in range(32)) < d


# -- 2. the layer is affine in the sign pattern ----------------------------


def test_layer_decomposition_is_exact_against_the_models_own_gates():
    """M(z) = S + Delta diag(sign(z)), checked against the gates the model actually builds."""
    torch.manual_seed(0)
    d = 6
    net = _net(5, d, "xavier")
    x = torch.randn(7, d, dtype=torch.float64)
    zs, Ds = net.pre_activations(x), net.gates(x)

    for layer in (1, 2, 3):
        W = net.weights[layer]
        S, Delta = crelu.split_layer(W)
        for b in range(x.shape[0]):
            actual = W @ Ds[layer - 1][b]
            claim = S + Delta @ torch.diag(torch.sign(zs[layer - 1][b]))
            assert torch.allclose(actual, claim, atol=1e-12), (layer, b)


def test_looks_linear_is_exactly_delta_zero():
    """Not approximately linear at init -- exactly linear, with a zero nonlinear part."""
    for depth in (2, 16, 128):
        net = _net(depth)
        assert max(crelu.layer_profile(net)) == pytest.approx(0.0, abs=1e-15), depth


def test_xavier_init_has_a_nonzero_nonlinear_part():
    assert min(crelu.layer_profile(_net(8, init="xavier"))) > 0.1


# -- 3. the nonlinearity is bounded uniformly over inputs ------------------


def test_sign_flips_leave_the_perturbation_spectrum_unchanged():
    """diag(sign(z)) is orthogonal, so Delta.sign(z) has Delta's singular values, for any x."""
    torch.manual_seed(0)
    d = 8
    net = _perturb(_net(6, d), 0.15)
    x = torch.randn(50, d, dtype=torch.float64) * 3
    zs = net.pre_activations(x)
    _, Delta = crelu.split_layer(net.weights[3])
    base = torch.linalg.svdvals(Delta)
    for b in range(x.shape[0]):
        R = torch.diag(torch.sign(zs[2][b]))
        assert torch.allclose(torch.linalg.svdvals(Delta @ R), base, atol=1e-12)


def test_every_input_layer_spectrum_sits_within_delta_of_the_linear_one():
    """The uniform Weyl bound: this is what transfers separation from the linear part."""
    torch.manual_seed(0)
    d = 8
    net = _perturb(_net(6, d), 0.15)
    x = torch.randn(200, d, dtype=torch.float64) * 3
    zs = net.pre_activations(x)
    W = net.weights[3]
    S, Delta = crelu.split_layer(W)
    sS = torch.linalg.svdvals(S)
    bound = crelu.spectral_bound(W)

    worst = max(
        float((torch.linalg.svdvals(S + Delta @ torch.diag(torch.sign(zs[2][b]))) - sS)
              .abs().max())
        for b in range(x.shape[0])
    )
    assert worst <= bound + 1e-12, (worst, bound)


# -- the expansion ---------------------------------------------------------


def test_first_order_expansion_has_an_O_delta_squared_remainder():
    """J = S_L..S_1 + sum_l A_l Delta_l sign(z_l) B_l + O(||Delta||^2).

    The remainder divided by eps^2 must be constant across decades; if the first-order term
    were wrong the ratio would drift.
    """
    d, L = 6, 5
    x = torch.randn(1, d, generator=torch.Generator().manual_seed(0), dtype=torch.float64)

    ratios = []
    for eps in (1e-1, 1e-2, 1e-3):
        net = _perturb(_net(L, d), eps, seed=1)
        S = crelu.linear_part(net)
        Delta = [torch.zeros_like(S[0])] + [crelu.split_layer(W)[1]
                                            for W in net.weights[1:]]
        z = net.pre_activations(x)
        J = net.operator(x)[0].detach()

        J0 = crelu.linear_operator(net)
        first = torch.zeros_like(J0)
        for l in range(1, L):
            A = torch.eye(d, dtype=torch.float64)
            for j in range(L - 1, l, -1):
                A = A @ S[j]
            B = S[0]
            for j in range(1, l):
                B = S[j] @ B
            first = first + A @ (Delta[l] @ torch.diag(torch.sign(z[l - 1][0]))) @ B

        ratios.append(float((J - J0 - first).norm()) / eps**2)

    assert max(ratios) / min(ratios) < 1.3, ratios          # constant => O(eps^2)


def test_zeroth_order_is_input_independent():
    """The zeroth order is one shared linear network -- the object the fixed-gates theory is about."""
    net = _perturb(_net(6, 8), 0.1)
    P = crelu.linear_operator(net)
    assert P.shape == (8, 8)
    # it does not depend on x by construction; the model's own operator does
    x = torch.randn(4, 8, dtype=torch.float64)
    J = net.operator(x)
    assert J.shape[0] == 4
    assert not torch.allclose(J[0], J[1], atol=1e-6), "gates should differ across inputs here"
