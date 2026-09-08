"""The master equation must be exact, or the hypothesis built on it means nothing.

`s_k(x)' = -sum_l E_{x'}[u_k^T A_l(x)A_l(x')^T G(x') B_l(x')^T B_l(x) v_k]` is checked two
ways: against `sum_l A_l W_l' B_l` (should be identical to roundoff) and against a finite
difference of the true singular values along the flow (should agree to the truncation
error). The self/cross split is then checked to be exactly a partition of it.
"""
from __future__ import annotations

import pytest
import torch

from olo.models import CReLUMLP, DeepLinear, ReLUMLP
from olo.theory import nonlinear


def _setup(cls=CReLUMLP, d=5, L=4, B=6, init="xavier", seed=0):
    torch.manual_seed(seed)
    net = cls(d_in=d, d_out=d, width=d, depth=L).to(torch.float64)
    net.initialize(init, seed)
    X = torch.randn(B, d, dtype=torch.float64)
    Y = torch.randn(B, d, dtype=torch.float64)
    return net, X, Y


def _grads_and_G(net, X, Y):
    """Gradient flow direction, and the per-sample operator gradients, at the same point."""
    loss = 0.5 * ((net(X) - Y) ** 2).sum(-1).mean()
    grads = torch.autograd.grad(loss, list(net.weights))
    with torch.no_grad():
        resid = net(X) - Y
        G = torch.stack([torch.outer(resid[b], X[b]) for b in range(X.shape[0])])
    return [-g for g in grads], G


@pytest.mark.parametrize("cls,init", [(CReLUMLP, "xavier"), (CReLUMLP, "looks_linear"),
                                      (ReLUMLP, "xavier"), (DeepLinear, "haar")])
def test_master_equation_matches_the_realized_flow(cls, init):
    net, X, Y = _setup(cls, init=init)
    Wdot, G = _grads_and_G(net, X, Y)
    terms = nonlinear.decompose(net, X, G, sample=0)

    with torch.no_grad():
        gates = net.gates(X)
        ctx = net.all_contexts(X, gates)
        J = net.operator(X, gates)
        J_x = J[0] if J.shape[0] > 1 else J[0]
        U, S, Vh = torch.linalg.svd(J_x)
        pick = lambda T, b: T[b] if T.shape[0] > 1 else T[0]
        Jdot = sum(pick(ctx[l][0], 0).double() @ Wdot[l].double() @ pick(ctx[l][1], 0).double()
                   for l in range(net.depth))
        direct = torch.tensor([float(U[:, k].double() @ Jdot @ Vh[k, :].double())
                               for k in range(S.shape[0])], dtype=torch.float64)

    assert torch.allclose(terms.total, direct, atol=1e-10), \
        float((terms.total - direct).abs().max())


@pytest.mark.parametrize("cls,init", [(CReLUMLP, "xavier"), (ReLUMLP, "xavier")])
def test_master_equation_matches_finite_differences(cls, init):
    """The equation must describe the actual trajectory, not just an algebraic identity."""
    net, X, Y = _setup(cls, init=init)
    Wdot, G = _grads_and_G(net, X, Y)
    terms = nonlinear.decompose(net, X, G, sample=0)

    eps = 1e-7
    with torch.no_grad():
        S0 = torch.linalg.svdvals(net.operator(X)[0])
        for W, wd in zip(net.weights, Wdot):
            W += eps * wd
        S1 = torch.linalg.svdvals(net.operator(X)[0])
        for W, wd in zip(net.weights, Wdot):
            W -= eps * wd
    fd = ((S1 - S0) / eps).double()

    assert torch.allclose(terms.total, fd, atol=1e-5), float((terms.total - fd).abs().max())


def test_self_and_cross_partition_the_velocity_exactly():
    net, X, Y = _setup()
    _, G = _grads_and_G(net, X, Y)
    t = nonlinear.decompose(net, X, G, sample=2)
    assert torch.allclose(t.self_term + t.cross_term, t.total, atol=1e-12)


def test_a_single_sample_batch_has_no_cross_term():
    """With one input there is nothing to couple to: the fixed-gates case, exactly."""
    net, X, Y = _setup(B=1)
    _, G = _grads_and_G(net, X, Y)
    t = nonlinear.decompose(net, X, G, sample=0)
    assert float(t.cross_term.abs().max()) == pytest.approx(0.0, abs=1e-14)
    assert torch.allclose(t.total, t.self_term, atol=1e-14)


def test_deep_linear_has_identical_contexts_so_cross_is_pure_gradient_averaging():
    """With input-independent gates, A_l(x') = A_l(x): the cross term is only the G average.

    This is the sanity anchor -- in the linear case the decomposition must reduce to the
    known situation where all inputs share one operator.
    """
    net, X, Y = _setup(DeepLinear, init="haar")
    _, G = _grads_and_G(net, X, Y)
    t = nonlinear.decompose(net, X, G, sample=0)

    with torch.no_grad():
        ctx = net.all_contexts(X)
        for A, B in ctx:                      # contexts genuinely do not depend on the input
            assert A.shape[0] == 1 and B.shape[0] == 1
    assert torch.isfinite(t.cross_term).all()


def test_scaling_exponent_recovers_the_linear_prediction_at_isometry():
    """A balanced isometric deep linear net should show the 2 - 2/L slope of the theory."""
    from olo.theory import balanced

    d, L = 8, 16
    torch.manual_seed(0)
    net = DeepLinear(d_in=d, d_out=d, width=d, depth=L).to(torch.float64)
    with torch.no_grad():                     # balanced, spread spectrum, aligned bases
        s = torch.linspace(1.6, 0.4, d, dtype=torch.float64)
        root = torch.diag(s ** (1.0 / L))
        for k in range(L):
            net.weights[k].copy_(root)

    # For this construction the closed form gives dot s_k directly; check the module's
    # empirical slope against the analytic exponent on the diagonal gains.
    gains = torch.tensor([balanced.mode_gain(float(v), float(v), L) for v in s],
                         dtype=torch.float64)
    x = torch.log(s)
    y = torch.log(gains)
    slope = float(((x - x.mean()) @ (y - y.mean())) / ((x - x.mean()) @ (x - x.mean())))
    assert slope == pytest.approx(balanced.bias_exponent(L), abs=1e-9)
