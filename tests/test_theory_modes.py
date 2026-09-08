"""The mode reformulation must be exact, and its degeneracies explicit.

A mode is a gate pattern chosen freely, not one realized by an input. The claim that the
network *is* a sign-indexed family of deep linear networks is what lets randomness come
from the mode rather than from the weights, so it has to hold identically, not
approximately.
"""
from __future__ import annotations

import pytest
import torch

from olo.models import CReLUMLP
from olo.theory import modes


def _net(L=5, d=6, init="xavier", seed=0):
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=L).to(torch.float64)
    net.initialize(init, seed)
    return net


def test_realized_mode_reproduces_the_jacobian_exactly():
    """J(x) = J_{eps(x)}: the mode view is a rewriting, not an approximation."""
    torch.manual_seed(0)
    net = _net()
    X = torch.randn(8, 6, dtype=torch.float64)
    E = modes.realized_modes(net, X)
    J = net.operator(X)
    for b in range(X.shape[0]):
        Je = modes.operator_of_mode(net, [E[b, l] for l in range(net.depth - 1)])
        assert torch.allclose(Je, J[b], atol=1e-13), float((Je - J[b]).abs().max())


def test_modes_need_not_be_realizable():
    """An arbitrary sign pattern still defines an operator -- that is the point."""
    torch.manual_seed(0)
    net = _net()
    made_up = [torch.ones(net.width, dtype=torch.float64) for _ in range(net.depth - 1)]
    J = modes.operator_of_mode(net, made_up)
    assert torch.isfinite(J).all() and J.shape == (6, 6)


def test_the_ensemble_is_degenerate_at_looks_linear_init():
    """Delta = 0 makes every mode give the same operator; mode structure needs training."""
    net = _net(init="looks_linear")
    g = torch.Generator().manual_seed(0)
    R = modes.random_modes(net, 4, g)
    Js = [modes.operator_of_mode(net, [R[i, l] for l in range(net.depth - 1)])
          for i in range(4)]
    for J in Js[1:]:
        assert torch.allclose(J, Js[0], atol=1e-13), "modes should coincide when Delta = 0"


def test_mean_of_the_ensemble_is_the_linear_part():
    """E_eps[M_l] = S_l, because E[diag(eps)] = 0 -- the ensemble is centred on the linear net."""
    torch.manual_seed(0)
    net = _net()
    S, D = modes.mode_factors(net)
    g = torch.Generator().manual_seed(1)
    n, l = 4000, 2
    acc = torch.zeros_like(S[l])
    for _ in range(n):
        e = torch.randint(0, 2, (net.width,), generator=g, dtype=torch.float64) * 2 - 1
        acc += (S[l] + D[l] @ torch.diag(e)) / n
    assert float((acc - S[l]).norm() / S[l].norm()) < 0.05


def test_wrong_number_of_sign_vectors_is_rejected():
    net = _net(L=4)
    with pytest.raises(ValueError, match="sign vectors"):
        modes.operator_of_mode(net, [torch.ones(net.width, dtype=torch.float64)])


def test_random_mode_statistics_hit_their_reference_values():
    """The comparison is only meaningful if the Rademacher baseline is right."""
    torch.manual_seed(0)
    net = _net(L=6)
    g = torch.Generator().manual_seed(2)
    R = modes.random_modes(net, 512, g)
    assert float(R.mean(0).abs().mean()) < 0.08            # ~1/sqrt(512)
    assert float((R.unsqueeze(0) == R.unsqueeze(1)).double().mean()) == pytest.approx(0.5, abs=0.02)
