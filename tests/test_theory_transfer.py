"""The transfer-operator theorems of theory/02-mode-ensemble.md, checked numerically.

Each test names the result it verifies. The proofs are only worth as much as their agreement
with what the code actually computes, and two of these (Cor. 8.1, primitivity) exist because
an earlier version of the write-up claimed strict positivity, which is false for d >= 3.
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


def _transfer(S, D, l):
    Sl, Dl = S[l].detach(), D[l].detach()
    return lambda X: Sl.T @ X @ Sl + torch.diag(torch.diag(Dl.T @ X @ Dl))


def _min_eig(X):
    return float(torch.linalg.eigvalsh((X + X.T) / 2).min())


# -- Theorem 5: the one-layer closed form -----------------------------------


def test_theorem5_one_layer_transfer_matches_monte_carlo():
    torch.manual_seed(0)
    d = 5
    net = _net(6, d)
    S, D = modes.mode_factors(net)
    X = torch.randn(d, d, dtype=torch.float64)
    X = X @ X.T

    T = _transfer(S, D, 3)
    g = torch.Generator().manual_seed(1)
    n = 40000
    mc = torch.zeros(d, d, dtype=torch.float64)
    Sl, Dl = S[3].detach(), D[3].detach()
    for _ in range(n):
        e = torch.randint(0, 2, (d,), generator=g, dtype=torch.float64) * 2 - 1
        M = Sl + Dl @ torch.diag(e)
        mc += M.T @ X @ M / n
    rel = float((mc - T(X)).norm() / T(X).norm())
    assert rel < 5 / n**0.5, rel          # within a few Monte-Carlo standard errors


# -- Theorem 6: the whole product -------------------------------------------


def test_theorem6_full_recursion_matches_random_modes():
    torch.manual_seed(0)
    d, L = 5, 5
    net = _net(L, d)
    S, D = modes.mode_factors(net)

    X = torch.eye(d, dtype=torch.float64)
    for l in range(L - 1, 0, -1):
        X = _transfer(S, D, l)(X)
    predicted = S[0].detach().T @ X @ S[0].detach()

    g = torch.Generator().manual_seed(2)
    n = 20000
    mc = torch.zeros(d, d, dtype=torch.float64)
    for _ in range(n):
        eps = [torch.randint(0, 2, (d,), generator=g, dtype=torch.float64) * 2 - 1
               for _ in range(L - 1)]
        J = modes.operator_of_mode(net, eps)
        mc += J.T @ J / n
    assert float((mc - predicted).norm() / predicted.norm()) < 0.05


# -- Theorem 7 / Corollary 6.1 ----------------------------------------------


def test_theorem7_transfer_preserves_the_psd_cone():
    torch.manual_seed(0)
    d = 6
    net = _net(5, d)
    S, D = modes.mode_factors(net)
    T = _transfer(S, D, 2)
    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        A = torch.randn(d, d, generator=g, dtype=torch.float64)
        X = A @ A.T                                    # PSD, generically full rank
        assert _min_eig(T(X)) >= -1e-12


def test_corollary61_energy_identity():
    torch.manual_seed(0)
    d = 6
    net = _net(5, d)
    S, D = modes.mode_factors(net)
    Sl, Dl = S[2].detach(), D[2].detach()
    A = torch.randn(d, d, dtype=torch.float64)
    X = A @ A.T
    lhs = float(torch.trace(_transfer(S, D, 2)(X)))
    rhs = float(torch.trace(X @ (Sl @ Sl.T + Dl @ Dl.T)))
    assert lhs == pytest.approx(rhs, rel=1e-10)


# -- Theorem 8 / Corollary 8.1: strict positivity FAILS ---------------------


def test_corollary81_transfer_is_not_strictly_positive_for_d_at_least_3():
    """The projector onto span{S e_i, Delta e_i}^perp is annihilated in direction e_i.

    An earlier draft asserted strict positivity; this is the counterexample that refutes it.
    """
    torch.manual_seed(0)
    d = 6
    net = _net(5, d)
    S, D = modes.mode_factors(net)
    Sl, Dl = S[3].detach(), D[3].detach()
    T = _transfer(S, D, 3)

    i = 0
    e_i = torch.eye(d, dtype=torch.float64)[i]
    V = torch.stack([Sl @ e_i, Dl @ e_i], dim=1)
    Q, _ = torch.linalg.qr(V)
    X = torch.eye(d, dtype=torch.float64) - Q @ Q.T           # PSD, nonzero, rank d-2

    assert _min_eig(X) >= -1e-12 and float(X.norm()) > 0.1    # a valid cone point
    assert abs(float(e_i @ T(X) @ e_i)) < 1e-12, "T(X) should be singular in direction e_i"


def test_primitivity_holds_in_practice_with_a_small_exponent():
    """Theorem 9 needs primitivity, not strict positivity. Checked, not assumed."""
    torch.manual_seed(0)
    d = 6
    net = _net(5, d)
    S, D = modes.mode_factors(net)
    Sl, Dl = S[3].detach(), D[3].detach()
    T = _transfer(S, D, 3)

    e_i = torch.eye(d, dtype=torch.float64)[0]
    V = torch.stack([Sl @ e_i, Dl @ e_i], dim=1)
    Q, _ = torch.linalg.qr(V)
    X = torch.eye(d, dtype=torch.float64) - Q @ Q.T

    for n in range(1, d + 1):
        X = T(X)
        if _min_eig(X) > 1e-13:
            assert n <= 3, f"primitive but with a large exponent ({n})"
            return
    pytest.fail("transfer operator was not primitive within d iterations")


def test_theorem9_stabilization_converges_geometrically():
    """lambda^-n T^n(X0) approaches a fixed shape, from any starting point in the cone."""
    torch.manual_seed(0)
    d = 6
    net = _net(5, d)
    S, D = modes.mode_factors(net)
    T = _transfer(S, D, 2)

    def normalize(X):
        return X / X.norm()

    g = torch.Generator().manual_seed(5)
    limits = []
    for _ in range(3):                                    # different cone starting points
        A = torch.randn(d, d, generator=g, dtype=torch.float64)
        X = A @ A.T
        for _ in range(120):
            X = normalize(T(X))
        limits.append(X)
    for X in limits[1:]:
        assert float((X - limits[0]).norm()) < 1e-6, "iterates did not reach a common ray"
