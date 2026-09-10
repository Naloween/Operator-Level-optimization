"""Basis-free spectral dynamics: the exact criterion, and where it becomes a theorem."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
import torch

from olo.models.crelu_mlp import CReLUMLP
from olo.theory.moments import (
    balanced_theta,
    chebyshev_correlation,
    power_law_criterion,
    rank_flow,
)


def test_criterion_identity_is_exact_symbolically():
    """d/dt log PR = -2 Cheb / (trM trM^2), for every spectrum and every velocity."""
    n = 4
    mu = sp.symbols(f"mu1:{n+1}", positive=True)
    x = sp.symbols(f"x1:{n+1}", real=True)
    trM, trM2 = sum(mu), sum(m**2 for m in mu)
    d_log_pr = 2 * (2 * sum(x)) / trM - (4 * sum(m * y for m, y in zip(mu, x))) / trM2
    cheb = sum(mu[k] * mu[j] * (x[k] / mu[k] - x[j] / mu[j]) * (mu[k] - mu[j])
               for k in range(n) for j in range(n))
    assert sp.simplify(d_log_pr + 2 * cheb / (trM * trM2)) == 0


def test_chebyshev_matches_the_trace_form():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = int(rng.integers(2, 8))
        mu = np.exp(rng.normal(size=d))
        x = rng.normal(size=d)
        direct = sum(mu[k] * mu[j] * (x[k] / mu[k] - x[j] / mu[j]) * (mu[k] - mu[j])
                     for k in range(d) for j in range(d))
        assert chebyshev_correlation(mu, x) == pytest.approx(direct, rel=1e-9)


def test_uniform_relative_growth_leaves_the_rank_fixed():
    """x proportional to mu is a pure rescaling: no change in effective rank."""
    mu = np.array([4.0, 2.0, 1.0, 0.25])
    assert chebyshev_correlation(mu, 0.7 * mu) == pytest.approx(0.0, abs=1e-12)


def test_power_law_criterion_sign_is_the_sign_of_theta_minus_one():
    """Log-convexity: rank falls iff theta >= 1, for any non-degenerate spectrum."""
    rng = np.random.default_rng(1)
    for _ in range(20_000):
        d = int(rng.integers(2, 12))
        mu = np.exp(rng.normal(0, rng.uniform(0.05, 2.0), size=d))
        theta = float(rng.uniform(-1.5, 4.0))
        v = power_law_criterion(mu, theta)
        if np.ptp(mu) < 1e-9:
            continue
        if theta > 1 + 1e-6:
            assert v >= -1e-12
        if theta < 1 - 1e-6:
            assert v <= 1e-12


def test_power_law_criterion_agrees_with_chebyshev():
    rng = np.random.default_rng(2)
    for _ in range(500):
        d = int(rng.integers(2, 7))
        mu = np.exp(rng.normal(size=d))
        theta = float(rng.uniform(-1, 3))
        assert chebyshev_correlation(mu, mu**theta) == pytest.approx(
            2 * power_law_criterion(mu, theta), rel=1e-9
        )


def test_balanced_theta_exceeds_one_exactly_when_depth_does():
    assert balanced_theta(1) == pytest.approx(1.0)
    for L in (2, 4, 16, 256):
        assert balanced_theta(L) > 1.0
        assert balanced_theta(L) == pytest.approx(2 - 1 / L)


def test_rank_flow_matches_finite_differences_on_a_real_network():
    """The trace formula is the true derivative of log PR under the real weight gradient."""
    torch.manual_seed(0)
    d, L = 10, 8
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=L).double()
    net.initialize("xavier", seed=0)
    g = torch.Generator().manual_seed(1)
    X = torch.randn(1, d, generator=g, dtype=torch.float64)
    grads = [torch.randn(W.shape, generator=g, dtype=torch.float64) for W in net.weights]
    gates = net.gates(X)
    with torch.no_grad():
        J = net.operator(X, gates)[0].double()
        ctx = net.all_contexts(X, gates)
        Jd = -sum(ctx[l][0][0].double() @ grads[l].double() @ ctx[l][1][0].double()
                  for l in range(L))
        rf = rank_flow(J, Jd)
        h = 1e-7
        for W, G in zip(net.weights, grads):
            W.sub_(h * G)
        J2 = net.operator(X, gates)[0].double()
        M2 = J2.T @ J2
        pr2 = float(torch.trace(M2) ** 2 / torch.trace(M2 @ M2))
    fd = (np.log(pr2) - np.log(rf.pr)) / h
    assert rf.d_log_pr == pytest.approx(fd, rel=1e-4)


def test_participation_ratio_is_a_sensible_effective_rank():
    iso = torch.eye(6, dtype=torch.float64)
    assert rank_flow(iso, torch.zeros_like(iso)).pr == pytest.approx(6.0)
    r1 = torch.zeros(6, 6, dtype=torch.float64)
    r1[0, 0] = 3.0
    assert rank_flow(r1, torch.zeros_like(r1)).pr == pytest.approx(1.0)
