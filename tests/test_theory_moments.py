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


def test_eigenvalue_velocity_is_twice_the_diagonal_of_X():
    """(4.1): mudot_k = 2 x_k, with x_k the diagonal of J^T Jdot in M's eigenbasis."""
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        d = int(torch.randint(3, 8, (1,), generator=g))
        J = torch.randn(d, d, generator=g, dtype=torch.float64)
        Jd = torch.randn(d, d, generator=g, dtype=torch.float64)
        M = J.T @ J
        mu, W = torch.linalg.eigh(M)
        if float((mu[1:] - mu[:-1]).min()) < 1e-3:
            continue                                  # (4.1) needs a simple spectrum
        x = (W.T @ (J.T @ Jd) @ W).diagonal()
        h = 1e-7
        mu2 = torch.linalg.eigvalsh((J + h * Jd).T @ (J + h * Jd))
        assert torch.allclose((mu2 - mu) / h, 2 * x, atol=1e-4, rtol=1e-4)


def test_the_two_traces_are_the_two_sums_over_directions():
    """(4.2): tr(X) = sum_k x_k and tr(MX) = sum_k mu_k x_k, computed with no eigenbasis."""
    g = torch.Generator().manual_seed(1)
    for _ in range(100):
        d = int(torch.randint(2, 9, (1,), generator=g))
        J = torch.randn(d, d, generator=g, dtype=torch.float64)
        Jd = torch.randn(d, d, generator=g, dtype=torch.float64)
        M = J.T @ J
        mu, W = torch.linalg.eigh(M)
        X = J.T @ Jd
        x = (W.T @ X @ W).diagonal()
        assert float(torch.trace(X)) == pytest.approx(float(x.sum()), abs=1e-10)
        assert float(torch.trace(M @ X)) == pytest.approx(float((mu * x).sum()), abs=1e-10)


def test_flat_spectrum_and_uniform_rate_both_give_zero():
    """The two degenerate cases Theorem 21 must get right."""
    flat = np.ones(5)
    assert chebyshev_correlation(flat, np.arange(5.0)) == pytest.approx(0.0, abs=1e-12)
    mu = np.array([5.0, 2.0, 1.0])
    assert chebyshev_correlation(mu, 0.3 * mu) == pytest.approx(0.0, abs=1e-12)


# -- file 11: the mean transfer and its two branches -------------------------


def test_rademacher_transfer_identities():
    """Lemma 1: E[M X M^T] = S X S^T + Delta Dg(X) Delta^T, by exhaustive sign averaging."""
    g = torch.Generator().manual_seed(0)
    for d in (2, 3):
        S = torch.randn(d, d, generator=g, dtype=torch.float64)
        D = torch.randn(d, d, generator=g, dtype=torch.float64)
        X = torch.randn(d, d, generator=g, dtype=torch.float64)
        fwd = torch.zeros(d, d, dtype=torch.float64)
        bwd = torch.zeros(d, d, dtype=torch.float64)
        signs = torch.tensor([[1.0, -1.0]] * d, dtype=torch.float64)
        for bits in range(2**d):
            e = torch.tensor([signs[i][(bits >> i) & 1] for i in range(d)],
                             dtype=torch.float64)
            M = S + D @ torch.diag(e)
            fwd += M @ X @ M.T
            bwd += M.T @ X @ M
        fwd /= 2**d
        bwd /= 2**d
        assert torch.allclose(fwd, S @ X @ S.T + D @ torch.diag(X.diagonal()) @ D.T, atol=1e-12)
        assert torch.allclose(bwd, S.T @ X @ S + torch.diag((D.T @ X @ D).diagonal()),
                              atol=1e-12)


def test_nonlinear_branch_closes_on_diagonals():
    """Thm 5: diag(Delta Dg(X) Delta^T) = (Delta o Delta) diag(X), a nonnegative map."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        d = int(rng.integers(2, 7))
        D = rng.normal(size=(d, d))
        X = rng.normal(size=(d, d))
        X = X @ X.T
        F = D @ np.diag(np.diag(X)) @ D.T
        assert np.allclose(np.diag(F), (D**2) @ np.diag(X))


def test_nonlinear_branch_saturates_while_linear_branch_spreads():
    """Thm 4 vs Thm 5: iterate F from I and compare the log-spectrum spread."""
    def spread(P):
        w = np.linalg.eigvalsh(P)
        w = w[w > 1e-300]
        return np.log(w.max() / w.min()) if w.size > 1 else 0.0

    d = 8
    g = np.random.default_rng(0)
    S = g.normal(size=(d, d)) / np.sqrt(d)
    D = g.normal(size=(d, d)) / np.sqrt(d)
    lin, non = np.eye(d), np.eye(d)
    for _ in range(30):
        lin = S @ lin @ S.T
        lin = lin / np.trace(lin) * d
        non = D @ np.diag(np.diag(non)) @ D.T
        non = non / np.trace(non) * d
    early_non = D @ np.diag(np.diag(np.eye(d))) @ D.T
    early_non = early_non / np.trace(early_non) * d
    assert spread(lin) > 20.0                        # linear branch has spread wide open
    assert abs(spread(non) - spread(early_non)) < 2.0   # nonlinear branch has converged


def test_schur_horn_contraction():
    """Thm 6: diag(X) is majorized by lambda(X); equal trace, smaller Frobenius norm."""
    rng = np.random.default_rng(1)
    for _ in range(5000):
        d = int(rng.integers(2, 9))
        A = rng.normal(size=(d, d))
        X = A + A.T
        dg = np.sort(np.diag(X))[::-1]
        lam = np.sort(np.linalg.eigvalsh(X))[::-1]
        assert dg.sum() == pytest.approx(lam.sum(), abs=1e-9)
        assert np.all(np.cumsum(dg) <= np.cumsum(lam) + 1e-9)
        assert np.linalg.norm(dg) <= np.linalg.norm(lam) + 1e-9


def test_gate_cross_product_is_the_agreement_indicator():
    """Lemma 8: D(e)^T D(e') = diag(1[e = e']); the self case recovers D^T D = I."""
    def gate(e):
        return torch.cat([torch.diag((e > 0).double()), -torch.diag((e < 0).double())], 0)

    g = torch.Generator().manual_seed(0)
    for _ in range(500):
        d = int(torch.randint(2, 9, (1,), generator=g))
        e1 = torch.randint(0, 2, (d,), generator=g).double() * 2 - 1
        e2 = torch.randint(0, 2, (d,), generator=g).double() * 2 - 1
        assert torch.allclose(gate(e1).T @ gate(e2), torch.diag((e1 == e2).double()))
        assert torch.allclose(gate(e1).T @ gate(e1), torch.eye(d, dtype=torch.float64))


def test_agreement_probability_is_near_uniform_across_units():
    """Thm 9's hypothesis: pi ~ 1/2 with small unit-to-unit spread, so the drift is a scale."""
    from olo.models.crelu_mlp import CReLUMLP

    g = torch.Generator().manual_seed(0)
    net = CReLUMLP(d_in=12, d_out=12, width=12, depth=8).double()
    net.initialize("looks_linear", seed=0)
    X = torch.randn(512, 12, generator=g, dtype=torch.float64)
    for z in net.pre_activations(X)[:3]:
        s = torch.sign(z)
        pi = (s.unsqueeze(0) == s.unsqueeze(1)).double().mean(dim=(0, 1))
        assert abs(float(pi.mean()) - 0.5) < 0.02
        assert float(pi.std()) < 0.01          # near-uniform => drift is a rescaling


def test_context_context_covariance_vanishes_under_independence():
    """Thm 10 term (ii): E[At^T Gbar Bt^T] = 0 when At, Bt are independent and centred."""
    rng = np.random.default_rng(0)
    d = 4
    Sa, Da = rng.normal(size=(d, d)), rng.normal(size=(d, d))
    Sb, Db = rng.normal(size=(d, d)), rng.normal(size=(d, d))
    Gb = rng.normal(size=(d, d))
    vals = []
    for n in (10**3, 10**5):
        ea = rng.choice([-1.0, 1.0], size=(n, d))
        eb = rng.choice([-1.0, 1.0], size=(n, d))
        A = Sa[None] + Da[None] * ea[:, None, :]
        B = Sb[None] + Db[None] * eb[:, None, :]
        At, Bt = A - A.mean(0), B - B.mean(0)
        vals.append(np.abs(np.einsum("nji,jk,nlk->il", At, Gb, Bt) / n).max())
    # exactly zero, so the sample estimate must shrink with n rather than plateau
    assert vals[1] < vals[0] / 3
