"""The separation expansion, checked symbolically and against the exact flow."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from olo.theory.balanced import bias_exponent
from olo.theory.instability import (
    Amplification,
    amplification,
    balanced_flow,
    drive,
    feedback_rate,
    fit_scale,
    log_velocity_exponent,
    predict_spectrum,
    scale_separation_law,
    separation,
    separation_trajectory,
    spectral_flow,
)


def test_log_velocity_exponent_is_bias_exponent_minus_one():
    """`omega = s'/s` drops one power: `phi = (2 - 2/L) - 1`."""
    for L in (2, 8, 64, 1024):
        assert log_velocity_exponent(L) == pytest.approx(bias_exponent(L) - 1.0)


def test_expansion_is_the_first_order_taylor_series():
    """Symbolic: `r' = D + lambda r + O(r^2)` with the stated D and lambda."""
    r, s, gj, gk, L = sp.symbols("r s g_j g_k L", positive=True)
    phi = 1 - 2 / L
    sj, sk = s * sp.exp(r / 2), s * sp.exp(-r / 2)
    rdot = -L * (gj * sj**phi - gk * sk**phi)  # omega_j - omega_k

    D = sp.simplify(rdot.subs(r, 0))
    assert sp.simplify(D - (-L * s**phi * (gj - gk))) == 0

    lam = sp.simplify(sp.diff(rdot, r).subs(r, 0))
    assert sp.simplify(lam - (-(L - 2) * s**phi * (gj + gk) / 2)) == 0


def test_feedback_vanishes_at_depth_two_and_at_isotropy():
    """No feedback at L=2 (phi=0), and no drive when the gradient is isotropic."""
    assert feedback_rate(1.0, -0.3, depth=2) == 0.0
    assert drive(1.0, 0.0, depth=64) == 0.0
    # ...but the rate at depth 64 is 62x a unit gradient, at an isometry.
    assert feedback_rate(1.0, -1.0, depth=64) == pytest.approx(62.0)


def test_feedback_sign_follows_growing_modes():
    """Separation amplifies while modes grow (gbar<0) and damps while they shrink."""
    assert feedback_rate(1.0, gbar=-0.5, depth=32) > 0
    assert feedback_rate(1.0, gbar=+0.5, depth=32) < 0


def test_linearization_matches_the_exact_flow_for_small_separation():
    """Integrate `s' = -L s^{2-2/L} g` and compare r(t) with the closed form."""
    L, g, r0 = 16, np.array([-0.05, -0.05]), 0.02
    s0 = np.array([np.exp(r0 / 2), np.exp(-r0 / 2)])
    traj = balanced_flow(s0, g, depth=L, dt=1e-3, steps=400)
    t = np.arange(traj.shape[0]) * 1e-3
    r_exact = np.log(traj[:, 0] / traj[:, 1])
    r_lin = separation_trajectory(r0, 1.0, g[0], g[1], L, t)
    assert np.max(np.abs(r_exact - r_lin) / np.abs(r_exact)) < 0.05


def test_separation_grows_exponentially_at_the_predicted_rate():
    """The measured growth rate of log(s1/s2) matches `-(L-2) gbar s^phi`."""
    for L in (8, 32, 128):
        g = np.array([-0.02, -0.02])
        s0 = np.array([np.exp(0.005), np.exp(-0.005)])
        traj = balanced_flow(s0, g, depth=L, dt=1e-4, steps=200)
        t = np.arange(traj.shape[0]) * 1e-4
        r = np.log(traj[:, 0] / traj[:, 1])
        measured = np.polyfit(t, np.log(r), 1)[0]  # exponential growth of r
        assert measured == pytest.approx(feedback_rate(1.0, -0.02, L), rel=0.05)


def test_isotropic_spectrum_is_a_fixed_point_of_separation():
    """Exactly equal singular values stay equal, at every depth, under isotropic g."""
    for L in (4, 64, 1024):
        traj = balanced_flow(np.ones(5), np.full(5, -0.1), depth=L, dt=1e-5, steps=50)
        assert separation(traj[-1]) < 1e-12


def test_amplification_rate_recovers_the_flow_rate():
    """The regression of omega on log s returns exactly the feedback rate."""
    L = 32
    s = np.array([1.6, 1.2, 1.0, 0.8, 0.5])
    g = np.full(5, -0.02)
    ds = -L * s ** bias_exponent(L) * g
    a = amplification(s, ds)
    # omega = -L g s^phi is exactly exponential in log s, so a linear fit is approximate;
    # over this spread it recovers the local rate at the geometric mean to a few percent.
    s_bar = float(np.exp(np.log(s).mean()))
    assert a.rate == pytest.approx(feedback_rate(s_bar, -0.02, L), rel=0.1)
    # R^2 falls short of 1 by the curvature of exp(phi*m) over a 3.2x spread, not by noise.
    assert a.rate > 0 and a.r2 > 0.95


def test_amplification_is_undefined_on_an_isometric_spectrum():
    """No spread in log s means no rate to identify -- which is the looks-linear start."""
    a = amplification(np.ones(6), np.full(6, -0.3))
    assert np.isnan(a.rate)
    assert a.uniform == pytest.approx(-0.3)


def test_amplification_splits_linearly_over_self_and_cross():
    """b = b_self + b_cross, so the split is a decomposition of the bias itself."""
    rng = np.random.default_rng(0)
    s = np.exp(rng.normal(size=8))
    a_self, a_cross = rng.normal(size=8), rng.normal(size=8)
    a = amplification(s, a_self + a_cross, a_self, a_cross)
    assert a.rate == pytest.approx(a.rate_self + a.rate_cross)


def test_uniform_log_velocity_changes_no_ratio():
    """A velocity proportional to s rescales the operator and leaves separation fixed."""
    s = np.array([3.0, 1.0, 0.4])
    a = amplification(s, 0.7 * s)
    assert a.rate == pytest.approx(0.0, abs=1e-12)
    assert a.uniform == pytest.approx(0.7)
    assert separation(s) == pytest.approx(np.log(7.5))


def test_cross_share_is_nan_without_a_measured_bias():
    assert np.isnan(Amplification(0.0, 0.0, 0.0, 1.0, 4).cross_share)


# -- the integrated law -----------------------------------------------------


def test_flow_map_reproduces_the_integrated_ode():
    """`T_c` is the exact solution of `s' = -L g s^{2-2/L}` for mode-independent g."""
    for L in (4, 16, 256):
        s0 = np.array([1.7, 1.0, 0.6, 0.25])
        g, dt, steps = -0.03 / L, 1e-3, 500  # keep c away from the finite-time blowup
        traj = balanced_flow(s0, np.full(4, g), depth=L, dt=dt, steps=steps)
        c = L * g * dt * steps  # the Box-Cox translation is L*gamma*t, with no psi factor
        assert np.allclose(traj[-1], spectral_flow(s0, c, L), rtol=1e-6)


def test_fit_scale_recovers_c_from_the_geometric_mean_alone():
    """One scalar read off the mean predicts the whole spectrum, with no per-mode fit."""
    L, s0, c = 32, np.array([1.4, 1.1, 0.9, 0.7, 0.45]), -0.4
    s1 = spectral_flow(s0, c, L)
    assert fit_scale(s0, s1, L) == pytest.approx(c, rel=1e-9)
    assert np.allclose(spectral_flow(s0, fit_scale(s0, s1, L), L), s1, rtol=1e-9)


def test_scale_separation_law_is_the_first_order_flow_map():
    """r(t)/r(0) = (growth)^(1-2/L), to first order in the separation."""
    for L in (8, 64, 1024):
        r0 = 1e-3
        s0 = np.array([np.exp(r0 / 2), np.exp(-r0 / 2)])
        s1 = spectral_flow(s0, -0.3, L)
        measured = np.log(s1[0] / s1[1]) / r0
        assert measured == pytest.approx(scale_separation_law(s0, s1, L), rel=1e-5)


def test_separation_is_frozen_when_the_operator_does_not_grow():
    """No scale growth, no bias -- at any depth. Growth is the clock, not time."""
    for L in (4, 1024):
        s0 = np.array([1.3, 1.0, 0.7])
        assert scale_separation_law(s0, s0, L) == pytest.approx(1.0)
        assert np.allclose(spectral_flow(s0, 0.0, L), s0)


def test_amplification_saturates_in_depth():
    """256 and 1024 differ by well under a percent: no depth turns the feedback on."""
    s0, s1 = np.array([1.1, 0.9]), np.array([3.3, 2.7])  # a 3x growth of scale
    a, b = scale_separation_law(s0, s1, 256), scale_separation_law(s0, s1, 1024)
    assert abs(a / b - 1) < 0.007
    # ...while depth 2 has no feedback at all: separation is untouched by growth.
    assert scale_separation_law(s0, s1, 2) == pytest.approx(1.0)


def test_law_is_multiplicative_in_the_seed():
    """Doubling the initial anisotropy doubles the final one: the seed never washes out."""
    L, errs = 64, []
    for r0 in (1e-4, 1e-3, 1e-2, 1e-1):
        s0 = np.array([np.exp(r0 / 2), np.exp(-r0 / 2)])
        s1 = spectral_flow(s0, -0.5, L)
        ratio = np.log(s1[0] / s1[1]) / r0
        pred = scale_separation_law(s0, s1, L)
        assert ratio == pytest.approx(pred, rel=0.02)
        errs.append(abs(ratio / pred - 1))
    # The residual is second order in the seed: three decades of r0 give six of error.
    assert np.polyfit(np.log([1e-4, 1e-3, 1e-2, 1e-1]), np.log(errs), 1)[0] == pytest.approx(2.0, abs=0.1)


# -- architecture vs task ---------------------------------------------------


def test_psi_splits_into_depth_and_task():
    from olo.theory.instability import amplification_exponent

    assert amplification_exponent(64, 0.0) == pytest.approx(1 - 2 / 64)
    # A pure-rescale target (g_k proportional to s_k) adds 1 to the exponent.
    assert amplification_exponent(64, 1.0) == pytest.approx(2 - 2 / 64)
    # ...and a task can cancel the depth bias exactly.
    assert amplification_exponent(64, -(1 - 2 / 64)) == pytest.approx(0.0)


def test_measured_psi_recovers_the_task_exponent():
    """Build `omega_k = -L gamma s_k^{phi+p}` and read `p` back off the regression."""
    L = 32
    s = np.exp(np.linspace(-0.1, 0.1, 12))  # a small spread, where the fit is local
    for p in (-1.0, 0.0, 1.0):
        psi = log_velocity_exponent(L) + p
        ds = -0.01 * s**psi * s  # omega = -0.01 s^psi
        a = amplification(s, ds)
        assert a.psi == pytest.approx(psi, rel=0.02)
        assert a.task_exponent(L) == pytest.approx(p, abs=0.02)


def test_a_task_can_cancel_the_depth_bias():
    """At p = -phi the log-velocity is mode independent: growth without separation."""
    L = 64
    s = np.array([2.0, 1.0, 0.5, 0.25])
    s1 = spectral_flow(s, -0.3, L, task_exponent=-log_velocity_exponent(L))
    assert np.allclose(s1 / s, (s1 / s)[0])           # a pure rescale
    assert separation(s1) == pytest.approx(separation(s))


def test_predict_spectrum_uses_one_number_to_get_the_rest():
    """Fit the scale, predict every other mode; exact when the law holds."""
    L, s0 = 16, np.array([1.5, 1.1, 0.9, 0.6, 0.3])
    for p in (0.0, 1.0):
        s1 = spectral_flow(s0, -0.2, L, p)
        assert np.allclose(predict_spectrum(s0, s1, L, p), s1, rtol=1e-9)


def test_shrinking_the_operator_undoes_separation():
    """The law runs backwards: c > 0 shrinks the scale and contracts the spectrum."""
    L, s0 = 32, np.array([2.0, 1.0, 0.5])
    grown = spectral_flow(s0, -0.2, L)
    assert separation(grown) > separation(s0)
    assert scale_separation_law(s0, grown, L) > 1
    shrunk = spectral_flow(s0, +0.2, L)
    assert separation(shrunk) < separation(s0)
    assert scale_separation_law(s0, shrunk, L) < 1


# -- Theorem 16: where the seed comes from ----------------------------------


def _looks_linear_net(depth=5, width=6, seed=0):
    from olo.models.crelu_mlp import CReLUMLP

    net = CReLUMLP(d_in=width, d_out=width, width=width, depth=depth).double()
    net.initialize("looks_linear", seed=seed)
    return net


def test_seed_theorem_blocks_from_the_gate_split():
    """The exact block identity of Theorem 16, read off one real gradient step."""
    import torch

    from olo.theory.crelu import split_layer

    net, B, w = _looks_linear_net(depth=4, width=6), 9, 6
    g = torch.Generator().manual_seed(11)
    X = torch.randn(B, w, generator=g, dtype=torch.float64)
    G = torch.randn(B, w, w, generator=g, dtype=torch.float64)

    J = net.operator(X)
    ((G * J).sum() / B).backward()
    ctx = net.all_contexts(X)
    zs = net.pre_activations(X)

    for l in range(1, net.depth):
        A, Bc = ctx[l]
        Ab = A.expand(B, *A.shape[1:]) if A.shape[0] == 1 else A
        Bb = Bc.expand(B, *Bc.shape[1:]) if Bc.shape[0] == 1 else Bc
        # R_b = A^T G_b N^T, with N recovered exactly as D(z_b)^T B_l(x_b) (D^T D = I).
        Dz = net.gates(X)[l - 1]
        Nb = torch.einsum("bij,bik->bjk", Dz, Bb)
        R = torch.einsum("bji,bjk,blk->bil", Ab, G, Nb)
        E = torch.stack([torch.diag(torch.sign(zs[l - 1][b])) for b in range(B)])

        dS, dD = split_layer(net.weights[l].grad)
        assert torch.allclose(dS, R.mean(0) / 2, atol=1e-12)
        assert torch.allclose(dD, torch.einsum("bij,bjk->ik", R, E) / (2 * B), atol=1e-12)


def test_seed_vanishes_when_the_gate_signs_average_out():
    """A force shared across inputs makes dDelta exactly R * diag(2*phat - 1)."""
    import torch

    from olo.theory.crelu import split_layer

    net, w = _looks_linear_net(depth=3, width=4), 4
    g = torch.Generator().manual_seed(5)
    # Sign-symmetrized batch: every x appears with -x, so each gate frequency is exactly
    # 1/2 and the seed is exactly zero -- the finite-batch fluctuation removed by hand.
    half = torch.randn(16, w, generator=g, dtype=torch.float64)
    X = torch.cat([half, -half])
    G = torch.eye(w, dtype=torch.float64).expand(X.shape[0], w, w)

    J = net.operator(X)
    ((G * J).sum() / X.shape[0]).backward()
    for l in range(1, net.depth):
        _, dD = split_layer(net.weights[l].grad)
        assert float(dD.abs().max()) < 1e-12


def test_symmetrized_batch_keeps_the_network_exactly_linear():
    """Theorem 17: a batch closed under negation leaves the looks-linear manifold fixed."""
    import torch

    from olo.tasks.base import mse
    from olo.theory.crelu import split_layer

    w, L, mu = 6, 5, 1.0                       # mu > 0: a sign-asymmetric input law
    g = torch.Generator().manual_seed(0)
    P_star = torch.linalg.qr(torch.randn(w, w, generator=g, dtype=torch.float64))[0] * 2

    finals = {}
    for arm in ("raw", "symmetrized"):
        net = _looks_linear_net(depth=L, width=w)
        opt = torch.optim.SGD(net.parameters(), lr=1e-3 / L)
        gg = torch.Generator().manual_seed(7)
        for _ in range(60):
            X = torch.randn(32, w, generator=gg, dtype=torch.float64) + mu
            if arm == "symmetrized":
                X = torch.cat([X, -X])
            opt.zero_grad(set_to_none=True)
            mse(net(X), X @ P_star.T).backward()
            opt.step()
        with torch.no_grad():
            finals[arm] = max(float(split_layer(W)[1].abs().max())
                              for W in net.weights[1:])

    assert finals["symmetrized"] < 1e-14      # machine zero: Delta never leaves 0
    assert finals["raw"] > 1e-4               # ...and without it, the seed is systematic


def test_drive_is_nonzero_at_an_isometry_when_the_gradient_is_anisotropic():
    """The hypothesis in Cor. 13.1 is load-bearing: without it, an isometry is not fixed."""
    s = np.ones(4)
    ds = -np.array([1.0, 0.5, -0.2, 0.3])   # anisotropic gradient at an isometric spectrum
    a = amplification(s, ds)
    assert np.isnan(a.rate)                 # no spread in log s, so no rate is identifiable
    # ...but the velocities differ across modes, so separation is about to appear.
    assert ds.std() > 0
    assert drive(1.0, ds[0] - ds[1], depth=64) != 0.0
