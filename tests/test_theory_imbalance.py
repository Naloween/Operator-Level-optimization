"""What survives when balancedness is dropped: the exponent, unless imbalance varies by mode."""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
import torch

from olo.models.deep_linear import DeepLinear
from olo.theory.imbalance import (
    ModeGain,
    balanced_exponent,
    imbalance_factor_from_scales,
    imbalance_invariants,
    imbalance_norm,
    mode_gains,
    trajectory_exponent_from_scales,
)

D = 16


def _net(w: torch.Tensor, spread: float = 2.0, width: int = D) -> DeepLinear:
    """Diagonal deep linear net. `w` is (L, width): per-layer, per-mode log scale offsets.

    Diagonal by construction, so alignment holds exactly and *only* balancedness varies --
    the confound is removed rather than argued away.
    """
    L = w.shape[0]
    a = torch.linspace(-0.5, 0.5, width, dtype=torch.float64) * spread
    net = DeepLinear(d_in=width, d_out=width, width=width, depth=L).double()
    with torch.no_grad():
        for l, W in enumerate(net.weights):
            W.copy_(torch.diag(torch.exp(a / L + w[l])))
    return net


def _slopes(net) -> tuple[float, float]:
    return mode_gains(net, torch.eye(net.width, dtype=torch.float64)).exponent()


# -- the decomposition ------------------------------------------------------


def test_gain_factors_into_the_balanced_part_times_K():
    """`c_k = s_k^{2-2/L} K_k` is a definition; check the measured pieces satisfy it."""
    mg = mode_gains(_net(torch.zeros(6, D)), torch.eye(D, dtype=torch.float64))
    assert np.allclose(mg.gain, mg.s ** balanced_exponent(6) * mg.imbalance_factor)


def test_K_is_at_least_L_with_equality_only_at_balance():
    """prod_l x_l = 1 by construction, so AM-GM floors K at L."""
    rng = np.random.default_rng(0)
    for L in (2, 5, 16):
        assert imbalance_factor_from_scales(np.full(L, 0.7)) == pytest.approx(L)
        for _ in range(50):
            a = np.exp(rng.normal(0, 1.5, size=L))
            assert imbalance_factor_from_scales(a) >= L - 1e-9
            if a.std() > 1e-6:
                assert imbalance_factor_from_scales(a) > L


def test_K_is_scale_free():
    a = np.array([0.3, 1.7, 0.9, 2.2])
    assert imbalance_factor_from_scales(a) == pytest.approx(
        imbalance_factor_from_scales(a * 37.0)
    )


# -- the result: mode-independent imbalance does not move the exponent ------


def test_balanced_network_has_the_balanced_exponent():
    slope, dK = _slopes(_net(torch.zeros(8, D)))
    assert slope == pytest.approx(balanced_exponent(8), abs=1e-12)
    assert dK == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("kind", ["linear", "bottleneck", "random"])
def test_mode_independent_imbalance_leaves_the_exponent_exact(kind):
    """A layerwise imbalance shared by every mode changes K's level, not its slope."""
    L = 8
    if kind == "linear":
        col = torch.linspace(-0.5, 0.5, L, dtype=torch.float64) * 4.0
    elif kind == "bottleneck":
        col = torch.zeros(L, dtype=torch.float64)
        col[0] = -3.0                                   # a 20x bottleneck layer
    else:
        col = torch.randn(L, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    net = _net(col.unsqueeze(1).expand(L, D).contiguous())

    slope, dK = _slopes(net)
    assert slope == pytest.approx(balanced_exponent(L), abs=1e-10)
    assert dK == pytest.approx(0.0, abs=1e-10)
    # ...and the network really is unbalanced, so this is not a vacuous check.
    assert imbalance_norm(net) > 0.5
    mg = mode_gains(net, torch.eye(D, dtype=torch.float64))
    assert float(np.median(mg.imbalance_factor)) > L * 1.2


def test_mode_dependent_imbalance_moves_the_exponent_by_exactly_dlogK():
    """The decomposition is exact, including when the shift is large and sign-flipping."""
    L = 8
    cases = [
        torch.linspace(-0.5, 0.5, L).unsqueeze(1) * torch.linspace(0, 2.0, D).unsqueeze(0),
        torch.cat([-3.0 * torch.linspace(0, 1, D).unsqueeze(0), torch.zeros(L - 1, D)]),
    ]
    shifts = []
    for w in cases:
        slope, dK = _slopes(_net(w.double()))
        assert slope == pytest.approx(balanced_exponent(L) + dK, abs=1e-10)
        shifts.append(dK)
    assert max(abs(s) for s in shifts) > 1.0          # the correction can be large
    assert min(shifts) < -1.0                          # ...and can flip the exponent's sign


def test_xavier_has_a_weaker_exponent_than_the_balanced_law_predicts():
    """The regime where collapse happens is mode-dependent, and biased *less* than balanced."""
    for L in (4, 8, 16):
        net = DeepLinear(d_in=D, d_out=D, width=D, depth=L).double()
        net.initialize("xavier", seed=0)
        slope, dK = _slopes(net)
        assert slope < balanced_exponent(L)
        assert dK < 0
        assert slope == pytest.approx(balanced_exponent(L) + dK, abs=1e-6)


# -- the other derivative, kept apart on purpose ----------------------------


def test_trajectory_exponent_is_a_different_quantity():
    """`2 - 2/L_eff` is the along-trajectory derivative, and it is NOT the bias exponent."""
    L = 8
    col = torch.zeros(L, dtype=torch.float64)
    col[0] = -3.0
    a = np.exp((torch.zeros(L) + col).numpy())
    traj = trajectory_exponent_from_scales(a)
    assert traj < balanced_exponent(L) - 0.5           # it does move with imbalance
    # ...while the cross-mode exponent, which is what governs separation, does not.
    slope, _ = _slopes(_net(col.unsqueeze(1).expand(L, D).contiguous()))
    assert slope == pytest.approx(balanced_exponent(L), abs=1e-10)


def test_trajectory_exponent_matches_the_invariant_constrained_derivative():
    """Symbolic: along `a_l^2 = t + C_l`, `d log c/d log s = 2 - 2 H_2/H^2`."""
    L = 4
    t = sp.symbols("t", positive=True)
    C = sp.symbols("C1:5", nonnegative=True)
    a2 = [t + C[l] for l in range(L)]
    s, H = sp.sqrt(sp.prod(a2)), sum(1 / x for x in a2)
    H2 = sum(1 / x**2 for x in a2)
    c = s**2 * H
    got = sp.simplify(sp.diff(sp.log(c), t) / sp.diff(sp.log(s), t))
    assert sp.simplify(got - (2 - 2 * H2 / H**2)) == 0


# -- the measurement itself -------------------------------------------------


def test_measured_gain_reproduces_the_induced_step():
    """q is the (k,k) entry of `sum_l A_l A_l^T G B_l^T B_l` under `G = u_k v_k^T`."""
    net = DeepLinear(d_in=4, d_out=4, width=4, depth=4).double()
    net.initialize("xavier", seed=1)
    X = torch.eye(4, dtype=torch.float64)
    mg = mode_gains(net, X)
    with torch.no_grad():
        P = net.operator(X)[0].double()
        U, S, Vh = torch.linalg.svd(P)
        ctx = net.all_contexts(X)
        for k in (0, 2):
            G = torch.outer(U[:, k], Vh[k, :])
            step = sum(A[0].double() @ A[0].double().T @ G @ B[0].double().T @ B[0].double()
                       for A, B in ctx)
            assert float(U[:, k] @ step @ Vh[k, :]) == pytest.approx(mg.gain[k], rel=1e-9)


def test_measured_gain_matches_the_balanced_closed_form():
    net = _net(torch.zeros(5, D), spread=0.0)
    mg = mode_gains(net, torch.eye(D, dtype=torch.float64))
    assert np.allclose(mg.gain, mg.balanced_gain)


def test_invariants_vanish_exactly_for_a_balanced_net():
    net = _net(torch.zeros(4, D), spread=0.0)
    assert max(float(C.abs().max()) for C in imbalance_invariants(net)) < 1e-14
    assert imbalance_norm(net) < 1e-14


def test_exponent_is_nan_when_too_few_modes_survive():
    mg = ModeGain(s=np.array([1.0, 0.0]), q=np.array([[1.0, 1.0], [0.0, 0.0]]), L=2)
    assert np.isnan(mg.exponent()[0])


def test_exponent_is_nan_on_an_isometric_spectrum():
    """looks-linear gives J orthogonal, so no slope exists -- it must not fit noise."""
    from olo.models.crelu_mlp import CReLUMLP

    net = CReLUMLP(d_in=8, d_out=8, width=8, depth=6).double()
    net.initialize("looks_linear", seed=0)
    X = torch.randn(4, 8, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    mg = mode_gains(net, X)
    assert np.ptp(mg.s) < 1e-10                    # the spectrum really is flat
    assert np.isnan(mg.exponent()[0])


def test_imbalance_norm_is_nan_when_the_invariant_does_not_apply():
    """CReLU layers are d x 2d, so Du's invariant has no comparable boundary."""
    from olo.models.crelu_mlp import CReLUMLP

    net = CReLUMLP(d_in=8, d_out=8, width=8, depth=4).double()
    net.initialize("xavier", seed=0)
    assert imbalance_invariants(net) == []
    assert np.isnan(imbalance_norm(net))
