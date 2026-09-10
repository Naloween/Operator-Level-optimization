"""The gain-exponent certificate: Lemma A, Lemma B, and the composed bound.

The two lemmas are elementary enough to check by hand, so these tests are here to catch
transcription errors and to exercise the claims at scale: Lemma A over 200k random
instances, Lemma B on real networks including the case where it is tight.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from olo.models.crelu_mlp import CReLUMLP
from olo.models.deep_linear import DeepLinear
from olo.theory.imbalance import (
    balanced_exponent,
    gain_certificate,
    gain_floor,
    mode_gains,
    slope_bound,
)
from olo.theory.modes import gates_of_pattern, random_modes


# -- Lemma A: a band on y bounds the regression slope -----------------------


def test_lemma_a_holds_over_many_random_instances():
    """|slope| <= w / (2 sd(x)) whenever every y lies in a band of width w."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200_000):
        n = int(rng.integers(3, 20))
        x = rng.normal(size=n) * rng.uniform(0.1, 5)
        w = rng.uniform(1e-3, 10)
        lo = rng.normal() * 3
        y = lo + rng.uniform(0, w, size=n)
        xc = x - x.mean()
        if xc @ xc < 1e-12:
            continue
        slope = xc @ (y - y.mean()) / (xc @ xc)
        assert abs(slope) <= slope_bound(x, lo, lo + w) + 1e-12
        worst = max(worst, abs(slope) / slope_bound(x, lo, lo + w))
    assert worst > 0.5          # the bound is approached, so it is not vacuous


def test_lemma_a_is_tight_up_to_the_factor_two():
    """Two points at the band's ends realise slope = w / (x range), i.e. w/(2 sd)."""
    x = np.array([-1.0, 1.0])
    b = slope_bound(x, 0.0, 2.0)
    slope = (2.0 - 0.0) / (1.0 - (-1.0))
    assert b == pytest.approx(slope)


def test_lemma_a_is_infinite_without_spread_in_x():
    assert np.isinf(slope_bound(np.ones(5), 0.0, 1.0))


# -- Lemma B: an assumption-free floor under the gain -----------------------


def _pattern_gain(net, eps):
    X = torch.zeros(1, net.d_in, dtype=torch.float64)
    gates = gates_of_pattern(net, eps)
    with torch.no_grad():
        J = net.operator(X, gates)[0].double()
        U, S, Vh = torch.linalg.svd(J)
        V = Vh.T
        ctx = net.all_contexts(X, gates)
        k = S.shape[0]
        c = torch.zeros(k, dtype=torch.float64)
        for l in range(net.depth):
            A, B = ctx[l][0][0].double(), ctx[l][1][0].double()
            c += (A.T @ U).pow(2).sum(0)[:k] * (B @ V).pow(2).sum(0)[:k]
    return S.numpy(), c.numpy()


def _norms(net):
    with torch.no_grad():
        return [float(torch.linalg.matrix_norm(W.detach(), ord=2)) for W in net.weights]


@pytest.mark.parametrize("init", ["xavier", "haar", "looks_linear"])
@pytest.mark.parametrize("depth", [4, 16])
def test_lemma_b_floor_is_never_violated(init, depth):
    """`c_k >= L s_k^2 g^{-2/L}` on real CReLU networks, at random gate patterns."""
    for seed in range(4):
        net = CReLUMLP(d_in=8, d_out=8, width=8, depth=depth).double()
        net.initialize(init, seed=seed)
        gen = torch.Generator().manual_seed(seed + 4)
        for _ in range(4):
            s, c = _pattern_gain(net, random_modes(net, 1, gen)[0])
            m = s > 1e-11
            assert np.all(c[m] >= gain_floor(s[m], _norms(net), depth) - 1e-12)


def test_lemma_b_is_tight_for_an_orthogonal_chain():
    """Every layer orthogonal: q_l = 1, g = 1, c_k = L -- the bound is an equality."""
    net = DeepLinear(d_in=6, d_out=6, width=6, depth=5).double()
    net.initialize("haar", seed=0)
    mg = mode_gains(net, torch.eye(6, dtype=torch.float64))
    assert np.allclose(mg.gain, gain_floor(mg.s, _norms(net), 5), rtol=1e-9)
    assert np.allclose(mg.gain, 5.0, rtol=1e-9)


# -- the composed certificate ----------------------------------------------


@pytest.mark.parametrize("init", ["xavier", "haar", "looks_linear"])
@pytest.mark.parametrize("depth", [4, 8, 32])
def test_certificate_holds_measured_and_proved(init, depth):
    """The certified bound is never violated, in either the measured or a-priori form."""
    d = 8
    for seed in range(3):
        net = CReLUMLP(d_in=d, d_out=d, width=d, depth=depth).double()
        net.initialize(init, seed=seed)
        gen = torch.Generator().manual_seed(seed + 4)
        if init == "looks_linear":                # otherwise the spectrum is flat
            with torch.no_grad():
                a = torch.linspace(-0.5, 0.5, d, dtype=torch.float64) * 2.0
                for W in net.weights:
                    W.copy_(torch.diag(torch.exp(a / depth)) @ W)
        for _ in range(4):
            s, c = _pattern_gain(net, random_modes(net, 1, gen)[0])
            for norms in (None, _norms(net)):
                cert = gain_certificate(s, c, depth, norms)
                if np.isfinite(cert.bound):
                    assert cert.holds
                    assert cert.floor_is_proved == (norms is not None)


def test_certificate_is_sharp_at_looks_linear():
    """Near the linear manifold the certificate proves the exponent is essentially 2-2/L."""
    d, depth = 10, 16
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=depth).double()
    net.initialize("looks_linear", seed=0)
    with torch.no_grad():
        a = torch.linspace(-0.5, 0.5, d, dtype=torch.float64) * 2.0
        for W in net.weights:
            W.copy_(torch.diag(torch.exp(a / depth)) @ W)
    s, c = _pattern_gain(net, random_modes(net, 1, torch.Generator().manual_seed(1))[0])
    cert = gain_certificate(s, c, depth)
    assert cert.holds
    assert cert.bound < 0.1                       # a genuinely small certified interval
    assert abs(cert.exponent - balanced_exponent(depth)) < 0.05


def test_certificate_reports_nan_when_too_few_directions_survive():
    cert = gain_certificate(np.array([1.0, 0.0]), np.array([2.0, 0.0]), 2)
    assert np.isnan(cert.bound)
