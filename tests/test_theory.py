"""The closed form must match a measured step exactly, or it predicts nothing.

These tests are the bridge between the analysis and the harness: if `theory.alignment`
agrees with what the runner measures to machine precision, then the scaling laws derived
from it (usable lr ~ 1/L, collapse onto tau = L eta ||G||) are statements about the
experiments rather than about a model of them.
"""
from __future__ import annotations

import pytest
import torch

from olo import theory
from olo.diagnostics.mismatch import gd_first_order_alignment, step_alignment
from olo.models import DeepLinear
from olo.optim.baselines.torch_optims import HeavyBall
from olo.tasks.teacher_student import TeacherStudent


def _identity_setup(depth, d=8, seed=0):
    torch.manual_seed(seed)
    net = DeepLinear(d_in=d, d_out=d, width=d, depth=depth).to(torch.float64)
    net.initialize("identity", seed)
    task = TeacherStudent(d=d, n=16, seed=seed).to(torch.device("cpu"), torch.float64)
    return net, task


@pytest.mark.parametrize("depth", [2, 8, 32, 128])
@pytest.mark.parametrize("eta", [1e-4, 1e-3, 1e-2])
def test_closed_form_matches_the_measured_step(depth, eta):
    """(I - eta G)^L - I is the exact operator increment, not an approximation."""
    net, task = _identity_setup(depth)
    x, y = task.train_batch(0)
    G = task.operator_gradient(net, x, y)[0]
    with torch.no_grad():
        P0 = net.operator(x)[0].clone()

    HeavyBall(net, lr=eta, momentum=0.0).step(x, y, task)
    with torch.no_grad():
        P1 = net.operator(x)[0]

    predicted = theory.gd_operator_step(G, eta, depth)
    assert torch.allclose(predicted, P1 - P0, atol=1e-12), float((predicted - (P1 - P0)).norm())

    measured = step_alignment(P0.unsqueeze(0), P1.unsqueeze(0), G.unsqueeze(0), eta)["cos_target"]
    assert measured == pytest.approx(theory.alignment(G, eta, depth), abs=1e-12)


@pytest.mark.parametrize("depth", [2, 8, 32, 128])
def test_first_order_alignment_is_blind_to_the_real_mismatch(depth):
    """At identity init the first-order term is exactly parallel to -G at every depth.

    So Proposition 3.1 alone reports perfect alignment here, while the realized step is
    misaligned by its higher-order terms. Reporting only the first-order quantity would
    mean reporting "no mismatch" for a network that has one.
    """
    net, task = _identity_setup(depth)
    x, y = task.train_batch(0)
    G = task.operator_gradient(net, x, y)

    first = gd_first_order_alignment(net.all_contexts(x), G)
    assert first["cos_gd_first_order"] == pytest.approx(1.0, abs=1e-10)

    realized = theory.alignment(G[0], eta=1e-2, depth=depth)
    if depth >= 32:
        assert realized < 0.999, (depth, realized)


def test_mismatch_collapses_onto_L_times_eta():
    """Depth and step size enter only through tau = L eta ||G||.

    Two configurations with matched tau must show the same alignment even though their
    depths differ by 8x -- which is what makes tau a design rule rather than a curve fit.
    """
    net, task = _identity_setup(32)
    x, y = task.train_batch(0)
    G = task.operator_gradient(net, x, y)[0]

    scale = float(torch.linalg.matrix_norm(G, ord=2))
    tau = 0.5
    a = theory.alignment(G, eta=tau / (4 * scale), depth=4)
    b = theory.alignment(G, eta=tau / (32 * scale), depth=32)
    c = theory.alignment(G, eta=tau / (256 * scale), depth=256)
    assert a == pytest.approx(b, rel=2e-2), (a, b)
    assert b == pytest.approx(c, rel=2e-2), (b, c)


def test_usable_learning_rate_scales_as_one_over_depth():
    """`max_stable_lr` halves when depth doubles, and holds alignment fixed."""
    net, task = _identity_setup(16)
    x, y = task.train_batch(0)
    G = task.operator_gradient(net, x, y)[0]

    prev_lr, prev_cos = None, None
    for depth in (16, 32, 64, 128):
        lr = theory.max_stable_lr(G, depth, tol=0.2)
        cos = theory.alignment(G, lr, depth)
        if prev_lr is not None:
            assert lr == pytest.approx(prev_lr / 2, rel=1e-9), depth
            assert cos == pytest.approx(prev_cos, rel=5e-2), depth
        prev_lr, prev_cos = lr, cos


def test_alignment_degrades_monotonically_in_tau():
    net, task = _identity_setup(64)
    x, y = task.train_batch(0)
    G = task.operator_gradient(net, x, y)[0]
    etas = [1e-5, 1e-4, 1e-3, 1e-2]
    taus, cosines = theory.alignment_curve(G, 64, etas)
    assert taus == sorted(taus)
    assert cosines == sorted(cosines, reverse=True), cosines
