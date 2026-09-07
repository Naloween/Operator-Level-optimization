"""The diagnostics must report what they claim, in cases where the answer is known.

Each test below pins a diagnostic to a configuration whose value can be derived by hand,
so a regression shows up as a wrong number rather than as a plausible-looking curve.
"""
from __future__ import annotations

import pytest
import torch

from olo.diagnostics.conditioning import context_conditioning
from olo.diagnostics.linearization import linearization_validity
from olo.diagnostics.mismatch import gd_first_order_alignment, step_alignment
from olo.diagnostics.spectrum import operator_spectrum
from olo.models import CReLUMLP, DeepLinear, ReLUMLP
from olo.optim.als import OperatorALS
from olo.optim.baselines.torch_optims import HeavyBall
from olo.tasks.teacher_student import TeacherStudent


def _linear(depth, init, d=8, seed=0):
    torch.manual_seed(seed)
    net = DeepLinear(d_in=d, d_out=d, width=d, depth=depth).to(torch.float64)
    net.initialize(init, seed)
    task = TeacherStudent(d=d, n=16, seed=seed).to(torch.device("cpu"), torch.float64)
    return net, task


# -- Proposition 3.1 --------------------------------------------------------


def test_identity_init_has_no_direction_mismatch_only_a_gain_of_L():
    """All contexts equal I, so sum_k A_k A_k^T G B_k^T B_k = L G: right way, L times far.

    This is the sharpest statement that conditioning and direction are separate axes --
    perfectly conditioned contexts give a perfectly aligned gradient step whose only defect
    is its length.
    """
    for depth in (2, 8, 64):
        net, task = _linear(depth, "identity")
        x, y = task.train_batch(0)
        G = task.operator_gradient(net, x, y)
        out = gd_first_order_alignment(net.all_contexts(x), G)
        assert out["cos_gd_first_order"] == pytest.approx(1.0, abs=1e-10), depth
        assert out["gd_gain"] == pytest.approx(float(depth), rel=1e-9), depth


def test_haar_init_has_no_mismatch_either_at_initialization():
    """Orthogonal layers compose to orthogonal contexts, so A A^T = B^T B = I exactly.

    This contradicts the submitted paper's appendix, which attributed the failure of
    gradient methods at Haar initialization to "random orientations reintroducing the
    Gram-matrix mismatch". For square orthogonal layers every context is orthogonal, so
    sum_k A_k A_k^T kron B_k^T B_k = L I on the nose and there is no direction mismatch to
    reintroduce -- at initialization.
    """
    for depth in (2, 16, 128):
        net, task = _linear(depth, "haar")
        x, y = task.train_batch(0)
        G = task.operator_gradient(net, x, y)
        out = gd_first_order_alignment(net.all_contexts(x), G)
        assert out["cos_gd_first_order"] == pytest.approx(1.0, abs=1e-9), depth
        assert out["gd_gain"] == pytest.approx(float(depth), rel=1e-8), depth


def test_mismatch_is_generated_by_training_at_a_rate_set_by_depth():
    """The mismatch is dynamical, not a property of the initialization.

    Starting from perfectly aligned contexts, gradient descent itself destroys the
    alignment, and deeper networks lose it faster: after 200 identical steps the cosine
    falls to ~0.97 at L=2, ~0.65 at L=16 and ~0.52 at L=128. That is the phenomenon this
    project is about, and it is invisible if the cosine is only measured at step 0.
    """
    final = {}
    for depth in (2, 16, 128):
        net, task = _linear(depth, "haar")
        opt = HeavyBall(net, lr=1e-3, momentum=0.0)
        for s in range(200):
            opt.step(*task.train_batch(s), task)
        x, y = task.train_batch(200)
        G = task.operator_gradient(net, x, y)
        final[depth] = gd_first_order_alignment(net.all_contexts(x), G)["cos_gd_first_order"]

    assert final[2] > final[16] > final[128], final
    assert final[2] > 0.95, final
    assert final[128] < 0.7, final


def test_als_realizes_the_operator_step_and_gradient_descent_does_not():
    """cos_target ~ 1 for the projection; strictly below for a parameter-space step."""
    results = {}
    for name, make in (("als", lambda n: OperatorALS(n, lr=0.05, lam=1e-10, n_sweeps=4)),
                       ("heavyball", lambda n: HeavyBall(n, lr=0.05, momentum=0.0))):
        net, task = _linear(32, "haar")
        x, y = task.train_batch(0)
        G = task.operator_gradient(net, x, y)
        with torch.no_grad():
            P0 = net.operator(x).clone()
        opt = make(net)
        opt.step(x, y, task)
        with torch.no_grad():
            P1 = net.operator(x)
        results[name] = step_alignment(P0, P1, G, opt.lr)["cos_target"]

    assert results["als"] > 0.999, results
    assert results["heavyball"] < results["als"], results


# -- the collapse monitor ---------------------------------------------------


def test_monitor_flags_xavier_collapse_and_clears_looks_linear():
    """The two regimes the project contrasts must be distinguishable by rho alone."""
    torch.manual_seed(0)
    d, depth, lam = 8, 64, 1e-4
    x = torch.randn(8, d, dtype=torch.float64)

    collapsed = ReLUMLP(d_in=d, d_out=d, width=d, depth=depth).to(torch.float64)
    collapsed.initialize("xavier", 0)
    healthy = CReLUMLP(d_in=d, d_out=d, width=d, depth=depth).to(torch.float64)
    healthy.initialize("looks_linear", 0)

    bad, _ = context_conditioning(collapsed.all_contexts(x), lam)
    good, _ = context_conditioning(healthy.all_contexts(x), lam)

    assert bad["collapsed_fraction"] > 0.5, bad
    assert good["collapsed_fraction"] == 0.0, good
    assert good["context_min_sv"] == pytest.approx(1.0, abs=1e-6), good


def test_looks_linear_contexts_are_perfectly_conditioned_at_any_depth():
    torch.manual_seed(0)
    d = 8
    x = torch.randn(4, d, dtype=torch.float64)
    for depth in (2, 32, 256):
        net = CReLUMLP(d_in=d, d_out=d, width=d, depth=depth).to(torch.float64)
        net.initialize("looks_linear", 0)
        s, _ = context_conditioning(net.all_contexts(x), 1e-4)
        assert s["context_cond_max"] == pytest.approx(1.0, abs=1e-6), (depth, s)


# -- spectrum and linearization --------------------------------------------


def test_orthogonal_operator_has_full_effective_rank():
    torch.manual_seed(0)
    d = 8
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=16).to(torch.float64)
    net.initialize("looks_linear", 0)
    s, arr = operator_spectrum(net.operator(torch.randn(4, d, dtype=torch.float64)))
    assert s["effective_rank"] == pytest.approx(float(d), rel=1e-6)
    assert s["condition"] == pytest.approx(1.0, abs=1e-8)
    assert arr["singular_values"].shape == (d,)


def test_linearization_is_exact_for_input_independent_models():
    net, task = _linear(4, "haar")
    x, _ = task.train_batch(0)
    out = linearization_validity(net, x, net.gates(x))
    assert out["gate_flip_rate"] == 0.0
    assert out["frozen_gate_error"] == 0.0


def test_gate_flips_are_detected_after_a_large_step():
    """A step big enough to move the gates must show up as a nonzero flip rate."""
    torch.manual_seed(0)
    d = 8
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=8).to(torch.float64)
    net.initialize("looks_linear", 0)
    x = torch.randn(8, d, dtype=torch.float64)
    before = net.gates(x)
    with torch.no_grad():
        for W in net.weights:
            W.add_(torch.randn_like(W))
    out = linearization_validity(net, x, before)
    assert out["gate_flip_rate"] > 0.0
    assert out["frozen_gate_error"] > 0.0
