"""Exactness properties of the ALS solver.

The method's whole claim is that it solves the operator-projection sub-problems *exactly*.
That is a falsifiable statement about the linear algebra, and these tests falsify it:
against closed forms where one exists, against the other solve path where both apply, and
against a brute-force least-squares solve where neither does.
"""
from __future__ import annotations

import pytest
import torch

from olo.models import CReLUMLP, DeepLinear, FGLN, ReLUMLP
from olo.optim.als import OperatorALS
from olo.optim.linalg import solve_kron_sum, solve_sylvester
from olo.tasks.teacher_student import TeacherStudent


def _setup(kind, d=6, L=3, n=8, scheme="haar", seed=0, **task_kw):
    torch.manual_seed(seed)
    cls = {"deep_linear": DeepLinear, "fgln": FGLN, "relu_mlp": ReLUMLP, "crelu_mlp": CReLUMLP}[kind]
    net = cls(d_in=d, d_out=d, width=d, depth=L).to(torch.float64)
    net.initialize(scheme, seed)
    task = TeacherStudent(d=d, n=n, seed=seed, **task_kw)
    task.to(torch.device("cpu"), torch.float64)
    return net, task


# -- the two solvers, against ground truth ----------------------------------


def test_sylvester_solves_its_equation():
    torch.manual_seed(0)
    n, m, lam = 5, 4, 1e-3
    A = torch.randn(7, n, dtype=torch.float64)
    B = torch.randn(m, 9, dtype=torch.float64)
    C = torch.randn(n, m, dtype=torch.float64)
    M, N = A.T @ A, B @ B.T

    X = solve_sylvester(M, N, C, lam)
    assert torch.allclose(M @ X @ N + lam * X, C, atol=1e-10)


def test_sylvester_matches_dense_least_squares():
    """The modewise filter must equal the brute-force normal-equation solve."""
    torch.manual_seed(0)
    n, m, lam = 4, 3, 1e-2
    A = torch.randn(6, n, dtype=torch.float64)
    B = torch.randn(m, 5, dtype=torch.float64)
    R = torch.randn(6, 5, dtype=torch.float64)

    got = solve_sylvester(A.T @ A, B @ B.T, A.T @ R @ B.T, lam)

    H = torch.kron(B @ B.T, A.T @ A) + lam * torch.eye(n * m, dtype=torch.float64)
    rhs = (A.T @ R @ B.T).T.reshape(-1)
    want = torch.linalg.solve(H, rhs).reshape(m, n).T
    assert torch.allclose(got, want, atol=1e-10)


def test_kron_sum_matches_sylvester_for_one_sample():
    """With a single sample the Kronecker sum collapses to one product: paths must agree."""
    torch.manual_seed(0)
    n, m, lam = 4, 3, 1e-3
    A = torch.randn(1, 6, n, dtype=torch.float64)
    B = torch.randn(1, m, 5, dtype=torch.float64)
    R = torch.randn(1, 6, 5, dtype=torch.float64)
    G = (A.transpose(1, 2) @ R @ B.transpose(1, 2)).mean(0)

    exact = solve_kron_sum(A, B, G, lam)
    mode = solve_sylvester(A[0].T @ A[0], B[0] @ B[0].T, G, lam)
    assert torch.allclose(exact, mode, atol=1e-9)


def test_kron_sum_minimizes_the_batched_objective():
    """Gradient of (1/S)sum_s ||A_s X B_s - R_s||^2 + lam||X||^2 must vanish at X."""
    torch.manual_seed(0)
    S, n, m, lam = 5, 3, 4, 1e-2
    A = torch.randn(S, 6, n, dtype=torch.float64)
    B = torch.randn(S, m, 5, dtype=torch.float64)
    R = torch.randn(S, 6, 5, dtype=torch.float64)
    G = (A.transpose(1, 2) @ R @ B.transpose(1, 2)).mean(0)

    X = solve_kron_sum(A, B, G, lam).requires_grad_(True)
    obj = ((A @ X @ B - R) ** 2).flatten(1).sum(1).mean() + lam * (X ** 2).sum()
    (grad,) = torch.autograd.grad(obj, X)
    assert grad.abs().max() < 1e-8, grad.abs().max()


# -- the solver, on real networks -------------------------------------------


def test_single_sweep_hits_the_target_exactly_when_unregularized():
    """lam=0, depth 2, well-conditioned contexts: one sweep must land on P_tgt.

    With no regularizer the last layer can absorb the entire displacement, so a solver
    that is genuinely exact leaves zero residual. Anything above roundoff means the layer
    solve is approximate.
    """
    net, task = _setup("deep_linear", d=5, L=2, scheme="haar")
    opt = OperatorALS(net, lr=1.0, lam=0.0, n_sweeps=1)
    out = opt.step(*task.train_batch(0), task)
    assert out["als_residual"] < 1e-12, out


@pytest.mark.parametrize("kind", ["deep_linear", "fgln", "relu_mlp", "crelu_mlp"])
def test_sweeps_reduce_the_projection_residual(kind):
    """More sweeps must not make the projection worse -- ALS is block-coordinate descent."""
    resid = []
    for n_sweeps in (1, 3, 8):
        net, task = _setup(kind, d=5, L=4, scheme="haar")
        opt = OperatorALS(net, lr=0.5, lam=1e-6, n_sweeps=n_sweeps, layers="all")
        resid.append(opt.step(*task.train_batch(0), task)["als_residual"])
    assert resid[1] <= resid[0] + 1e-12, resid
    assert resid[2] <= resid[1] + 1e-12, resid


def test_deep_linear_recovers_the_target_operator():
    """End to end: ALS must drive the relative operator error to near machine precision."""
    net, task = _setup("deep_linear", d=8, L=16, n=32, scheme="haar")
    opt = OperatorALS(net, lr=1.0, lam=1e-8, n_sweeps=2)
    for step in range(60):
        opt.step(*task.train_batch(step), task)
    assert task.evaluate(net)["rel_operator_error"] < 1e-6


def test_crelu_looks_linear_trains_at_large_depth():
    """The conditioning claim, end to end: no warm-start, depth 128, still trainable.

    Depth is supposed to be the hard case; here it is not even a handicap, because the
    looks-linear CReLU operator is orthogonal at initialization no matter how many layers
    are composed.
    """
    torch.manual_seed(0)
    d = 8
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=128).to(torch.float64)
    net.initialize("looks_linear", seed=0)
    task = TeacherStudent(d=d, n=16, seed=0).to(torch.device("cpu"), torch.float64)

    opt = OperatorALS(net, lr=0.1, lam=1e-4, n_sweeps=1, layers="all")
    # Training loss: the claim under test is that the optimization still works at depth,
    # which is separate from whether 16 samples generalize.
    before = task.evaluate(net)["train_loss"]
    for step in range(50):
        opt.step(*task.train_batch(step), task)
    after = task.evaluate(net)["train_loss"]
    assert after < before * 1e-3, (before, after)


def test_solving_the_projection_does_not_imply_reducing_the_loss():
    """The frozen-gate objective is a surrogate, and an aggressive step exposes the gap.

    At lr=0.5, lam=1e-6, depth 128, ALS drives its own projection residual to ~1e-8 while
    the network's loss diverges: the increments are far larger than the pre-activation
    margins the gates were frozen at, so the linearization the objective is written in no
    longer describes the network. This is a property of the surrogate, not a solver bug,
    and `olo.diagnostics.linearization` is what makes it visible during a run.
    """
    torch.manual_seed(0)
    d = 8
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=128).to(torch.float64)
    net.initialize("looks_linear", seed=0)
    task = TeacherStudent(d=d, n=16, seed=0).to(torch.device("cpu"), torch.float64)

    opt = OperatorALS(net, lr=0.5, lam=1e-6, n_sweeps=1, layers="all")
    before = task.evaluate(net)["train_loss"]
    residuals = [opt.step(*task.train_batch(s), task)["als_residual"] for s in range(10)]

    # The projection residual falls by four orders of magnitude...
    assert min(residuals) < 1e-3 * residuals[0], residuals
    # ...while the network it was meant to improve is destroyed.
    assert task.evaluate(net)["train_loss"] > 1e3 * before


# -- layer assignment ------------------------------------------------------


def test_oversized_layer_without_outer_optimizer_raises():
    """Refusing beats silently downgrading to an approximate solve."""
    torch.manual_seed(0)
    net = ReLUMLP(d_in=200, d_out=4, width=40, depth=3)
    with pytest.raises(ValueError, match="max_exact_dim"):
        OperatorALS(net, max_exact_dim=64)


def test_oversized_layer_is_delegated_to_the_outer_optimizer():
    torch.manual_seed(0)
    net = ReLUMLP(d_in=200, d_out=4, width=40, depth=3).to(torch.float64)
    net.initialize("xavier", 0)
    opt = OperatorALS(net, max_exact_dim=2048, outer={"type": "adam", "lr": 1e-3})
    assert opt.outer_layers == [0]                       # 40x200 = 8000 > 2048
    assert opt.als_layers == [1, 2]

    task = TeacherStudent(d=200, d_out=4, n=8, seed=0).to(torch.device("cpu"), torch.float64)
    before = net.weights[0].detach().clone()
    opt.step(*task.train_batch(0), task)
    assert not torch.allclose(before, net.weights[0]), "outer optimizer did not move layer 0"


def test_input_independent_models_never_need_an_outer_optimizer():
    """Their layer solve is O(d^3) regardless of width, so nothing is ever priced out."""
    net = DeepLinear(d_in=512, d_out=512, width=512, depth=3)
    opt = OperatorALS(net, max_exact_dim=16)
    assert opt.outer_layers == []


# -- what lambda actually penalizes ----------------------------------------


def test_lambda_anchors_at_the_step_start_not_at_each_sweep():
    """The regularizer penalizes ||W - W_start||, so more sweeps must not anneal it away.

    If each sweep were regularized against its own starting point, the total displacement
    could grow without bound across sweeps and `n_sweeps` would act as a hidden step-size
    knob: the same lam would produce a larger and larger move. Anchored correctly, the
    displacement converges as sweeps accumulate rather than growing linearly.
    """
    disp = {}
    for n_sweeps in (1, 4, 16):
        net, task = _setup("deep_linear", d=6, L=4, scheme="haar")
        W0 = [W.detach().clone() for W in net.weights]
        OperatorALS(net, lr=1.0, lam=1e-1, n_sweeps=n_sweeps).step(*task.train_batch(0), task)
        disp[n_sweeps] = sum(float((W.detach() - w0).norm() ** 2) for W, w0 in zip(net.weights, W0)) ** 0.5

    assert disp[16] < 1.5 * disp[4], disp        # converging, not growing with sweeps
    assert disp[4] < 3.0 * disp[1], disp


def test_larger_lambda_gives_a_smaller_displacement():
    """The basic contract of the regularizer, at fixed sweeps and target."""
    disp = {}
    for lam in (1e-6, 1e-2, 1.0):
        net, task = _setup("deep_linear", d=6, L=4, scheme="haar")
        W0 = [W.detach().clone() for W in net.weights]
        OperatorALS(net, lr=1.0, lam=lam, n_sweeps=3).step(*task.train_batch(0), task)
        disp[lam] = sum(float((W.detach() - w0).norm() ** 2) for W, w0 in zip(net.weights, W0)) ** 0.5

    assert disp[1e-6] > disp[1e-2] > disp[1.0], disp
