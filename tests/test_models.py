"""The factored-network protocol must agree with autograd, at every depth and init.

`operator(x)` is the object every method and diagnostic in this repo is defined against,
so if it silently disagreed with the true input-output Jacobian, every downstream number
would be wrong in a way no experiment would reveal. These tests pin it to autograd.
"""
from __future__ import annotations

import pytest
import torch

from olo.models import CReLUMLP, DeepLinear, FGLN, ReLUMLP

MODELS = {
    "deep_linear": lambda d, L: DeepLinear(d_in=d, d_out=d, width=d, depth=L),
    "fgln": lambda d, L: FGLN(d_in=d, d_out=d, width=d, depth=L, p_gate=0.8, gate_seed=1),
    "relu_mlp": lambda d, L: ReLUMLP(d_in=d, d_out=d, width=d, depth=L),
    "crelu_mlp": lambda d, L: CReLUMLP(d_in=d, d_out=d, width=d, depth=L),
}


def _net(kind, d, L, scheme="xavier", seed=0):
    net = MODELS[kind](d, L).to(torch.float64)
    net.initialize(scheme, seed)
    return net


@pytest.mark.parametrize("kind", list(MODELS))
@pytest.mark.parametrize("depth", [1, 2, 5])
def test_operator_matches_autograd_jacobian(kind, depth):
    torch.manual_seed(0)
    d = 6
    net = _net(kind, d, depth)
    x = torch.randn(4, d, dtype=torch.float64)

    J = net.operator(x)
    for b in range(x.shape[0]):
        want = torch.autograd.functional.jacobian(net.forward, x[b : b + 1]).reshape(d, d)
        got = J[b] if J.shape[0] > 1 else J[0]      # depth-1 rectifier nets are linear
        assert torch.allclose(got, want, atol=1e-9), f"{kind} L={depth} sample {b}"


@pytest.mark.parametrize("kind", list(MODELS))
@pytest.mark.parametrize("depth", [1, 2, 5])
def test_operator_reproduces_forward(kind, depth):
    """J(x) x must equal f(x): the network is exactly its own Jacobian applied to x."""
    torch.manual_seed(0)
    d = 6
    net = _net(kind, d, depth)
    x = torch.randn(4, d, dtype=torch.float64)
    J = net.operator(x)
    assert torch.allclose(torch.einsum("bij,bj->bi", J.expand(4, d, d), x), net(x), atol=1e-9)


@pytest.mark.parametrize("kind", list(MODELS))
@pytest.mark.parametrize("depth", [2, 5])
def test_contexts_factor_the_operator(kind, depth):
    """P = A_k W_k B_k must hold for every layer -- the identity ALS is built on."""
    torch.manual_seed(0)
    d = 5
    net = _net(kind, d, depth)
    x = torch.randn(3, d, dtype=torch.float64)
    P = net.operator(x)
    for k in range(depth):
        A, B = net.contexts(x, k)
        assert torch.allclose(A @ net.weights[k] @ B, P.expand_as(A @ net.weights[k] @ B),
                              atol=1e-9), f"{kind} layer {k}"


@pytest.mark.parametrize("kind", list(MODELS))
def test_right_contexts_match_contexts(kind):
    """The batched prefix scan must equal the per-layer construction."""
    torch.manual_seed(0)
    d, L = 5, 4
    net = _net(kind, d, L)
    x = torch.randn(3, d, dtype=torch.float64)
    stack = net.right_contexts(x)
    for k in range(L):
        _, B = net.contexts(x, k)
        assert torch.allclose(stack[k], B, atol=1e-12), f"{kind} layer {k}"


# -- the initialization claims the project rests on -------------------------


@pytest.mark.parametrize("depth", [2, 8, 64])
def test_crelu_looks_linear_is_exactly_orthogonal(depth):
    """The central claim: J(x) = O_L...O_1 for every x, at any depth. No collapse."""
    torch.manual_seed(0)
    d = 8
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=depth).to(torch.float64)
    net.initialize("looks_linear", seed=3)
    x = torch.randn(16, d, dtype=torch.float64) * 10.0        # far from the origin

    J = net.operator(x)
    assert J.shape[0] in (1, 16)
    J0 = J[0]
    for b in range(J.shape[0]):                                # same operator for every x
        assert torch.allclose(J[b], J0, atol=1e-10)
    sv = torch.linalg.svdvals(J0)
    assert torch.allclose(sv, torch.ones_like(sv), atol=1e-8), f"depth {depth}: {sv}"


@pytest.mark.parametrize("depth", [2, 8, 64])
def test_relu_looks_linear_is_exactly_orthogonal(depth):
    """The mirror construction gives the same guarantee -- at initialization only."""
    torch.manual_seed(0)
    d = 8
    net = ReLUMLP(d_in=d, d_out=d // 2, width=d, depth=depth).to(torch.float64)
    net.initialize("looks_linear", seed=3)
    x = torch.randn(16, d, dtype=torch.float64) * 10.0

    J = net.operator(x)
    sv = torch.linalg.svdvals(J[0])
    assert torch.allclose(sv, torch.ones_like(sv), atol=1e-8), f"depth {depth}: {sv}"
    for b in range(J.shape[0]):
        assert torch.allclose(J[b], J[0], atol=1e-10)


@pytest.mark.parametrize("depth", [2, 16])
def test_xavier_relu_collapses_with_depth(depth):
    """The control: the standard regime really does destroy the operator spectrum."""
    torch.manual_seed(0)
    d = 16
    net = ReLUMLP(d_in=d, d_out=d, width=d, depth=64).to(torch.float64)
    net.initialize("xavier", seed=0)
    x = torch.randn(8, d, dtype=torch.float64)
    sv = torch.linalg.svdvals(net.operator(x)[0])
    assert (sv[0] / sv[-1]) > 1e3, "expected an ill-conditioned Xavier operator at L=64"


@pytest.mark.parametrize("kind", ["deep_linear", "fgln"])
def test_identity_init_gives_identity_operator(kind):
    d, L = 6, 10
    net = MODELS[kind](d, L).to(torch.float64)
    net.initialize("identity", seed=0)
    x = torch.randn(4, d, dtype=torch.float64)
    P = net.operator(x)[0]
    if kind == "deep_linear":
        assert torch.allclose(P, torch.eye(d, dtype=torch.float64), atol=1e-12)
    else:                                       # gates zero some coordinates permanently
        assert torch.allclose(P, torch.diag(net.masks.prod(0).double()), atol=1e-12)


def test_looks_linear_rejected_for_linear_models():
    net = DeepLinear(d_in=4, d_out=4, width=4, depth=3)
    with pytest.raises(ValueError, match="looks_linear"):
        net.initialize("looks_linear", seed=0)


def test_crelu_gate_reproduces_concatenated_features():
    """D(z) z must literally be [relu(z); relu(-z)], including at z = 0."""
    torch.manual_seed(0)
    d = 5
    net = CReLUMLP(d_in=d, d_out=d, width=d, depth=3).to(torch.float64)
    net.initialize("xavier", seed=0)
    x = torch.randn(7, d, dtype=torch.float64)
    zs = net.pre_activations(x)
    Ds = net.gates(x)
    for z, D in zip(zs, Ds):
        want = torch.cat([torch.relu(z), torch.relu(-z)], dim=-1)
        assert torch.allclose(torch.einsum("bij,bj->bi", D, z), want, atol=1e-12)
