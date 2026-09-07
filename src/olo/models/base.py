"""The factored-network protocol every model in this repo satisfies.

A factored network is a chain of weight matrices interleaved with gates:

    J(x) = W_L D_{L-1}(x) W_{L-1} ... D_1(x) W_1 ,     W_k in R^{n_k x m_{k-1}}

where the gate D_l in R^{m_l x n_l} is the identity (deep linear), a fixed diagonal
(FGLN), a data-dependent 0/1 diagonal (ReLU), or a data-dependent *rectangular* selector
(CReLU, where m_l = 2 n_l). Writing all four this way is what lets a single exact ALS
implementation cover every model: the gate is absorbed into the contexts and never
appears in the solver.

The two objects the rest of the codebase asks for:

    P = A_k W_k B_k,   A_k = W_L D_{L-1} ... W_{k+1} D_k,   B_k = D_{k-1} W_{k-1} ... W_1

with A_L = I and B_1 = I. `right_contexts` returns every B_k in one pass; A_k is built
incrementally by the solver, which sweeps downward (see `olo.optim.als`).

Sample axis convention: every operator/context tensor carries a leading batch dimension.
Models whose gates do not depend on the input report `input_independent = True` and use a
batch dimension of 1, which broadcasts against per-sample quantities for free and lets the
solver take a cheaper exact path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from olo.init.schemes import make_weight


def chain_shapes(
    d_in: int, d_out: int, width: int, depth: int, expand: int = 1
) -> list[tuple[int, int]]:
    """Layer shapes for a uniform-width chain.

    `expand` is the gate's output/input ratio: 1 when the gate is square (linear, fixed
    gate, ReLU), 2 for CReLU, whose gate doubles the feature dimension.
    """
    if depth == 1:
        return [(d_out, d_in)]
    return (
        [(width, d_in)]
        + [(width, expand * width) for _ in range(depth - 2)]
        + [(d_out, expand * width)]
    )


class FactoredNet(nn.Module, ABC):
    """Base class: subclasses define the gates, everything else is derived."""

    #: gates do not depend on the input (deep linear, FGLN) -> contexts are shared
    input_independent: bool = False

    def __init__(self, d_in: int, d_out: int, width: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.width = int(width)
        self.depth = int(depth)
        self.layers = nn.ParameterList(
            nn.Parameter(torch.empty(n_out, n_in)) for n_out, n_in in self.layer_shapes()
        )

    # -- structure ----------------------------------------------------------

    @abstractmethod
    def layer_shapes(self) -> list[tuple[int, int]]:
        """(n_k, m_{k-1}) for k = 1..L, in forward order."""

    @abstractmethod
    def gates(self, x: torch.Tensor) -> list[torch.Tensor]:
        """D_1..D_{L-1} as dense (B, m_l, n_l) tensors; B == 1 when input-independent.

        Dense gates are affordable because exact ALS caps the widths anyway, and they keep
        the contexts a single expression rather than four special cases.
        """

    @property
    def weights(self) -> list[torch.Tensor]:
        """W_1..W_L in forward order."""
        return list(self.layers)

    @property
    def default_gain(self) -> float:
        """Std multiplier that keeps activation norm stable through this activation."""
        return 1.0

    # -- initialization -----------------------------------------------------

    def initialize(self, scheme: str, seed: int, gain: float | str = "auto") -> None:
        g = float(self.default_gain) if gain == "auto" else float(gain)
        gen = torch.Generator().manual_seed(int(seed))
        with torch.no_grad():
            for idx, W in enumerate(self.layers):
                W.copy_(self.make_layer(idx, tuple(W.shape), scheme, gen, g))

    def make_layer(
        self,
        idx: int,
        shape: tuple[int, int],
        scheme: str,
        gen: torch.Generator,
        gain: float,
    ) -> torch.Tensor:
        """One layer's initial weight. Overridden where a scheme needs block structure."""
        return make_weight(shape, scheme, gen, gain)

    # -- operator -----------------------------------------------------------

    def operator(
        self, x: torch.Tensor, gates: list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        """J(x) as (B, d_out, d_in); B == 1 when the gates are input-independent.

        Passing `gates` reuses a previously captured gate set instead of reading it off
        the current weights. Callers that hold the gates fixed while the weights move --
        the frozen-gate linearization every solver here uses -- must pass them, or the
        operator they compute will not be the one their contexts factor.
        """
        Ws = self.weights
        Ds = self.gates(x) if gates is None else gates
        P = Ws[0].unsqueeze(0)
        for k in range(1, self.depth):
            P = Ws[k] @ (Ds[k - 1] @ P)
        return P

    def right_contexts(
        self, x: torch.Tensor, gates: list[torch.Tensor] | None = None
    ) -> list[torch.Tensor]:
        """[B_1, ..., B_L]; B_k = D_{k-1} W_{k-1} ... W_1, with B_1 = I.

        Computed as a forward prefix scan, so the whole stack costs L matmuls. A reverse
        ALS sweep consumes it in order without invalidating it: solving layer k only
        touches W_k and above, while B_k depends on W_1..W_{k-1}.
        """
        Ws = self.weights
        Ds = self.gates(x) if gates is None else gates
        eye = torch.eye(self.d_in, device=Ws[0].device, dtype=Ws[0].dtype)
        out = [eye.unsqueeze(0)]
        for k in range(1, self.depth):
            out.append(Ds[k - 1] @ (Ws[k - 1] @ out[-1]))
        return out

    def contexts(
        self, x: torch.Tensor, k: int, gates: list[torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(A_k, B_k) for layer index k (0-based), each with a leading batch dim."""
        if not 0 <= k < self.depth:
            raise IndexError(f"layer index {k} out of range for depth {self.depth}")
        Ws = self.weights
        Ds = self.gates(x) if gates is None else gates
        B_k = self.right_contexts(x, Ds)[k]

        n_out = Ws[-1].shape[0]
        A = torch.eye(n_out, device=Ws[0].device, dtype=Ws[0].dtype).unsqueeze(0)
        for j in range(self.depth - 1, k, -1):
            A = A @ (Ws[j] @ Ds[j - 1])
        return A, B_k

    def all_contexts(
        self, x: torch.Tensor, gates: list[torch.Tensor] | None = None
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """[(A_k, B_k)] for every layer, in O(L) matmuls rather than O(L^2).

        The diagnostics need every layer's context pair at once. Building them one at a
        time via `contexts` would rebuild the same prefixes L times over, which at depth
        1024 is the difference between a diagnostic and a bottleneck.
        """
        Ds = self.gates(x) if gates is None else gates
        Bs = self.right_contexts(x, Ds)
        Ws = self.weights

        A = torch.eye(Ws[-1].shape[0], device=Ws[0].device, dtype=Ws[0].dtype).unsqueeze(0)
        As: list[torch.Tensor] = [A] * self.depth
        for k in range(self.depth - 1, -1, -1):
            As[k] = A
            if k > 0:
                A = A @ (Ws[k] @ Ds[k - 1])
        return list(zip(As, Bs))

    # -- convenience --------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: apply the chain. Subclasses override when it is cheaper."""
        h = x
        Ws = self.weights
        for k in range(self.depth):
            h = h @ Ws[k].T
            if k < self.depth - 1:
                h = self.activate(k, h)
        return h

    def activate(self, k: int, z: torch.Tensor) -> torch.Tensor:
        """h = D_{k+1}(z) z for the gate following layer index `k`, on the batch."""
        return z

    def forward_collect(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """(yhat, inputs, pre_activations) with both lists indexed by layer.

        `inputs[k]` is what layer k consumes and `pre_activations[k]` what it produces
        before its gate -- exactly the two quantities K-FAC's Kronecker factors are
        expectations over. Reading them off the chain directly gives the factors without
        module hooks, which this repo's `nn.ParameterList` models could not support anyway.
        """
        hs, zs = [], []
        h = x
        for k in range(self.depth):
            hs.append(h)
            z = h @ self.weights[k].T
            zs.append(z)
            h = self.activate(k, z) if k < self.depth - 1 else z
        return h, hs, zs

    def describe(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "d_in": self.d_in,
            "d_out": self.d_out,
            "width": self.width,
            "depth": self.depth,
            "n_params": sum(p.numel() for p in self.parameters()),
            "layer_shapes": [tuple(W.shape) for W in self.weights],
            "input_independent": self.input_independent,
        }
