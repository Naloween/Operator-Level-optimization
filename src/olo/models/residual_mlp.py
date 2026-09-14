"""ReLU MLP with skip connections: the third architecture in the comparison.

A residual block is `h <- h + W relu(h)`, so the Jacobian factor is `I + W D(h)` with
`D = diag(1[h>0])` — an *affine* perturbation of the identity rather than a product factor.
That is exactly why this architecture belongs in the comparison: the other two build `J` as a
product of gated matrices, and this one builds it as a product of `I + (.)`, which is a
different way of keeping the operator away from zero.

It deliberately does **not** subclass `FactoredNet`. That protocol writes
`J = W_L D_{L-1} \cdots W_1`, and `I + WD` is not of that form; forcing it (by treating the
layer as `[I | W]` against a gate `[I; D]`) would make the identity branch a trainable
parameter and change the model. Since the studies here need only `J(x)` and the weight
gradients, `operator` is implemented directly and finite differences supply everything else.

Initialisations, matched to the other two architectures so the comparison is fair:

    xavier    standard, every weight
    haar      orthogonal `W`, orthogonal in/out maps
    identity  `W = 0` in every block, identity in/out maps, so `J = I` exactly at init
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from olo.init.schemes import haar


class ResidualReLUMLP(nn.Module):
    """`x -> W_in x -> (h + W_l relu(h)) x depth -> W_out h`."""

    input_independent = False

    def __init__(self, d_in: int, d_out: int, width: int, depth: int) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.d_in, self.d_out, self.width, self.depth = int(d_in), int(d_out), int(width), int(depth)
        self.W_in = nn.Parameter(torch.empty(width, d_in))
        self.blocks = nn.ParameterList(
            nn.Parameter(torch.empty(width, width)) for _ in range(depth)
        )
        self.W_out = nn.Parameter(torch.empty(d_out, width))

    # -- initialisation -----------------------------------------------------

    def initialize(self, scheme: str, seed: int, gain: float | str = "auto") -> None:
        gen = torch.Generator().manual_seed(int(seed))
        g = math.sqrt(2.0) if gain == "auto" else float(gain)
        with torch.no_grad():
            if scheme == "xavier":
                for W in (self.W_in, *self.blocks, self.W_out):
                    W.copy_(torch.randn(W.shape, generator=gen) * (g / math.sqrt(W.shape[1])))
            elif scheme in ("haar", "orthogonal"):
                self.W_in.copy_(haar(*self.W_in.shape, gen))
                self.W_out.copy_(haar(*self.W_out.shape, gen))
                for W in self.blocks:
                    W.copy_(haar(*W.shape, gen))
            elif scheme == "identity":
                # J = W_out (prod of I) W_in = W_out W_in = I when the maps are (partial)
                # identities and every block is zero.
                self.W_in.copy_(torch.eye(*self.W_in.shape))
                self.W_out.copy_(torch.eye(*self.W_out.shape))
                for W in self.blocks:
                    W.zero_()
            else:
                raise ValueError(
                    f"unknown init {scheme!r} for a residual MLP; "
                    "known: xavier, haar/orthogonal, identity"
                )

    # -- forward and operator ----------------------------------------------

    @property
    def weights(self) -> list[torch.Tensor]:
        return [self.W_in, *self.blocks, self.W_out]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W_in.T
        for W in self.blocks:
            h = h + torch.relu(h) @ W.T
        return h @ self.W_out.T

    def pre_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """The block inputs `h_l`, whose signs are the gates."""
        out, h = [], x @ self.W_in.T
        for W in self.blocks:
            out.append(h)
            h = h + torch.relu(h) @ W.T
        return out

    def operator(self, x: torch.Tensor) -> torch.Tensor:
        """`J(x) = W_out (I + W_L D_L) ... (I + W_1 D_1) W_in`, as (B, d_out, d_in)."""
        B = x.shape[0]
        J = self.W_in.unsqueeze(0).expand(B, *self.W_in.shape)
        h = x @ self.W_in.T
        eye = torch.eye(self.width, device=x.device, dtype=x.dtype)
        for W in self.blocks:
            D = (h > 0).to(x.dtype)                       # (B, width)
            J = (eye.unsqueeze(0) + W.unsqueeze(0) * D.unsqueeze(1)) @ J
            h = h + torch.relu(h) @ W.T
        return self.W_out.unsqueeze(0) @ J

    def describe(self) -> dict[str, object]:
        return {"type": "ResidualReLUMLP", "d_in": self.d_in, "d_out": self.d_out,
                "width": self.width, "depth": self.depth,
                "n_params": sum(p.numel() for p in self.parameters())}
