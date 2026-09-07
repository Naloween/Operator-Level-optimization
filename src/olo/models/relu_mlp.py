"""Bias-free ReLU MLP. Gates are data-dependent square 0/1 diagonals.

The first model where the operator becomes sample-specific: J(x) is the frozen-gate
Jacobian along x's activation pattern, so the batch no longer shares one operator and the
ALS sub-problem becomes a sum of Kronecker products (`olo.optim.als`, exact path B).

`looks_linear` here is the mirror construction of Balduzzi et al.: hidden activations are
kept in mirrored form [u; -u], on which ReLU acts as CReLU, so the network is exactly
linear at initialization. Unlike `crelu_mlp`, nothing preserves that structure once
training starts -- which is precisely the comparison worth running.
"""
from __future__ import annotations

import math

import torch

from olo.init.schemes import haar, make_weight
from olo.models.base import FactoredNet, chain_shapes


class ReLUMLP(FactoredNet):
    input_independent = False

    def __init__(self, d_in: int, d_out: int | None = None, width: int | None = None,
                 depth: int = 2) -> None:
        width = d_in if width is None else width
        d_out = d_in if d_out is None else d_out
        super().__init__(d_in=d_in, d_out=d_out, width=width, depth=depth)

    @property
    def default_gain(self) -> float:
        """ReLU zeroes half the coordinates in expectation: He scaling."""
        return math.sqrt(2.0)

    def layer_shapes(self) -> list[tuple[int, int]]:
        return chain_shapes(self.d_in, self.d_out, self.width, self.depth)

    def activate(self, k: int, z: torch.Tensor) -> torch.Tensor:
        return torch.relu(z)

    def pre_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        """z_1..z_{L-1}, the pre-activations the gates are read off."""
        zs = []
        h = x
        for k in range(self.depth - 1):
            z = h @ self.layers[k].T
            zs.append(z)
            h = torch.relu(z)
        return zs

    def gates(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [torch.diag_embed((z > 0).to(z.dtype)) for z in self.pre_activations(x)]

    # -- looks-linear (mirror) init -----------------------------------------

    def make_layer(self, idx, shape, scheme, gen, gain):
        if scheme != "looks_linear":
            return make_weight(shape, scheme, gen, gain)
        if self.depth == 1:
            return haar(*shape, gen)                      # no gates: already linear
        if self.width % 2 != 0:
            raise ValueError(
                f"looks_linear on a ReLU MLP needs an even width (mirrored halves); "
                f"got width={self.width}"
            )
        h = self.width // 2
        if idx == 0:                                       # [V; -V]: x -> [u; -u]
            V = haar(h, self.d_in, gen)
            return torch.cat([V, -V], dim=0)
        if idx == self.depth - 1:                          # [U, -U]: [relu(u); relu(-u)] -> U u
            U = haar(self.d_out, h, gen)
            return torch.cat([U, -U], dim=1)
        V = haar(h, h, gen)                                # [[V, -V], [-V, V]]
        return torch.cat(
            [torch.cat([V, -V], dim=1), torch.cat([-V, V], dim=1)], dim=0
        )
