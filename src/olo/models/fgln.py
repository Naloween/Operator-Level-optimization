"""Fixed-gate linear network: P = W_L D_{L-1} ... D_1 W_1 with fixed Bernoulli diagonals.

The bridge between deep linear and a rectifier network: the operator is still a single
input-independent matrix (so everything stays exactly measurable), but the gates already
impose the rank and routing constraints that a real ReLU network has. Whatever breaks here
is caused by the gate geometry alone, with the data-dependence of real gates removed.
"""
from __future__ import annotations

import math

import torch

from olo.models.base import FactoredNet, chain_shapes


class FGLN(FactoredNet):
    input_independent = True

    def __init__(self, d_in: int, d_out: int | None = None, width: int | None = None,
                 depth: int = 2, p_gate: float = 0.9, gate_seed: int = 0) -> None:
        width = d_in if width is None else width
        d_out = d_in if d_out is None else d_out
        if not 0.0 < p_gate <= 1.0:
            raise ValueError(f"p_gate must be in (0, 1], got {p_gate}")
        self.p_gate = float(p_gate)
        super().__init__(d_in=d_in, d_out=d_out, width=width, depth=depth)

        g = torch.Generator().manual_seed(int(gate_seed))
        masks = torch.rand(max(depth - 1, 0), width, generator=g).lt(p_gate).float()
        self.register_buffer("masks", masks)

    @property
    def default_gain(self) -> float:
        """A mask keeping a fraction p of coordinates scales signal variance by p."""
        return 1.0 / math.sqrt(self.p_gate)

    def layer_shapes(self) -> list[tuple[int, int]]:
        return chain_shapes(self.d_in, self.d_out, self.width, self.depth)

    def gates(self, x: torch.Tensor) -> list[torch.Tensor]:
        W = self.layers[0]
        m = self.masks.to(device=W.device, dtype=W.dtype)
        return [torch.diag(m[k]).unsqueeze(0) for k in range(self.depth - 1)]

    def activate(self, k: int, z: torch.Tensor) -> torch.Tensor:
        return z * self.masks[k].to(device=z.device, dtype=z.dtype)
