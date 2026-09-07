"""Deep linear network: P = W_L ... W_1, no gates.

The reference setting. The composed operator is a single matrix independent of the input,
so operator error, context spectra, and update alignment are all directly measurable with
no linearization anywhere -- any mismatch observed here is the factorization's, not an
approximation's.
"""
from __future__ import annotations

import torch

from olo.models.base import FactoredNet, chain_shapes


class DeepLinear(FactoredNet):
    input_independent = True

    def __init__(self, d_in: int, d_out: int | None = None, width: int | None = None,
                 depth: int = 2) -> None:
        width = d_in if width is None else width
        d_out = d_in if d_out is None else d_out
        super().__init__(d_in=d_in, d_out=d_out, width=width, depth=depth)

    def layer_shapes(self) -> list[tuple[int, int]]:
        return chain_shapes(self.d_in, self.d_out, self.width, self.depth)

    def gates(self, x: torch.Tensor) -> list[torch.Tensor]:
        """All identity. The same tensor is returned L-1 times -- no copies."""
        if self.depth == 1:
            return []
        W = self.layers[0]
        eye = torch.eye(self.width, device=W.device, dtype=W.dtype).unsqueeze(0)
        return [eye] * (self.depth - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for W in self.layers:
            h = h @ W.T
        return h
