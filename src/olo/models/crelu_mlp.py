"""Bias-free CReLU MLP: h = W [relu(z); relu(-z)], W in R^{d x 2d}.

Writing the concatenated rectifier as a *rectangular gate* is what makes this model
tractable in the same framework as the others:

    c = [relu(z); relu(-z)] = D(z) z ,    D(z) = [ diag(1[z>0]) ; -diag(1[z<0]) ]  (2d x d)

so J(x) = W_L D_{L-1}(x) ... D_1(x) W_1 exactly as everywhere else, and the ALS solver
needs no special case.

Two properties matter for this project:

1. `||c|| = ||z||` -- CReLU is norm preserving, so no gain correction is needed and no
   signal is discarded at any depth.
2. With `looks_linear` (W = [O | -O], O orthogonal), W D(z) = O for *every* z:

       W D(z) = O diag(1[z>0]) + O diag(1[z<0]) = O

   so J(x) = O_L ... O_1 is exactly orthogonal, at every depth, for every input. All
   context singular values are 1 at initialization. This removes spectral collapse by
   construction rather than by the warm-start the earlier work needed, and is why this
   parameterization trains at depths (L ~ 1024) where the ReLU/Xavier network cannot.
"""
from __future__ import annotations

import torch

from olo.init.schemes import haar, make_weight
from olo.models.base import FactoredNet, chain_shapes


class CReLUMLP(FactoredNet):
    input_independent = False

    def __init__(self, d_in: int, d_out: int | None = None, width: int | None = None,
                 depth: int = 2) -> None:
        width = d_in if width is None else width
        d_out = d_in if d_out is None else d_out
        super().__init__(d_in=d_in, d_out=d_out, width=width, depth=depth)

    def layer_shapes(self) -> list[tuple[int, int]]:
        return chain_shapes(self.d_in, self.d_out, self.width, self.depth, expand=2)

    def activate(self, k: int, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.relu(z), torch.relu(-z)], dim=-1)

    def pre_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        zs = []
        h = x
        for k in range(self.depth - 1):
            z = h @ self.layers[k].T
            zs.append(z)
            h = self.activate(k, z)
        return zs

    def gates(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for z in self.pre_activations(x):
            pos = torch.diag_embed((z > 0).to(z.dtype))
            neg = torch.diag_embed((z < 0).to(z.dtype))
            out.append(torch.cat([pos, -neg], dim=-2))          # (B, 2d, d)
        return out

    # -- block-structured inits ---------------------------------------------

    def make_layer(self, idx, shape, scheme, gen, gain):
        """`identity` and `looks_linear` are the [O | -O] family; the rest are generic.

        Only the first layer (which has no preceding gate) is a plain matrix; every later
        layer must pair its two halves with opposite signs for the gate to cancel.
        """
        if scheme not in ("identity", "looks_linear"):
            return make_weight(shape, scheme, gen, gain)
        if self.depth == 1:
            return make_weight(shape, "identity" if scheme == "identity" else "haar", gen, 1.0)

        def block(n_out: int, n_in: int) -> torch.Tensor:
            O = torch.eye(n_out, n_in) if scheme == "identity" else haar(n_out, n_in, gen)
            return torch.cat([O, -O], dim=1)

        if idx == 0:                                            # no gate before layer 1
            return (torch.eye(self.width, self.d_in) if scheme == "identity"
                    else haar(self.width, self.d_in, gen))
        if idx == self.depth - 1:
            return block(self.d_out, self.width)
        return block(self.width, self.width)
