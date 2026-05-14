from __future__ import annotations

import torch.nn as nn


def make_tiny_mlp(d_in: int = 24, hidden: int = 8, n_cls: int = 5) -> nn.Module:
    """Bias-free ReLU MLP used across MLP variant experiments."""
    return nn.Sequential(
        nn.Linear(d_in, hidden, bias=False),
        nn.ReLU(inplace=False),
        nn.Linear(hidden, n_cls, bias=False),
    )


def make_mlp(
    depth: int,
    d_in: int,
    hidden: int,
    d_out: int,
    *,
    relu: bool = False,
) -> nn.Module:
    """Bias-free MLP of given depth.  Set ``relu=True`` for ReLU activations between layers."""
    layers: list[nn.Module] = []
    for l in range(depth):
        in_f  = d_in  if l == 0         else hidden
        out_f = d_out if l == depth - 1 else hidden
        layers.append(nn.Linear(in_f, out_f, bias=False))
        if relu and l < depth - 1:
            layers.append(nn.ReLU(inplace=False))
    return nn.Sequential(*layers)
