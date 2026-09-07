"""The two elementwise comparators, straight from torch.

Heavy Ball is plain gradient descent with momentum: the method whose operator motion
Proposition 3.1 describes exactly, and therefore the cleanest reference point for the
mismatch. Adam adds elementwise adaptivity, which ignores the left/right Kronecker
structure of the layer problem entirely -- the framework predicts it should not recover
from a collapsed operator spectrum, and that prediction is testable.
"""
from __future__ import annotations

import torch

from olo.optim.baselines.wrapper import ParamSpaceOptimizer


class HeavyBall(ParamSpaceOptimizer):
    display_name = "heavyball"

    def make(self, params, lr, momentum: float = 0.9, weight_decay: float = 0.0, **kw):
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kw)


class Adam(ParamSpaceOptimizer):
    display_name = "adam"

    def make(self, params, lr, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
             weight_decay: float = 0.0, **kw):
        return torch.optim.Adam(
            params, lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, **kw
        )
