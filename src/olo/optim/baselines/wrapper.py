"""Adapter turning a `torch.optim.Optimizer` into the repo's `Optimizer` interface.

Every parameter-space comparator does the same three things -- forward, backward, step --
so they share one implementation and differ only in which torch optimizer they build. That
uniformity is deliberate: it means the runner cannot accidentally give one method more
gradient evaluations, a different batch, or an extra update than another.
"""
from __future__ import annotations

from typing import Any

import torch

from olo.models.base import FactoredNet
from olo.optim.base import Optimizer


class ParamSpaceOptimizer(Optimizer):
    """Base for the parameter-space comparators."""

    def __init__(self, net: FactoredNet, lr: float, **kwargs: Any) -> None:
        super().__init__(net, lr)
        self.kwargs = kwargs
        self.opt = self.make(list(net.parameters()), lr, **kwargs)

    def make(self, params, lr: float, **kwargs) -> torch.optim.Optimizer:
        raise NotImplementedError

    def step(self, x, y, task) -> dict[str, float]:
        loss = task.loss(task.forward(self.net, x), y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return {"loss": float(loss.detach())}

    def describe(self) -> dict[str, object]:
        return {"type": self.display_name, "lr": self.lr, **self.kwargs}

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state):
        self.opt.load_state_dict(state)
