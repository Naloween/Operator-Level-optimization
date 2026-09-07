"""The optimizer interface, shared by ALS and every parameter-space comparator.

One `step` performs one update and returns scalar metrics. The interface deliberately
takes the raw batch rather than a pre-computed loss, because the two families need
different things from it: parameter-space methods need `loss.backward()`, while ALS needs
the operator-space gradient dL/dP, which it assembles from the per-sample output gradient.
Giving both the same entry point is what makes the runner method-agnostic -- and therefore
what makes matched step budgets and matched learning-rate grids automatic rather than a
protocol the experimenter has to remember to follow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from olo.models.base import FactoredNet


class Optimizer(ABC):
    """One update per `step`. Subclasses own their own state."""

    #: reported in run metadata so figures can label methods without a lookup table
    display_name: str = "optimizer"

    def __init__(self, net: FactoredNet, lr: float) -> None:
        self.net = net
        self.lr = float(lr)

    @abstractmethod
    def step(self, x: torch.Tensor, y: torch.Tensor, task) -> dict[str, float]:
        """Take one step on batch (x, y); return metrics including 'loss' (pre-update)."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass
