"""Operator-space targets: what displacement of P the solver is asked to realize.

Separating the target from the solver is the point of the framework. The projection step
answers "which factor updates realize this operator motion?", and the target answers
"which operator motion do we want?". Keeping them apart makes the second question an
experimental variable rather than a hard-coded choice:

    gradient   P - eta*G      the steepest-descent step of the paper.
    shadow     any torch optimizer applied directly to P, with persistent state. This is
               the counterfactual control: it runs a method as if the network were
               unfactorized, so the difference between it and the same method applied to
               the factors isolates the bias that factorization itself introduces.
    fixed      a prescribed P*, for operator-recovery experiments where the endpoint,
               not the trajectory, is under test.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class OperatorTarget(ABC):
    """Maps the current operator and its gradient to a desired next operator."""

    @abstractmethod
    def propose(self, P: torch.Tensor, G: torch.Tensor, lr: float) -> torch.Tensor:
        """Target operator(s), same shape as P."""

    def describe(self) -> dict[str, object]:
        return {"type": type(self).__name__}


class GradientTarget(OperatorTarget):
    """P_tgt = P - eta*G: the operator-space steepest-descent step."""

    def propose(self, P, G, lr):
        return P - lr * G


class FixedTarget(OperatorTarget):
    """P_tgt = P + eta*(P* - P): move a fraction of the way to a prescribed operator."""

    def __init__(self, P_star: torch.Tensor | None = None) -> None:
        self.P_star = P_star

    def set_target(self, P_star: torch.Tensor) -> None:
        self.P_star = P_star

    def propose(self, P, G, lr):
        if self.P_star is None:
            raise RuntimeError("FixedTarget needs P_star; set it via set_target() or the task")
        star = self.P_star.to(device=P.device, dtype=P.dtype)
        while star.dim() < P.dim():
            star = star.unsqueeze(0)
        return P + lr * (star - P)


class ShadowTarget(OperatorTarget):
    """P_tgt = one step of `inner` applied to P as if it were a free parameter.

    The shadow tensor is re-seated on the true operator every step while the optimizer's
    moment estimates persist, so the target is "what this method would do from here",
    accumulated statistics included -- not the trajectory of a separate run that has
    already drifted away.
    """

    def __init__(self, inner: str = "adam", **inner_kwargs) -> None:
        self.inner_name = inner
        self.inner_kwargs = inner_kwargs
        self._param: torch.Tensor | None = None
        self._opt: torch.optim.Optimizer | None = None

    def _ensure(self, P: torch.Tensor, lr: float) -> None:
        if self._param is not None and self._param.shape == P.shape:
            return
        if self._param is not None:
            raise RuntimeError(
                "ShadowTarget received an operator whose shape changed between steps "
                f"({tuple(self._param.shape)} -> {tuple(P.shape)}); the shadow optimizer's "
                "state would be meaningless. Per-sample operators need a fixed batch size."
            )
        self._param = P.detach().clone().requires_grad_(True)
        ctor = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "heavyball": lambda p, **kw: torch.optim.SGD(p, momentum=0.9, **kw),
        }.get(self.inner_name)
        if ctor is None:
            raise ValueError(f"unknown shadow optimizer {self.inner_name!r}")
        self._opt = ctor([self._param], lr=lr, **self.inner_kwargs)

    @torch.no_grad()
    def propose(self, P, G, lr):
        self._ensure(P, lr)
        self._param.data.copy_(P)
        self._param.grad = G.detach().clone()
        self._opt.step()
        return self._param.detach().clone()

    def describe(self):
        return {"type": "ShadowTarget", "inner": self.inner_name, **self.inner_kwargs}
