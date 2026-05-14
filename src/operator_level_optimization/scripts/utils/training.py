from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


class EarlyStopping:
    """Patience-based early stopping on a monotone metric (lower is better)."""

    def __init__(
        self,
        patience: int | None,
        rel_tol: float = 1e-4,
        abs_tol: float = 1e-10,
        min_steps: int = 200,
    ) -> None:
        self.patience = patience
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.min_steps = min_steps
        self._best = float("inf")
        self._stall = 0
        self.triggered_at: int | None = None

    def check(self, val: float, step: int) -> bool:
        """Return True if training should stop (patience exceeded)."""
        if self.patience is None:
            return False
        if self._best == float("inf"):
            self._best = val
            return False
        thr = max(self.rel_tol * self._best, self.abs_tol)
        if val < self._best - thr:
            self._best = val
            self._stall = 0
        elif step >= self.min_steps:
            self._stall += 1
        if step >= self.min_steps and self._stall >= self.patience:
            self.triggered_at = step
            return True
        return False


def run_mlp_loop(
    model: nn.Module,
    opt: Any,
    loss_fn: Callable[[], torch.Tensor],
    steps: int,
    *,
    catch: tuple[type[Exception], ...] = (RuntimeError, FloatingPointError),
) -> list[tuple[int, float]]:
    """Standard MLP training loop with eval-mode checkpointing.

    Evaluates ``loss_fn()`` under ``model.eval()`` at steps 0, 1, …, ``steps``,
    and runs optimizer steps under ``model.train()``.  Using eval mode for the
    evaluation passes prevents forward hooks from accumulating activations twice
    (the hook-accumulation bug that affects ALS variants).

    ``loss_fn`` must be a zero-argument callable — bind ``model``, ``x``, ``y``
    externally via a lambda or closure.

    Exceptions in ``catch`` are silently caught; the returned list ends at the
    last successful step.  Pass ``catch=()`` to let all exceptions propagate.

    Returns a list of ``(step, eval_loss)`` tuples.
    """
    hist: list[tuple[int, float]] = []
    for t in range(steps + 1):
        model.eval()
        with torch.no_grad():
            val = float(loss_fn().detach().cpu().item())
        model.train()
        hist.append((t, val))
        if t == steps:
            break
        try:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            opt.step()
        except catch:
            break
    return hist
