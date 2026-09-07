"""Muon: orthogonalized gradient updates via Newton-Schulz.

Muon replaces the gradient by its polar factor, so the update has all singular values
equal to one. That is a *one-sided* reshaping -- it normalizes the update's own spectrum
but carries no information about the context Grams that decide how a layer update moves
the composed operator. The framework therefore groups it with the elementwise methods
rather than with K-FAC and Shampoo, and predicts it will not recover a collapsed operator
spectrum despite its updates being perfectly conditioned in parameter space.
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

from olo.optim.baselines.wrapper import ParamSpaceOptimizer


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """Approximate polar factor of a 2D matrix by the quintic Newton-Schulz iteration."""
    a, b, c = 1.875, -1.25, 0.375
    X = G / (G.norm() + eps)
    transposed = X.shape[0] > X.shape[1]        # the iteration is stable on wide matrices
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if transposed else X


class _Muon(TorchOptimizer):
    def __init__(self, params, lr, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(group["momentum"]).add_(g)
                d = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                if d.ndim != 2:                  # no polar factor for a vector
                    p.add_(d / (d.norm() + group["eps"]), alpha=-group["lr"])
                    continue
                u = newton_schulz(d, group["ns_steps"], group["eps"])
                # Keep the RMS of the update comparable across layer shapes.
                p.add_(u, alpha=-group["lr"] * max(1.0, d.shape[0] / d.shape[1]) ** 0.5)
        return loss


class Muon(ParamSpaceOptimizer):
    display_name = "muon"

    def make(self, params, lr, **kw):
        return _Muon(params, lr=lr, **kw)
