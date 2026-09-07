"""K-FAC, with its Kronecker factors read directly off the chain.

K-FAC preconditions layer k's gradient by the inverses of two covariances,

    A_k = E[h_{k-1} h_{k-1}^T]   (inputs),      G_k = E[d_k d_k^T]   (pre-activation grads)

    dW_k = -eta (G_k + lam I)^{-1} G_{W_k} (A_k + lam I)^{-1}

This is the comparator the framework makes a *falsifiable* statement about. In the
isotropic deep-linear setting A_k = B_k B_k^T and G_k = A_k^T A_k, so K-FAC diagonalizes
in exactly the ALS bases (V_A, U_B) and differs only in its denominator:

    K-FAC   (sigma_A^2 + lam)(sigma_B^2 + lam)      vs      ALS   sigma_A^2 sigma_B^2 + lam

a separable approximation of the exact modewise filter, with gap
lam(sigma_A^2 + sigma_B^2 + lam - 1) that grows with the damping. So K-FAC is predicted to
be the closest baseline, and to depart from the projection precisely in the high-damping
regime usually adopted for stability at depth. `olo.diagnostics.mismatch` is what turns
that prediction into a measurement.
"""
from __future__ import annotations

import torch

from olo.models.base import FactoredNet
from olo.optim.base import Optimizer
from olo.tasks.base import Task


class KFAC(Optimizer):
    display_name = "kfac"

    def __init__(
        self,
        net: FactoredNet,
        lr: float,
        damping: float = 1e-2,
        factor_decay: float = 0.95,
        momentum: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(net, lr)
        self.damping = float(damping)
        self.factor_decay = float(factor_decay)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self._A: list[torch.Tensor | None] = [None] * net.depth
        self._G: list[torch.Tensor | None] = [None] * net.depth
        self._buf: list[torch.Tensor | None] = [None] * net.depth

    def step(self, x, y, task) -> dict[str, float]:
        net = self.net
        if type(task).forward is not Task.forward:
            raise NotImplementedError(
                f"K-FAC reads its Kronecker factors off the layer chain, so it needs the "
                f"batch to be network inputs. Task {type(task).__name__} defines its own "
                "forward (its inputs are not feature vectors), and K-FAC's activation "
                "covariance is undefined there."
            )
        yhat, hs, zs = net.forward_collect(x)
        loss = task.loss(yhat, y)

        # One backward pass yields every pre-activation gradient and every weight
        # gradient, so the factors and the gradient are all evaluated at the same point.
        grads = torch.autograd.grad(loss, list(net.weights) + zs)
        w_grads, z_grads = grads[: net.depth], grads[net.depth :]

        with torch.no_grad():
            B = x.shape[0]
            for k in range(net.depth):
                h, d = hs[k].detach(), z_grads[k].detach() * B      # undo the batch mean
                self._A[k] = _ema(self._A[k], (h.T @ h) / B, self.factor_decay)
                self._G[k] = _ema(self._G[k], (d.T @ d) / B, self.factor_decay)

                nat = (
                    _inv_damped(self._G[k], self.damping, self.eps)
                    @ w_grads[k]
                    @ _inv_damped(self._A[k], self.damping, self.eps)
                )
                if self.momentum:
                    if self._buf[k] is None:
                        self._buf[k] = torch.zeros_like(nat)
                    self._buf[k].mul_(self.momentum).add_(nat)
                    nat = self._buf[k]
                net.weights[k].add_(nat, alpha=-self.lr)

        return {"loss": float(loss.detach())}

    def describe(self) -> dict[str, object]:
        return {
            "type": "kfac",
            "lr": self.lr,
            "damping": self.damping,
            "factor_decay": self.factor_decay,
            "momentum": self.momentum,
        }


def _ema(prev: torch.Tensor | None, new: torch.Tensor, decay: float) -> torch.Tensor:
    return new.clone() if prev is None else prev.mul_(decay).add_(new, alpha=1.0 - decay)


def _inv_damped(m: torch.Tensor, damping: float, eps: float) -> torch.Tensor:
    eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
    return torch.linalg.solve(m + (damping + eps) * eye, eye)
