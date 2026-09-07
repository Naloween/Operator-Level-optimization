"""Shampoo: Kronecker-factored preconditioning from accumulated gradient statistics.

Shampoo is two-sided, which by the framework's account is the property that matters: its
update carries left and right factors and so can act on the operator's row and column
geometry. But those factors come from an EMA of past gradient outer products, not from the
current contexts, so its diagonalizing bases coincide with the ALS bases (V_A, U_B) only
when both contexts are near isometries. It is included as an empirical comparator, not as
an approximation of the projection objective.
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

from olo.optim.baselines.wrapper import ParamSpaceOptimizer
from olo.optim.linalg import svd_psd


def _inv_pow_spd(m: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    """M^power for symmetric PSD M, with an eigenvalue floor at eps.

    Goes through `svd_psd` rather than `eigh`: at an identity or orthogonal
    initialization the accumulated factors have many repeated eigenvalues, and LAPACK's
    symmetric eigensolver fails outright on them. A baseline that crashed in exactly the
    well-conditioned regime this project uses as its control would be no baseline at all.
    """
    eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
    Q, evals = svd_psd(m + eps * eye)
    return (Q * evals.clamp(min=eps).pow(power).unsqueeze(0)) @ Q.T


class _Shampoo(TorchOptimizer):
    def __init__(self, params, lr, beta: float = 0.9, momentum: float = 0.0,
                 weight_decay: float = 0.0, eps: float = 1e-4):
        super().__init__(params, dict(lr=lr, beta=beta, momentum=momentum,
                                      weight_decay=weight_decay, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, b, eps = group["lr"], group["beta"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if group["weight_decay"]:
                    g = g + group["weight_decay"] * p
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                st["buf"].mul_(group["momentum"]).add_(g)
                g = st["buf"]

                if p.ndim != 2:
                    p.add_(g, alpha=-lr)
                    continue
                if "L" not in st:
                    st["L"] = torch.zeros(p.shape[1], p.shape[1], device=p.device, dtype=p.dtype)
                    st["R"] = torch.zeros(p.shape[0], p.shape[0], device=p.device, dtype=p.dtype)
                st["L"].mul_(b).add_(g.T @ g, alpha=1.0 - b)
                st["R"].mul_(b).add_(g @ g.T, alpha=1.0 - b)
                upd = _inv_pow_spd(st["R"], -0.25, eps) @ g @ _inv_pow_spd(st["L"], -0.25, eps)
                p.add_(upd, alpha=-lr)
        return loss


class Shampoo(ParamSpaceOptimizer):
    display_name = "shampoo"

    def make(self, params, lr, **kw):
        return _Shampoo(params, lr=lr, **kw)
