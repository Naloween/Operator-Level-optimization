"""SOAP: Adam run inside Shampoo's eigenbasis.

Like Shampoo, its geometry comes from accumulated gradient statistics rather than the
current contexts; unlike Shampoo, the preconditioner is used only to choose a basis, with
elementwise Adam doing the scaling inside it. Included as an empirical comparator.
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

from olo.optim.baselines.wrapper import ParamSpaceOptimizer
from olo.optim.linalg import svd_psd


class _SOAP(TorchOptimizer):
    def __init__(self, params, lr, betas: tuple = (0.9, 0.999), shampoo_beta: float = 0.95,
                 eps: float = 1e-8, weight_decay: float = 0.0, correct_bias: bool = True):
        super().__init__(params, dict(lr=lr, betas=tuple(betas), shampoo_beta=shampoo_beta,
                                      eps=eps, weight_decay=weight_decay,
                                      correct_bias=correct_bias))

    @staticmethod
    def _eigenbasis(sym: torch.Tensor, eps: float) -> torch.Tensor:
        """Eigenvectors, descending. Via `svd_psd`: `eigh` fails on repeated eigenvalues,
        which is the norm at an identity or orthogonal initialization."""
        eye = torch.eye(sym.shape[0], device=sym.device, dtype=sym.dtype)
        q, _ = svd_psd(sym + eps * eye)
        return q

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2) = group["lr"], group["betas"]
            sb, eps, corr = group["shampoo_beta"], group["eps"], group["correct_bias"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                st["step"] = st.get("step", 0) + 1

                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                    if p.ndim == 2:
                        st["GG_L"] = torch.zeros(p.shape[0], p.shape[0], device=p.device, dtype=p.dtype)
                        st["GG_R"] = torch.zeros(p.shape[1], p.shape[1], device=p.device, dtype=p.dtype)

                if p.ndim == 2:
                    st["GG_L"].mul_(sb).add_(g @ g.T, alpha=1.0 - sb)
                    st["GG_R"].mul_(sb).add_(g.T @ g, alpha=1.0 - sb)
                    ql = self._eigenbasis(st["GG_L"], eps)
                    qr = self._eigenbasis(st["GG_R"], eps)
                    gt = ql.T @ g @ qr
                else:
                    ql = qr = None
                    gt = g

                st["exp_avg"].mul_(beta1).add_(gt, alpha=1.0 - beta1)
                st["exp_avg_sq"].mul_(beta2).add_(gt.square(), alpha=1.0 - beta2)

                step_size = lr
                if corr:
                    step_size = lr * (1.0 - beta2 ** st["step"]) ** 0.5 / (1.0 - beta1 ** st["step"])
                upd = st["exp_avg"] / st["exp_avg_sq"].sqrt().add(eps)
                if ql is not None:
                    upd = ql @ upd @ qr.T
                p.add_(upd, alpha=-step_size)
                if group["weight_decay"]:
                    p.add_(p, alpha=-lr * group["weight_decay"])
        return loss


class SOAP(ParamSpaceOptimizer):
    display_name = "soap"

    def make(self, params, lr, **kw):
        return _SOAP(params, lr=lr, **kw)
