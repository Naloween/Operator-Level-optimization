"""
SOAP (Shampoo with Adam in the preconditioner's eigenbasis).

Adapted from Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam"
(ICLR 2025); implementation follows the structure of the authors' reference code:
https://github.com/nikhilvyas/SOAP — simplified here to rank-2 (matrix) parameters,
full eigendecomposition each step (cheap when fan dims are small).

Left/right Kronecker statistics use EMAs of G Gᵀ and Gᵀ G; eigenbases Q_L, Q_R
define Z = Q_Lᵀ G Q_R where Adam's moments run; updates map back as Q_L Z Q_Rᵀ.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class SOAP(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        shampoo_beta: float = -1.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        b2_for_precond = shampoo_beta if shampoo_beta >= 0.0 else betas[1]
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=b2_for_precond,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _eigenbases(sym: torch.Tensor, eps: float) -> torch.Tensor:
        """Orthonormal eigenvectors (columns), descending eigenvalues (flip like reference)."""
        w, q = torch.linalg.eigh(sym + eps * torch.eye(sym.shape[0], device=sym.device, dtype=sym.dtype))
        q = torch.flip(q, dims=[1])
        return q

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            sb = group["shampoo_beta"]
            eps = group["eps"]
            wd = group["weight_decay"]
            corr = group["correct_bias"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0

                if p.ndim != 2:
                    if "m" not in st:
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                    st["m"].mul_(beta1).add_(g, alpha=1 - beta1)
                    st["v"].mul_(beta2).add_(g.square(), alpha=1 - beta2)
                    st["step"] += 1
                    bias_c1 = 1.0 - beta1 ** st["step"]
                    bias_c2 = 1.0 - beta2 ** st["step"]
                    step_size = lr * (bias_c2 ** 0.5 / bias_c1) if corr else lr
                    denom = st["v"].sqrt().add_(eps)
                    p.add_(st["m"] / denom, alpha=-step_size)
                    continue

                d0, d1 = p.shape
                if "GG_L" not in st:
                    st["GG_L"] = torch.zeros(d0, d0, device=p.device, dtype=p.dtype)
                    st["GG_R"] = torch.zeros(d1, d1, device=p.device, dtype=p.dtype)
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)

                gg_l = g @ g.T
                gg_r = g.T @ g
                st["GG_L"].mul_(sb).add_(gg_l, alpha=1.0 - sb)
                st["GG_R"].mul_(sb).add_(gg_r, alpha=1.0 - sb)

                ql = self._eigenbases(st["GG_L"], eps)
                qr = self._eigenbases(st["GG_R"], eps)

                gt = ql.T @ g @ qr

                st["step"] += 1
                step_no = st["step"]

                exp_avg = st["exp_avg"]
                exp_avg_sq = st["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(gt, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).add_(gt.square(), alpha=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr
                if corr:
                    bias_c1 = 1.0 - beta1 ** step_no
                    bias_c2 = 1.0 - beta2 ** step_no
                    step_size = lr * (bias_c2 ** 0.5 / bias_c1)

                upd_t = exp_avg / denom
                upd = ql @ upd_t @ qr.T
                p.add_(upd, alpha=-step_size)

                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

        return loss
