"""
Shampoo: Kronecker-factored preconditioning (Anil et al., 2020).

EMA factors L ≈ 𝔼[G^T G] (d_in × d_in) and R ≈ 𝔼[G G^T] (d_out × d_out);
update uses (R + εI)^{-1/4} G (L + εI)^{-1/4}. Optional SGD-style momentum
is applied to G before the factors are accumulated and the step is taken.
1D parameters use the same momentum rule without preconditioning.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def _inv_pow_spd(
    m: torch.Tensor, power: float, eps: float
) -> torch.Tensor:
    """Return M^power for symmetric PD M, via eigh, with floor on eigenvalues."""
    eye = torch.eye(m.shape[0], device=m.device, dtype=m.dtype)
    evals, Q = torch.linalg.eigh(m + eps * eye)
    evals = evals.clamp(min=eps)
    scaled = Q * evals.pow(power).unsqueeze(0)
    return scaled @ Q.T


class Shampoo(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        beta: float = 0.9,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-4,
    ):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0,1), got {beta}")
        defaults = dict(
            lr=lr,
            beta=beta,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b = group["beta"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g + wd * p

                st = self.state[p]
                if "m_buf" not in st:
                    st["m_buf"] = torch.zeros_like(p)
                buf = st["m_buf"]
                buf.mul_(mom).add_(g)
                g_step = buf

                if p.ndim != 2:
                    p.add_(g_step, alpha=-lr)
                    continue

                d_out, d_in = p.shape
                if "L" not in st:
                    st["L"] = torch.zeros(
                        d_in, d_in, device=p.device, dtype=p.dtype
                    )
                    st["R"] = torch.zeros(
                        d_out, d_out, device=p.device, dtype=p.dtype
                    )

                gtg = g_step.T @ g_step
                ggt = g_step @ g_step.T
                st["L"].mul_(b).add_(gtg, alpha=(1.0 - b))
                st["R"].mul_(b).add_(ggt, alpha=(1.0 - b))

                p_r = _inv_pow_spd(st["R"], -0.25, eps)
                p_l = _inv_pow_spd(st["L"], -0.25, eps)
                upd = p_r @ g_step @ p_l
                p.add_(upd, alpha=-lr)

        return loss
