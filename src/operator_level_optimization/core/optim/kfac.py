"""
Simplified K-FAC for unbatched nn.Linear (no bias required).

For each 2D weight, maintains A ≈ 𝔼[aaᵀ] (input), B ≈ 𝔼[ggᵀ] (output-side grad) and
updates with (B + πI)^{-1} G (A + πI)^{-1}.

Requires :meth:`attach_hooks` before training.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

warnings.filterwarnings(
    "ignore",
    message="Full backward hook is firing",
    category=UserWarning,
)


def _inv_damped(
    m: torch.Tensor, pi: float, eps: float, floor: float
) -> torch.Tensor:
    """(m + (pi+floor)*I)^{-1} with an absolute floor so near-singular factors (typical
    in deep nets when the grad-output statistic is tiny) do not blow up the inverse.
    """
    d = m.shape[0]
    eye = torch.eye(d, device=m.device, dtype=m.dtype)
    damp = float(pi) + max(float(eps), float(floor))
    mat = m + damp * eye
    return torch.linalg.solve(mat, eye)


class KFAC(Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        factor_decay: float = 0.95,
        damping: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        inv_floor: float = 1e-2,
    ):
        if not 0.0 <= factor_decay < 1.0:
            raise ValueError("factor_decay must be in [0,1).")
        defaults = dict(
            lr=lr,
            factor_decay=factor_decay,
            damping=damping,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            inv_floor=inv_floor,
        )
        super().__init__(params, defaults)
        self.hooks: list = []
        self._pre_list: Dict[int, List[torch.Tensor]] = {}
        self._go_list: Dict[int, List[torch.Tensor]] = {}
        self.pre_acts: Dict[int, torch.Tensor] = {}
        self.grad_acts: Dict[int, torch.Tensor] = {}
        self._linears: Optional[List[nn.Linear]] = None

    def _merge(self, lst: List[torch.Tensor]) -> torch.Tensor:
        t = lst[0] if len(lst) == 1 else torch.cat(lst, dim=0)
        if t.ndim > 2:
            t = t.reshape(-1, t.shape[-1])
        return t

    def _consolidate(self) -> None:
        for k, lst in self._pre_list.items():
            self.pre_acts[k] = self._merge(lst)
        for k, lst in self._go_list.items():
            self.grad_acts[k] = self._merge(lst)
        self._pre_list.clear()
        self._go_list.clear()

    def attach_hooks(self, model: nn.Module) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self._pre_list = {}
        self._go_list = {}
        self.pre_acts = {}
        self.grad_acts = {}
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self._linears = linears
        for idx, layer in enumerate(linears):

            def make_fwd(i: int):
                def fwd(_m: nn.Module, inp: tuple, _out: torch.Tensor) -> None:
                    if not _m.training:
                        return
                    self._pre_list.setdefault(i, []).append(inp[0].detach())

                return fwd

            def make_bwd(i: int):
                def bwd(
                    _m: nn.Module,
                    _gi: tuple,
                    go: tuple,
                ) -> None:
                    if go[0] is not None:
                        self._go_list.setdefault(i, []).append(
                            go[0].detach()
                        )

                return bwd

            self.hooks.append(layer.register_forward_hook(make_fwd(idx)))
            self.hooks.append(layer.register_full_backward_hook(make_bwd(idx)))

    @torch.no_grad()
    def step(self, closure=None):
        if self._linears is None:
            raise RuntimeError("Call attach_hooks(model) before the first KFAC step.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._consolidate()
        if not self.pre_acts or not self.grad_acts:
            return loss

        for group in self.param_groups:
            lr = group["lr"]
            b = group["factor_decay"]
            dmp = group["damping"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            teps = group["eps"]
            inv_floor = group["inv_floor"]

            param_ids = {id(p) for p in group["params"]}

            for idx, layer in enumerate(self._linears):
                p = layer.weight
                if id(p) not in param_ids:
                    continue
                if p.grad is None:
                    continue
                g = p.grad
                if wd:
                    g = g + wd * p

                st = self.state[p]
                if "m_buf" not in st:
                    st["m_buf"] = torch.zeros_like(p)
                st["m_buf"].mul_(mom).add_(g)
                g_use = st["m_buf"]

                if idx not in self.pre_acts or idx not in self.grad_acts:
                    p.add_(g_use, alpha=-lr)
                    continue

                a = self.pre_acts[idx]
                g_o = self.grad_acts[idx]
                n = max(float(a.shape[0]), 1.0)
                ata = a.T @ a / n
                bmt = g_o.T @ g_o / n
                d_in, d_out = ata.shape[0], bmt.shape[0]
                if "A" not in st:
                    st["A"] = torch.zeros_like(ata)
                    st["B"] = torch.zeros_like(bmt)
                st["A"].mul_(b).add_(ata, alpha=1.0 - b)
                st["B"].mul_(b).add_(bmt, alpha=1.0 - b)

                tr_a = st["A"].trace() / max(d_in, 1)
                tr_b = st["B"].trace() / max(d_out, 1)
                pi = dmp * max((tr_a * tr_b) ** 0.5, 1e-8)
                inv_a = _inv_damped(st["A"], pi, teps, inv_floor)
                inv_b = _inv_damped(st["B"], pi, teps, inv_floor)
                upd = inv_b @ g_use @ inv_a
                p.add_(upd, alpha=-lr)

        return loss
