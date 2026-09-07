"""Exact operator projection by Alternating Least Squares -- the method.

Given a target operator, find factor increments whose product realizes it:

    min_{dW}  || prod_l (W_l + dW_l) - P_tgt ||_F^2  +  lam sum_l ||dW_l||_F^2

Each layer's sub-problem is a Tikhonov least-squares problem with a closed-form solution
(`olo.optim.linalg`); ALS cycles the layers, and the whole thing is exact per block. No
separable approximation of the layer system is made anywhere -- that approximation is what
K-FAC is, and keeping it out is what makes this a control rather than another heuristic.

Sweep order and cost. Sweeps run *downward*, k = L..1, which is not merely a convention:

  - A_k = W_L D_{L-1} ... W_{k+1} D_k depends on layers above k, all already updated this
    sweep, and satisfies A_{k-1} = A_k W_k D_{k-1} -- one matmul per layer.
  - B_k = D_{k-1} W_{k-1} ... W_1 depends on layers below k, none of which the sweep has
    touched yet, so the whole stack is computed once up front by a prefix scan.

So a sweep costs O(L) matmuls plus one solve per layer, rather than the O(L^2) of
rebuilding both contexts per layer. That is what makes depth 1024 reachable.

Gates are frozen for the duration of an outer step (the frozen-gate linearization) and
recomputed on the next step.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

from olo.models.base import FactoredNet
from olo.optim.base import Optimizer
from olo.optim.linalg import solve_kron_sum, solve_sylvester

#: Per-sample layer systems are dense and O(flat^3) to solve, with flat = n_out * n_in.
#: Past a few thousand this stops being a computation and becomes a wall; the default is
#: set where an exact solve is still seconds rather than hours.
DEFAULT_MAX_EXACT_DIM = 2048


class OperatorALS(Optimizer):
    display_name = "ALS-exact"

    def __init__(
        self,
        net: FactoredNet,
        lr: float = 1.0,
        lam: float = 1e-4,
        n_sweeps: int = 1,
        target: Any = "gradient",
        layers: str | Iterable[int] = "auto",
        max_exact_dim: int = DEFAULT_MAX_EXACT_DIM,
        outer: dict | str | None = None,
        solve_dtype: str = "auto",
    ) -> None:
        super().__init__(net, lr)
        self.lam = float(lam)
        self.n_sweeps = int(n_sweeps)
        self.max_exact_dim = int(max_exact_dim)
        self.target = _build_target(target)
        self.solve_dtype = solve_dtype

        self.als_layers, self.outer_layers = self._assign_layers(layers)
        self.outer = self._build_outer(outer)
        if self.outer_layers and self.outer is None:
            shapes = [tuple(net.weights[k].shape) for k in self.outer_layers]
            raise ValueError(
                f"layers {self.outer_layers} (shapes {shapes}) exceed max_exact_dim="
                f"{self.max_exact_dim} for the exact per-sample solve, and no `outer` "
                "optimizer was configured to handle them. Either set "
                "`optim.outer: {type: adam, lr: 1.0e-3}`, reduce the width, or raise "
                "max_exact_dim (cost grows as flat^3, memory as flat^2)."
            )

    # -- layer assignment ---------------------------------------------------

    def _assign_layers(self, layers) -> tuple[list[int], list[int]]:
        """Split layers into those solved exactly by ALS and those left to `outer`.

        Input-independent models have one shared context pair, so their layer solve is the
        O(d^3) modewise filter and no layer is ever too large. Only the per-sample path
        builds a dense (n*m)^2 system, so only it can be priced out.
        """
        all_idx = list(range(self.net.depth))
        if layers == "all":
            return all_idx, []
        if not isinstance(layers, str):
            chosen = sorted(int(k) for k in layers)
            bad = [k for k in chosen if not 0 <= k < self.net.depth]
            if bad:
                raise ValueError(f"layer indices out of range for depth {self.net.depth}: {bad}")
            return chosen, [k for k in all_idx if k not in set(chosen)]
        if layers != "auto":
            raise ValueError(f"layers must be 'auto', 'all', or a list of indices; got {layers!r}")

        if self.net.input_independent:
            return all_idx, []
        als, outer = [], []
        for k, W in enumerate(self.net.weights):
            (als if W.shape[0] * W.shape[1] <= self.max_exact_dim else outer).append(k)
        return als, outer

    def _build_outer(self, outer) -> torch.optim.Optimizer | None:
        if outer is None or not self.outer_layers:
            return None
        cfg = {"type": outer} if isinstance(outer, str) else dict(outer)
        kind = cfg.pop("type", "adam")
        params = [self.net.weights[k] for k in self.outer_layers]
        ctor = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "heavyball": lambda p, **kw: torch.optim.SGD(p, momentum=kw.pop("momentum", 0.9), **kw),
        }.get(kind)
        if ctor is None:
            raise ValueError(f"unknown outer optimizer {kind!r}")
        cfg.setdefault("lr", 1e-3)
        return ctor(params, **cfg)

    # -- the step -----------------------------------------------------------

    def step(self, x, y, task) -> dict[str, float]:
        net = self.net
        loss = task.loss(task.forward(net, x), y)

        G = task.operator_gradient(net, x, y)
        if self.outer is not None:
            self.outer.zero_grad(set_to_none=True)
            loss.backward()                      # before ALS mutates the weights in place

        with torch.no_grad():
            gates = net.gates(x)                 # frozen for the whole outer step
            P = net.operator(x, gates)
            P_tgt = self.target.propose(P, G, self.lr)

            W0 = [W.detach().clone() for W in net.weights]   # regularizer anchor
            for _ in range(self.n_sweeps):
                self._sweep(x, P_tgt, gates, W0)

            P_new = net.operator(x, gates)
            resid = _rel(P_new - P_tgt, P_tgt)

        if self.outer is not None:
            self.outer.step()

        return {
            "loss": float(loss.detach()),
            "als_residual": resid,               # how well the projection itself was solved
            "target_norm": float(P_tgt.norm()),
        }

    def _sweep(self, x, P_tgt, gates, W0: list[torch.Tensor]) -> None:
        """One reverse Gauss-Seidel sweep, with the regularizer anchored at `W0`.

        The objective penalizes `lam ||W_k - W_k^0||^2` where `W_k^0` is the weight at the
        *start of the outer step* -- not the increment this particular sweep happens to
        add. The distinction only appears once `n_sweeps > 1`, and getting it wrong is
        invisible: solving each sweep for a freely-regularized increment lets the total
        displacement grow without bound across sweeps, so the effective lam anneals toward
        zero and `n_sweeps` silently becomes a second, undeclared step-size knob.

        Writing `delta` for this sweep's increment and `D = W_k - W_k^0` for the
        displacement already accumulated, the anchored sub-problem is

            min_delta ||A delta B - R||^2 + lam ||D + delta||^2

        whose normal equation is the usual one with `- lam D` added to the right-hand
        side. At the first sweep `D = 0` and this reduces to the plain solve.
        """
        net = self.net
        Ws = net.weights
        Bstack = net.right_contexts(x, gates)    # valid for the whole downward sweep
        als = set(self.als_layers)

        n_out = Ws[-1].shape[0]
        A = torch.eye(n_out, device=Ws[0].device, dtype=Ws[0].dtype).unsqueeze(0)
        for k in range(net.depth - 1, -1, -1):
            if k in als:
                B = Bstack[k]
                R = P_tgt - A @ Ws[k] @ B
                displacement = Ws[k] - W0[k]
                Ws[k].add_(self._solve_layer(A, B, R, displacement).to(Ws[k].dtype))
            if k > 0:
                A = A @ (Ws[k] @ gates[k - 1])   # A_{k-1} = A_k W_k D_{k-1}, updated W_k

    def _solve_layer(self, A, B, R, displacement) -> torch.Tensor:
        """The layer sub-problem, by whichever exact route the contexts allow."""
        wdt = self._work_dtype(A.dtype)
        A, B, R = A.to(wdt), B.to(wdt), R.to(wdt)
        anchor = self.lam * displacement.to(wdt)     # the `- lam D` anchoring term

        if A.shape[0] == 1 and B.shape[0] == 1 and R.shape[0] == 1:
            a, b, r = A[0], B[0], R[0]
            return solve_sylvester(a.T @ a, b @ b.T, a.T @ r @ b.T - anchor, self.lam)

        S = max(A.shape[0], B.shape[0], R.shape[0])
        A = A.expand(S, *A.shape[1:])
        B = B.expand(S, *B.shape[1:])
        R = R.expand(S, *R.shape[1:])
        G = (A.transpose(1, 2) @ R @ B.transpose(1, 2)).mean(dim=0) - anchor
        return solve_kron_sum(A, B, G, self.lam)

    def _work_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Promote to float64 where the modewise filter would otherwise overflow.

        With a small lam, 1/(sigma_A^2 sigma_B^2 + lam) is enormous for collapsed modes;
        in float32 the resulting increments overflow when re-composed across many layers,
        which shows up as a divergence that looks like an optimization failure but is
        arithmetic.
        """
        if self.solve_dtype == "float64":
            return torch.float64
        if self.solve_dtype == "model":
            return dtype
        if self.solve_dtype != "auto":
            raise ValueError(f"solve_dtype must be auto|float64|model, got {self.solve_dtype!r}")
        return torch.float64 if (dtype == torch.float32 and self.lam <= 1e-3) else dtype

    # -- bookkeeping --------------------------------------------------------

    def describe(self) -> dict[str, object]:
        return {
            "type": "als",
            "lr": self.lr,
            "lam": self.lam,
            "n_sweeps": self.n_sweeps,
            "als_layers": list(self.als_layers),
            "outer_layers": list(self.outer_layers),
            "target": self.target.describe(),
            "solve_dtype": self.solve_dtype,
        }


def _build_target(target):
    from olo.config import Spec
    from olo.optim.targets import OperatorTarget
    from olo.registry import build

    if isinstance(target, OperatorTarget):
        return target
    return build("target", Spec.parse(target, where="optim.target"))


def _rel(diff: torch.Tensor, ref: torch.Tensor) -> float:
    denom = float(ref.norm())
    return float(diff.norm()) / denom if denom > 0 else float(diff.norm())
