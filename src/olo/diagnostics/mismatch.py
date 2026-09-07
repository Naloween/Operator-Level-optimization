"""The operator-step mismatch: did the update move P where it was asked to?

This is the measurement the whole framework exists to make, and it needs no ALS solve --
only two operator evaluations around a step. It applies to any optimizer, which is what
makes it a diagnostic rather than a property of one method.

Three numbers, answering three different questions:

`cos_target`    cos(realized dP, -eta G). How much of the intended operator motion the
                update actually achieved. This is the quantity the paper's Table 1
                reports, and it is learning-rate invariant to first order because the
                positive scalar cancels under normalization -- so a comparison across
                methods at different learning rates is still meaningful.

`cos_gd_first_order`  cos(-sum_k A_k A_k^T G B_k^T B_k, -G), from Proposition 3.1. A
                property of the *current contexts alone*: computable before any step is
                taken, for any optimizer.

                It is first order only, and that limitation is not academic. At an
                identity initialization every context is I, so this term is -L eta G --
                exactly parallel to -G, reporting perfect alignment at every depth. The
                realized step is nonetheless misaligned, because the factors multiply
                rather than add: `(I - eta G)^L - I` carries every power of G, and G^2 is
                not parallel to G. Read `cos_gd_first_order` as "the distortion the
                contexts impose", never as "the mismatch"; `cos_target` is the mismatch.
                See `olo.theory.deep_linear` for the closed form and the scaling it gives.

`cos_linear`    cos(realized dP, sum_k A_k dW_k B_k). How well the first-order expansion
                describes what actually happened. It drops below 1 when the step is large
                enough that higher-order terms matter -- the regime where reasoning about
                the linearized objective stops being safe.
"""
from __future__ import annotations

import torch


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float((a.flatten() @ b.flatten()) / (na * nb))


@torch.no_grad()
def step_alignment(
    P_before: torch.Tensor,
    P_after: torch.Tensor,
    G: torch.Tensor,
    lr: float,
    contexts: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    dW: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    """Alignment of the realized operator motion with the intended one."""
    dP = (P_after - P_before).to(torch.float64)
    want = (-lr * G).to(torch.float64)

    out = {
        "cos_target": _cos(dP, want),
        "dP_norm": float(dP.norm()),
        "target_norm": float(want.norm()),
        "norm_ratio": float(dP.norm() / want.norm()) if want.norm() > 0 else float("nan"),
    }
    if contexts is not None and dW is not None:
        first = torch.zeros_like(dP)
        for (A, B), d in zip(contexts, dW):
            first = first + A.to(torch.float64) @ d.to(torch.float64) @ B.to(torch.float64)
        out["cos_linear"] = _cos(dP, first)
        out["linearization_error"] = float((dP - first).norm() / dP.norm()) if dP.norm() > 0 else 0.0
    return out


@torch.no_grad()
def gd_first_order_alignment(
    contexts: list[tuple[torch.Tensor, torch.Tensor]], G: torch.Tensor
) -> dict[str, float]:
    """Proposition 3.1: the operator motion gradient descent would induce from here.

        dP_GD = -eta sum_k A_k A_k^T G B_k^T B_k

    Agreement with -eta G for all G would require sum_k A_k A_k^T kron B_k^T B_k = I,
    which no training trajectory preserves. Computed from the contexts alone, so it can be
    logged for any optimizer -- including ones that are not gradient descent -- as a
    property of the geometry the network currently sits in.
    """
    G64 = G.to(torch.float64)
    dP = torch.zeros_like(G64)
    for A, B in contexts:
        A64, B64 = A.to(torch.float64), B.to(torch.float64)
        dP = dP + (A64 @ A64.transpose(-2, -1)) @ G64 @ (B64.transpose(-2, -1) @ B64)
    return {
        "cos_gd_first_order": _cos(dP, G64),
        "gd_gain": float(dP.norm() / G64.norm()) if G64.norm() > 0 else float("nan"),
    }
