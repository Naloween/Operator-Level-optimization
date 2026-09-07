"""The collapse monitor: is the layer solve still doing anything?

The modewise ALS filter divides each mode by sigma_A^2 sigma_B^2 + lam. When the contexts
collapse, that denominator is dominated by lam for essentially every mode and the update
degenerates to a uniform rescaling of the gradient with no operator-correcting structure
left. The solver does not fail loudly when this happens -- it returns a perfectly valid
answer to a problem that has stopped carrying information.

    rho_k = (min_i sigma_{A,i}^2)(min_j sigma_{B,j}^2) / lam

with rho_k << 1 flagging that layer. Structural zeros -- modes a gate removes by
construction, which no initialization can restore -- are excluded, so the monitor reports
collapse rather than architecture.

Reported for every optimizer, not just ALS: the same context geometry decides whether any
method's update can reach a given operator direction.
"""
from __future__ import annotations

import torch

#: modes below this fraction of the largest are treated as structurally absent
STRUCTURAL_TOL = 1e-10


@torch.no_grad()
def context_conditioning(
    contexts: list[tuple[torch.Tensor, torch.Tensor]], lam: float
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Per-layer collapse statistics, plus the spectra behind them."""
    rho, cond_A, cond_B, min_sv = [], [], [], []
    for A, B in contexts:
        sa = _spectrum(A)
        sb = _spectrum(B.transpose(-2, -1))
        a_min, a_max = _live_range(sa)
        b_min, b_max = _live_range(sb)
        # With no regularizer -- and for every parameter-space method, which has no lam at
        # all -- rho is reported as the bare context product. It still says how much
        # context a layer's weakest live mode has; it just is not a ratio against lam.
        denom = lam if lam > 0 else 1.0
        rho.append((a_min ** 2) * (b_min ** 2) / denom)
        cond_A.append(a_max / a_min if a_min > 0 else float("inf"))
        cond_B.append(b_max / b_min if b_min > 0 else float("inf"))
        min_sv.append(min(a_min, b_min))

    rho_t = torch.tensor(rho, dtype=torch.float64)
    scalars = {
        "rho_min": float(rho_t.min()),
        "rho_median": float(rho_t.median()),
        "collapsed_layers": int((rho_t < 1.0).sum()),
        "collapsed_fraction": float((rho_t < 1.0).to(torch.float64).mean()),
        "context_min_sv": float(min(min_sv)),
        "context_cond_max": float(max(max(cond_A), max(cond_B))),
    }
    arrays = {
        "rho_per_layer": rho_t,
        "cond_A_per_layer": torch.tensor(cond_A, dtype=torch.float64),
        "cond_B_per_layer": torch.tensor(cond_B, dtype=torch.float64),
    }
    return scalars, arrays


def _spectrum(M: torch.Tensor) -> torch.Tensor:
    """Singular values of a context, averaged over samples when it is per-sample."""
    sv = torch.linalg.svdvals(M.to(torch.float64))
    return sv.mean(dim=0) if sv.dim() > 1 else sv


def _live_range(sv: torch.Tensor) -> tuple[float, float]:
    """(min, max) over modes that are not structurally zero."""
    if sv.numel() == 0:
        return 0.0, 0.0
    top = float(sv.max())
    if top == 0.0:
        return 0.0, 0.0
    live = sv[sv > STRUCTURAL_TOL * top]
    return (float(live.min()) if live.numel() else 0.0), top
