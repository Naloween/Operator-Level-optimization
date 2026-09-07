"""Spectrum of the composed operator over training.

The operator's singular values are where the collapse-and-recovery transient shows up:
starting from a perfectly flat spectrum, parameter-space optimizers have been observed to
destroy the tail and then spend the rest of training rebuilding it. A loss curve cannot
show that; this can.

`effective_rank` is the entropy rank exp(-sum p_i log p_i) with p = sigma / sum(sigma),
which unlike a thresholded count moves continuously as mass shifts between modes.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def operator_spectrum(P: torch.Tensor, max_samples: int = 4):
    """Spectral statistics of P, averaged over samples when the operator is per-sample."""
    P64 = P.to(torch.float64)
    if P64.shape[0] > max_samples:
        P64 = P64[:max_samples]
    sv = torch.linalg.svdvals(P64)                     # (S, r)

    total = sv.sum(dim=-1, keepdim=True).clamp(min=1e-300)
    p = sv / total
    entropy = -(p * p.clamp(min=1e-300).log()).sum(dim=-1)
    eff_rank = entropy.exp()
    stable_rank = (sv ** 2).sum(-1) / (sv[..., 0] ** 2).clamp(min=1e-300)

    sv_mean = sv.mean(dim=0)
    scalars = {
        "sv_max": float(sv_mean[0]),
        "sv_min": float(sv_mean[-1]),
        "condition": float(sv_mean[0] / sv_mean[-1]) if sv_mean[-1] > 0 else float("inf"),
        "effective_rank": float(eff_rank.mean()),
        "stable_rank": float(stable_rank.mean()),
        "operator_norm": float(P64.flatten(1).norm(dim=1).mean()),
    }
    return scalars, {"singular_values": sv_mean}
