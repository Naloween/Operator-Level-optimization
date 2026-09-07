"""Validity of the frozen-gate linearization.

Every operator-space method here writes its objective in terms of gates frozen at the
start of the step. That objective describes the network only while the step is small
relative to the pre-activation margins; past that, the solver can drive its own residual
to machine precision while the network it was supposed to be improving diverges. This has
been observed at depth 128 with an aggressive step (see `tests/test_als.py`), and without
this measurement it is indistinguishable from a solver bug.

Two numbers:

`gate_flip_rate`      fraction of gate entries that changed sign across the step. The
                      direct measure of how far the linearization was left behind.
`frozen_gate_error`   ||P_frozen - P_actual|| / ||P_actual||, where P_frozen recomposes
                      the *new* weights against the *old* gates. This is the error the
                      objective was blind to.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def linearization_validity(
    net, x: torch.Tensor, gates_before: list[torch.Tensor]
) -> dict[str, float]:
    """How badly the frozen gates misdescribe the network after the step."""
    if net.input_independent or not gates_before:
        return {"gate_flip_rate": 0.0, "frozen_gate_error": 0.0}

    gates_after = net.gates(x)
    flipped = total = 0
    for before, after in zip(gates_before, gates_after):
        flipped += int((before != after).sum())
        total += before.numel()

    P_frozen = net.operator(x, gates_before)
    P_actual = net.operator(x, gates_after)
    denom = float(P_actual.norm())
    return {
        "gate_flip_rate": flipped / total if total else 0.0,
        "frozen_gate_error": float((P_frozen - P_actual).norm()) / denom if denom > 0 else 0.0,
    }
