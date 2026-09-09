"""The mode ensemble: a source of randomness that survives training.

A *mode* is a gate pattern, chosen freely rather than realized by any input. For CReLU it is
exactly a sign vector `eps = (eps_1, ..., eps_{L-1})` in `{+-1}^{(L-1)d}`, because
`W_l D(z) = S_l + Delta_l diag(sign z)` depends on `z` only through its signs. So at any
time t the network *is* a family of deep linear networks indexed by sign vectors:

    J_eps = M_L(eps_{L-1}) ... M_1,     M_l(eps) = S_l + Delta_l diag(eps_l)

all sharing the same `S_l` and `Delta_l`. Verified exact: `J(x) = J_{eps(x)}` at the
realized signs, to 1e-16.

**Why this is a better object than fixing an input.** Fixing `x` does not fix the gates:
as the weights move, `sign(z_l(x))` flips, so "the Jacobian at x" is not a fixed-gates
product over time and its trajectory crosses region boundaries. Fixing a *mode* gives a
genuinely time-invariant object -- `J_eps(t)` is a smooth function of the weights for all t,
with no discontinuities to handle.

**Why it gives leverage.** Under Rademacher `eps`, `E[M_l] = S_l` with independent
fluctuation `Delta_l diag(eps_l)`, so `J_eps` is a product of independent random matrices --
Furstenberg/Oseledets territory -- with the randomness carried by the *mode* rather than by
the weights. The fixed-gates analysis of Haas et al. (ICML 2026) draws its randomness from
the initialization, which is why it speaks about `t = 0`; mode randomness is available at
every t, trained weights included.

**What is measured, and the one-sided hypothesis it suggests.** On trained CReLU networks,
realized gate patterns are close to Rademacher in low-order statistics -- mean sign at the
finite-sample noise floor, cross-input and cross-layer agreement within 0.02 of 0.5 -- but
their Jacobians are *more* spectrally separated than typical modes, by a factor of about
2.2 at depth 64. Realized modes are not typical; they are better than typical, in the
direction that helps.

That makes the useful hypothesis one-sided, and empirically true rather than merely
convenient:

    (H-mode)   sep(J_{eps(x)}) >= sep(J_eps) for typical eps

Under it, a separation lower bound for the random-mode ensemble -- where Lyapunov theory
applies -- transfers to the modes the data actually realizes. What this does *not* address
is the cross-input coupling in the weight gradient (`olo.theory.nonlinear`): the mode view
constrains the spectrum at fixed t, not the dynamics that couple modes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from olo.theory.crelu import linear_part, split_layer


def mode_factors(net) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """`(S_l, Delta_l)` for every layer; `Delta_0` is zero (the first layer has no gate)."""
    S = linear_part(net)
    D = [torch.zeros_like(S[0])] + [split_layer(W)[1] for W in net.weights[1:]]
    return S, D


@torch.no_grad()
def operator_of_mode(net, eps: list[torch.Tensor]) -> torch.Tensor:
    """`J_eps`: the Jacobian this network would have on gate pattern `eps`.

    `eps` holds `L-1` sign vectors. They need not be realized by any input.
    """
    S, D = mode_factors(net)
    if len(eps) != net.depth - 1:
        raise ValueError(f"need {net.depth - 1} sign vectors, got {len(eps)}")
    P = S[0]
    for l in range(1, net.depth):
        P = (S[l] + D[l] @ torch.diag(eps[l - 1].to(P.dtype))) @ P
    return P


@torch.no_grad()
def realized_modes(net, X: torch.Tensor) -> torch.Tensor:
    """Sign patterns the inputs actually produce: (B, L-1, width)."""
    zs = net.pre_activations(X)
    return torch.stack([torch.sign(torch.stack([zs[l][b] for l in range(net.depth - 1)]))
                        for b in range(X.shape[0])])


@torch.no_grad()
def random_modes(net, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """`n` Rademacher gate patterns, shaped like `realized_modes`."""
    shape = (n, net.depth - 1, net.width)
    bits = torch.randint(0, 2, shape, generator=generator, dtype=torch.float64)
    return bits * 2 - 1


@dataclass
class ModeStats:
    """How the realized modes compare with the Rademacher ensemble."""

    mean_sign: float          # 0 for Rademacher; the noise floor is 1/sqrt(B)
    input_agreement: float    # 0.5 for Rademacher
    layer_agreement: float    # 0.5 for Rademacher
    separation_random: float  # mean log(s1/s2) over random modes
    separation_realized: float

    @property
    def separation_ratio(self) -> float:
        """How much more separated the realized modes are. >1 supports (H-mode)."""
        d = self.separation_random
        return self.separation_realized / d if d > 0 else float("nan")


@torch.no_grad()
def compare(net, X: torch.Tensor, n_random: int = 32,
            generator: torch.Generator | None = None) -> ModeStats:
    """Measure realized modes against the Rademacher ensemble, including separation."""
    E = realized_modes(net, X)
    R = random_modes(net, n_random, generator)

    def sep(modes):
        out = []
        for m in modes:
            sv = torch.linalg.svdvals(operator_of_mode(net, [m[l] for l in range(net.depth - 1)]))
            if sv.shape[0] > 1 and float(sv[1]) > 0:
                out.append(float(torch.log(sv[0] / sv[1])))
        return sum(out) / len(out) if out else float("nan")

    n = min(X.shape[0], n_random)
    return ModeStats(
        mean_sign=float(E.mean(0).abs().mean()),
        input_agreement=float((E.unsqueeze(0) == E.unsqueeze(1)).double().mean()),
        layer_agreement=float((E[:, :-1, :] == E[:, 1:, :]).double().mean())
        if net.depth > 2 else float("nan"),
        separation_random=sep(R[:n]),
        separation_realized=sep(E[:n]),
    )


# -- dynamics at a fixed gate pattern ---------------------------------------
#
# NAMING. This module's "mode" is a **gate pattern** `eps` -- a sign vector per layer.
# `olo.theory.instability` and `olo.theory.imbalance` use "mode" for the k-th **singular
# direction** of an operator. They are unrelated objects and the collision is mine; in the
# code below the gate one is `pattern` and the spectral one is `direction k`.
#
# The point of working at a fixed pattern: `J_eps` is a polynomial in the weights with no
# discontinuity, so `t -> J_eps(t)` is smooth even where `t -> J(x)` jumps. And the weight
# gradient `Gamma_l` is one matrix shared by every pattern -- the patterns differ only in
# how they compose it. So the dynamics of every pattern is a fixed-gates linear network
# driven by the same `Gamma`, which is what makes the question tractable at all.


def gates_of_pattern(net, eps: torch.Tensor) -> list[torch.Tensor]:
    """The gate tensors a CReLU net would have on prescribed signs `eps`, shape (L-1, d).

    Returned in the layout `net.gates` uses, so `operator`, `all_contexts` and every
    diagnostic work at a prescribed pattern with no special-casing.
    """
    if eps.shape[0] != net.depth - 1:
        raise ValueError(f"need {net.depth - 1} sign vectors, got {eps.shape[0]}")
    out = []
    for z in eps:
        pos = torch.diag_embed((z > 0).to(torch.float64)).unsqueeze(0)
        neg = torch.diag_embed((z < 0).to(torch.float64)).unsqueeze(0)
        out.append(torch.cat([pos, -neg], dim=-2))
    return out


@torch.no_grad()
def pattern_dynamics(net, eps: torch.Tensor, grads: list[torch.Tensor],
                     x: torch.Tensor | None = None) -> dict:
    """Singular-value velocity of `J_eps` under the *realized* weight gradients.

    `grads[l]` is `Gamma_l = dL/dW_l`, computed from the real loss on the real data (so
    every cross-input effect is already inside it). Then, exactly,

        sdot_k(eps) = - sum_l u_k^T A_l^eps Gamma_l B_l^eps v_k

    with no approximation and no alignment assumption: `J_eps = A_l^eps W_l B_l^eps` holds
    by construction. Also returns the pattern's own mode gain
    `c_k = sum_l ||A_l^T u_k||^2 ||B_l v_k||^2` and the effective drive `g_k := -sdot_k/c_k`,
    which is defined so that `sdot_k = -c_k g_k` holds identically -- the split is a
    change of variables, not a hypothesis.
    """
    width = net.width
    gates = gates_of_pattern(net, eps)
    probe = torch.zeros(1, net.d_in, dtype=torch.float64) if x is None else x[:1]
    J = net.operator(probe, gates)[0].double()
    U, S, Vh = torch.linalg.svd(J)
    V = Vh.T
    ctx = net.all_contexts(probe, gates)

    k = S.shape[0]
    sdot = torch.zeros(k, dtype=torch.float64)
    c = torch.zeros(k, dtype=torch.float64)
    for l in range(net.depth):
        A = ctx[l][0][0].double()
        B = ctx[l][1][0].double()
        sdot -= (U[:, :k] * ((A @ grads[l].double() @ B) @ V[:, :k])).sum(0)
        c += (A.T @ U).pow(2).sum(0)[:k] * (B @ V).pow(2).sum(0)[:k]
    g = torch.where(c > 0, -sdot / c.clamp_min(1e-300), torch.zeros_like(c))
    return {"s": S.numpy(), "sdot": sdot.numpy(), "gain": c.numpy(), "drive": g.numpy(),
            "width": width}


def hamming_mix(realized: torch.Tensor, random_: torch.Tensor, frac: float,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Interpolate between two patterns by flipping `frac` of the realized signs.

    `frac = 0` is the realized pattern, `frac = 1` an independent Rademacher one, and the
    values between trace out the path from patterns the data actually produces to typical
    ones -- which is the axis (H-mode) in `theory/03-dynamics.md` is stated along.
    """
    mask = torch.rand(realized.shape, generator=generator) < frac
    return torch.where(mask, random_, realized)
