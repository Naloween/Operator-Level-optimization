"""Singular-value dynamics of a CReLU Jacobian, and the one hypothesis they need.

The goal is the low-rank bias itself: show that for *every* input x, the singular values of
J(x) obey a rich-get-richer law `s_k' ∝ s_k^{2-2/L}`, which is what produces exponential
spectral separation. Not shared subspaces across inputs, not closeness to a linear network
-- just the bias, per input.

**The master equation.** For fixed gates (so, away from region boundaries, a.e. in t),
gradient flow on the weights gives `J(x)' = sum_l A_l(x) W_l' B_l(x)`, and for a simple
singular value `s_k(x) = u_k(x)^T J(x) v_k(x)`,

    s_k(x)' = - sum_l E_{x'} [ u_k(x)^T A_l(x) A_l(x')^T G(x') B_l(x')^T B_l(x) v_k(x) ]

with `G(x') = d l / d J(x')` the per-sample operator gradient. This is exact -- verified
against finite differences in `tests/test_theory_nonlinear.py`.

**Why this is the right object.** For a *fixed* x, `J(x) = W_L D_{L-1}(x) ... D_1(x) W_1` is
precisely a fixed-gates linear network -- the object of Haas et al. (ICML 2026). Their
Theorem 5.3 (depth scaling) and Theorem 6.1 (separation forces alignment) therefore apply
per input, with no modification. Nothing about the nonlinearity touches them, because at
fixed x there is no nonlinearity left.

What the nonlinear setting *does* add is entirely in the expectation over `x'`: the weight
gradient averages over inputs whose contexts differ. Split the sum accordingly:

    s_k(x)' = self_k(x) + cross_k(x)

    self_k(x)  = -(1/N) sum_l u_k^T A_l(x) A_l(x)^T G(x) B_l(x)^T B_l(x) v_k
    cross_k(x) = -(1/N) sum_{x' != x} sum_l u_k^T A_l(x) A_l(x')^T G(x') B_l(x')^T B_l(x) v_k

The `self` term has exactly the structure the balanced/fixed-gates analyses handle: the same
`A_l A_l^T (.) B_l^T B_l` sandwich as Proposition 4.5, so under depth scaling and alignment
it reduces to `-L s_k^{2-2/L} <G(x), u_k v_k^T>` up to the paper's constants -- the
rich-get-richer law, per input.

So **the entire nonlinear extension reduces to controlling one object**: the cross-input
context overlap `A_l(x) A_l(x')^T` and `B_l(x')^T B_l(x)`. That is the honest location of
the remaining difficulty, and it is a single measurable quantity rather than a diffuse
"nonlinearity is hard".

**Candidate hypotheses, and what measurement says about them.** The bias would survive if

    (H-small)    |cross_k(x)| <= eps (N-1) |self_k(x)|                 -- inputs decouple
    (H-aligned)  cross_k(x) = c(x) self_k(x) (1 + O(eps)), c > -1      -- inputs reinforce

Under either, the `s_k^{2-2/L}` scaling of the self term carries through with a modified
rate. **Both were checked on trained CReLU networks and neither holds cleanly.** Under plain
gradient descent with `L*eta` held fixed, depths 4-64, up to 4000 steps: the per-input
coupling sits at 0.07-0.7 (not negligible), and `cos(self, cross)` scatters over
[-0.93, +0.85] with mean -0.02 and essentially no correlation with spectral separation
(+0.06). An earlier reading that suggested separation drives alignment came from a run that
was diverging, and does not survive a stable step size.

So the honest position is: the reduction below is rigorous and the difficulty is correctly
localized, but no sufficient condition on the gate patterns has yet been found that both
implies the bias and holds in practice. The next candidate, untested here, is that the cross
terms behave as mean-zero fluctuations across inputs and partially cancel, giving a
`sqrt(N)` rather than `N` contribution -- which would make the statement probabilistic
rather than deterministic.

`decompose` computes both terms exactly, so any such hypothesis is checked rather than
assumed. That is the point of this module: the uncertainty is isolated into one measurable
quantity per (input, mode) instead of being spread across "the nonlinear case is hard".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VelocityTerms:
    """Per-mode decomposition of `s_k(x)'` at one input."""

    s: torch.Tensor           # singular values s_k(x)
    self_term: torch.Tensor   # the fixed-gates term the linear theory covers
    cross_term: torch.Tensor  # the cross-input coupling, new to the nonlinear setting
    total: torch.Tensor       # self + cross = s_k(x)'

    #: number of other inputs the cross term sums over, for per-input normalization
    n_other: int = 1

    @property
    def ratio(self) -> torch.Tensor:
        """Per-input coupling strength: |cross| / ((N-1) |self|).

        Normalized by the number of other inputs, because `cross` sums over all of them
        while `self` is one. Without that, a ratio near N-1 -- meaning every input
        contributes about as much as the reference one -- would look like domination.
        Values well below 1 are the decoupled regime (H-small).
        """
        denom = (self.self_term.abs() * max(self.n_other, 1)).clamp(min=1e-300)
        return self.cross_term.abs() / denom

    @property
    def cosine(self) -> float:
        """cos between the self and cross vectors over modes: 1 means they reinforce (H-aligned)."""
        a, b = self.self_term, self.cross_term
        n = a.norm() * b.norm()
        return float((a @ b) / n) if n > 0 else float("nan")

    @property
    def scaling_exponent(self) -> float:
        """Slope of log|s_k'| against log s_k: the rich-get-richer exponent, empirically.

        The prediction from the balanced/fixed-gates analysis is `2 - 2/L`. Estimated over
        the modes whose velocity is resolvable.
        """
        m = (self.s > 1e-12) & (self.total.abs() > 1e-14)
        if int(m.sum()) < 3:
            return float("nan")
        x = torch.log(self.s[m].double())
        y = torch.log(self.total[m].abs().double())
        x = x - x.mean()
        var = float(x @ x)
        # An isometric spectrum has no spread in log s, so the slope is unidentifiable --
        # which is the situation at looks-linear initialization, not a large exponent.
        if var < 1e-6:
            return float("nan")
        return float((x @ (y - y.mean())) / var)


@torch.no_grad()
def _sandwich(u, A_x, inner, B_x, v) -> float:
    return float(u @ (A_x @ inner @ B_x) @ v)


def decompose(net, X: torch.Tensor, G: torch.Tensor, sample: int = 0) -> VelocityTerms:
    """Exact self/cross split of `s_k(x)'` at input `X[sample]` under gradient flow.

    `G` holds the per-sample operator gradients `G(x_b) = d l_b / d J(x_b)`, shape
    (B, n_L, n_0). The expectation over inputs is uniform over the batch, matching a mean
    training loss.
    """
    with torch.no_grad():
        gates = net.gates(X)
        ctx = net.all_contexts(X, gates)
        J = net.operator(X, gates)
        J_x = J[sample] if J.shape[0] > 1 else J[0]
        U, S, Vh = torch.linalg.svd(J_x)
        B = X.shape[0]
        pick = lambda T, b: T[b] if T.shape[0] > 1 else T[0]

        n_modes = S.shape[0]
        self_t = torch.zeros(n_modes, dtype=torch.float64)
        cross_t = torch.zeros(n_modes, dtype=torch.float64)

        for l in range(net.depth):
            A, Bc = ctx[l]
            A_x, B_x = pick(A, sample).double(), pick(Bc, sample).double()
            own = A_x.T @ G[sample].double() @ B_x.T
            other = torch.zeros_like(own)
            for bp in range(B):
                if bp == sample:
                    continue
                other += pick(A, bp).double().T @ G[bp].double() @ pick(Bc, bp).double().T

            for k in range(n_modes):
                u, v = U[:, k].double(), Vh[k, :].double()
                self_t[k] -= _sandwich(u, A_x, own / B, B_x, v)
                cross_t[k] -= _sandwich(u, A_x, other / B, B_x, v)

    return VelocityTerms(s=S.double(), self_term=self_t, cross_term=cross_t,
                         total=self_t + cross_t, n_other=max(X.shape[0] - 1, 1))


def operator_gradients(net, X: torch.Tensor, Y: torch.Tensor, task) -> torch.Tensor:
    """Per-sample `G(x_b) = d l_b / d J(x_b)`, with any batch averaging undone."""
    return task.operator_gradient(net, X, Y)
