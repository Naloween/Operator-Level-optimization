"""The modal reduction, and the assumption it rests on, made measurable.

Everything in [`04-instability.md`](../../../theory/04-instability.md) §2–§3 needs one step
that is stated but not proved: that the exact per-mode velocity

    sdot_k = - sum_l u_k^T A_l A_l^T G B_l^T B_l v_k                                (exact)

may be replaced by its **diagonal** part,

    sdot_k ~= - c_k g_k,    c_k = sum_l ||A_l^T u_k||^2 ||B_l v_k||^2,  g_k = u_k^T G v_k.

The two agree exactly when `A_l A_l^T` and `B_l^T B_l` are diagonal in the operator's own
singular bases -- i.e. when each subproduct is aligned with the whole product. That is the
content of Theorem 6.1 of Haas et al. (ICML 2026), which derives it from spectral separation
and requires the subproduct to be large enough. It is an assumption here, not a result, and
this module measures how badly it holds rather than arguing it away.

Three quantities, all computable in any network with no hypothesis of their own:

* `diagonality` -- for each layer, `||diag(M)|| / ||M||_F` with `M = U^T A_l A_l^T U`,
  which is 1 exactly when the Gram matrix is diagonal in the operator's basis. Same for
  `B_l^T B_l` against `V`. This is the alignment claim, stated as a number.
* `reduction_error` -- the relative discrepancy between the exact velocity vector and its
  diagonal approximation. This is what the reduction actually costs, and it is the honest
  quantity: alignment matters only insofar as it changes `sdot`.
* `separation` -- `log(s_1/s_d)`, the variable Theorem 6.1 says drives alignment.

**The prediction to test.** If Theorem 6.1 transfers to the nonlinear setting, then
`reduction_error` should fall as `separation` grows, and the "large enough subproduct"
condition should show up as a floor below which it does not. Nothing here assumes that; the
point is to find out. It has, as far as I know, never been checked outside the fixed-gates
linear setting the theorem was proved in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ModalCheck:
    """How much the diagonal reduction costs, and how aligned the network is."""

    s: np.ndarray            # singular values
    exact: np.ndarray        # per-mode velocity, no approximation
    diagonal: np.ndarray     # -c_k g_k, the reduction file 04 uses
    diag_A: np.ndarray       # per-layer diagonality of A_l A_l^T in U
    diag_B: np.ndarray       # per-layer diagonality of B_l^T B_l in V

    @property
    def reduction_error(self) -> float:
        """`||exact - diagonal|| / ||exact||`. 0 means the reduction is free."""
        d = np.linalg.norm(self.exact)
        return float(np.linalg.norm(self.exact - self.diagonal) / d) if d > 0 else float("nan")

    @property
    def cosine(self) -> float:
        """cos between the exact and reduced velocity vectors: direction, not magnitude."""
        n = np.linalg.norm(self.exact) * np.linalg.norm(self.diagonal)
        return float(self.exact @ self.diagonal / n) if n > 0 else float("nan")

    @property
    def separation(self) -> float:
        nz = self.s[self.s > 0]
        return float(np.log(nz.max() / nz.min())) if nz.size > 1 else 0.0

    @property
    def alignment(self) -> float:
        """Worst per-layer diagonality over both sides. 1 = perfectly aligned."""
        return float(min(self.diag_A.min(), self.diag_B.min()))


def _diagonality(M: torch.Tensor) -> float:
    """`||diag(M)|| / ||M||_F` in [0, 1]; 1 exactly when M is diagonal."""
    f = float(M.norm())
    return float(M.diagonal().norm() / f) if f > 0 else float("nan")


@torch.no_grad()
def modal_reduction(net, X: torch.Tensor, G: torch.Tensor, sample: int = 0) -> ModalCheck:
    """Exact per-mode velocity against its diagonal reduction, plus the alignment numbers.

    `G` holds per-sample operator gradients. The expectation over inputs is uniform over the
    batch, matching a mean training loss, so `exact` is the master equation's right-hand
    side and `diagonal` is what file 04 §2 replaces it with.
    """
    gates = net.gates(X)
    J = net.operator(X, gates)
    J_k = (J[sample] if J.shape[0] > 1 else J[0]).double()
    U, S, Vh = torch.linalg.svd(J_k)
    V = Vh.T
    ctx = net.all_contexts(X, gates)
    B = X.shape[0]
    pick = lambda T, b: (T[b] if T.shape[0] > 1 else T[0]).double()

    k = S.shape[0]
    exact = torch.zeros(k, dtype=torch.float64)
    c = torch.zeros(k, dtype=torch.float64)
    dA, dB = [], []

    for l in range(net.depth):
        A_s, B_s = pick(ctx[l][0], sample), pick(ctx[l][1], sample)
        dA.append(_diagonality(U.T @ (A_s @ A_s.T) @ U))
        dB.append(_diagonality(V.T @ (B_s.T @ B_s) @ V))

        inner = torch.zeros(A_s.shape[1], B_s.shape[0], dtype=torch.float64)
        for b in range(B):
            inner += pick(ctx[l][0], b).T @ pick(G, b).double() @ pick(ctx[l][1], b).T
        M = A_s @ (inner / B) @ B_s
        exact -= (U[:, :k] * (M @ V[:, :k])).sum(0)
        c += (A_s.T @ U).pow(2).sum(0)[:k] * (B_s @ V).pow(2).sum(0)[:k]

    Gm = sum(pick(G, b).double() for b in range(B)) / B
    g = (U.T @ Gm @ V).diagonal()[:k]
    return ModalCheck(s=S.numpy(), exact=exact.numpy(), diagonal=(-c * g).numpy(),
                      diag_A=np.array(dA), diag_B=np.array(dB))
