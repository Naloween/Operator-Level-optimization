"""Deep linear network, exposing the same interface as `nonlinear.py` and `crelu.py`.

Present so the diagnostic runs unchanged on the case whose answer is known in closed form: under
balanced initialisation the induced gain on singular direction k is exactly `L s_k^{2-2/L}`
(Arora, Cohen & Hazan 2018, Thm 1). Any measurement pipeline that does not reproduce that is
wrong, so this is the validation, not an experiment.
"""
from __future__ import annotations
import numpy as np


def init_net(d_in, d, d_out, L, init, rng):
    dims = [d_in] + [d] * (L - 1) + [d_out]
    Ws = []
    for l in range(L):
        m, k = dims[l + 1], dims[l]
        if init == "xavier":
            Ws.append(rng.standard_normal((m, k)) * np.sqrt(2.0 / k))
        elif init == "orth":
            Q = np.linalg.qr(rng.standard_normal((max(m, k), max(m, k))))[0]
            Ws.append(Q[:m, :k])
        else:
            raise ValueError(init)
    return Ws


def balanced_net(d, L, spectrum, rng):
    """A BALANCED chain with a prescribed, non-flat spectrum.

    W_l = U_l S^{1/L} U_{l-1}^T with U_l Haar gives
    W_{l+1}^T W_{l+1} = U_l S^{2/L} U_l^T = W_l W_l^T, so the chain is balanced in the sense of
    Arora et al. Thm 1, and J = U_L S U_0^T has exactly the prescribed spectrum. An orthogonal
    chain is balanced too but its spectrum is FLAT, so no exponent can be fitted from it -- that
    is why `orth` is not the validation setting.
    """
    Us = [np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(L + 1)]
    S = np.diag(np.asarray(spectrum, dtype=float) ** (1.0 / L))
    return [Us[l + 1] @ S @ Us[l].T for l in range(L)]


def forward(Ws, X):
    J = Ws[0]
    for W in Ws[1:]:
        J = W @ J
    return X @ J.T, []


def operators(Ws, gates, n):
    L = len(Ws)
    B = [np.eye(Ws[0].shape[1])]
    for l in range(L - 1):
        B.append(Ws[l] @ B[-1])
    A = [np.eye(Ws[-1].shape[0])]
    for l in range(L - 1, 0, -1):
        A.append(A[-1] @ Ws[l])
    A = A[::-1]
    J = A[0] @ Ws[0] @ B[0]
    tile = lambda Z: np.tile(Z, (n, 1, 1))
    return tile(J), [tile(a) for a in A], [tile(b) for b in B]
