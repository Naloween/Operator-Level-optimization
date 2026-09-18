"""CReLU MLP: forward pass, per-input operator, and contexts.

Layer map. With width `d`, the doubled feature is `c(z) = [relu(z); relu(-z)] in R^{2d}` and
`D(z) = [diag(1[z>0]) ; -diag(1[z<0])] in R^{2d x d}` satisfies `c(z) = D(z) z` and
`D(z)^T D(z) = I`, so the gate is an isometry. Shapes:

    W_1 in R^{d x d_in},   W_l in R^{d x 2d} (1 < l < L),   W_L in R^{d_out x 2d}
    z_1 = W_1 x,  c_l = D(z_l) z_l,  z_{l+1} = W_{l+1} c_l,  f(x) = W_L c_{L-1}

so `f(x) = J(x) x` exactly with `J(x) = W_L D(z_{L-1}) W_{L-1} ... D(z_1) W_1`.

Contexts, same convention as `nonlinear.py`: `J(x) = A_l(x) W_l B_l(x)` with
`B_1 = I`, `B_{l+1} = D(z_l) W_l B_l`, `A_L = I`, `A_l = A_{l+1} W_{l+1} D(z_l)`.
"""
from __future__ import annotations
import numpy as np


def init_net(d_in, d, d_out, L, init, rng):
    dims_in = [d_in] + [2 * d] * (L - 1)
    dims_out = [d] * (L - 1) + [d_out]
    Ws = []
    for l in range(L):
        m, k = dims_out[l], dims_in[l]
        if init == "xavier":
            Ws.append(rng.standard_normal((m, k)) * np.sqrt(2.0 / k))
        elif init == "looks_linear":
            # W = [O | -O] with O Haar: then W D(z) = O for every z, so J is orthogonal
            if l == 0:
                Q = np.linalg.qr(rng.standard_normal((max(m, k), max(m, k))))[0]
                Ws.append(Q[:m, :k])
            else:
                h = k // 2
                Q = np.linalg.qr(rng.standard_normal((max(m, h), max(m, h))))[0]
                O = Q[:m, :h]
                Ws.append(np.concatenate([O, -O], axis=1))
        else:
            raise ValueError(init)
    return Ws


def _gate_apply(z, V):
    """D(z) @ V  for V of shape (n, d, q); returns (n, 2d, q)."""
    pos = (z > 0).astype(V.dtype)[:, :, None]
    neg = (z < 0).astype(V.dtype)[:, :, None]
    return np.concatenate([pos * V, -neg * V], axis=1)


def _gate_rapply(A, z):
    """A @ D(z)  for A of shape (n, p, 2d); returns (n, p, d)."""
    d = z.shape[1]
    pos = (z > 0).astype(A.dtype)[:, None, :]
    neg = (z < 0).astype(A.dtype)[:, None, :]
    return A[:, :, :d] * pos - A[:, :, d:] * neg


def forward(Ws, X):
    """Returns (output, list of pre-activations z_1..z_{L-1})."""
    L = len(Ws)
    z = X @ Ws[0].T
    zs = []
    for l in range(1, L):
        zs.append(z)
        c = np.concatenate([np.maximum(z, 0.0), np.maximum(-z, 0.0)], axis=1)
        z = c @ Ws[l].T
    return z, zs


def operators(Ws, zs, n):
    """J(x), A_l(x), B_l(x) with J = A_l W_l B_l."""
    L = len(Ws)
    d_in, d_out = Ws[0].shape[1], Ws[-1].shape[0]
    B = [np.tile(np.eye(d_in), (n, 1, 1))]
    for l in range(L - 1):
        WB = np.einsum("ij,njk->nik", Ws[l], B[-1])          # W_l B_l : (n, d, d_in)
        B.append(_gate_apply(zs[l], WB))                     # D(z_l) W_l B_l : (n, 2d, d_in)
    A = [np.tile(np.eye(d_out), (n, 1, 1))]
    for l in range(L - 1, 0, -1):
        AW = np.einsum("nij,jk->nik", A[-1], Ws[l])          # A_{l+1} W_{l+1} : (n, d_out, 2d)
        A.append(_gate_rapply(AW, zs[l - 1]))                # ... D(z_l) : (n, d_out, d)
    A = A[::-1]
    J = np.einsum("nij,jk,nkl->nil", A[0], Ws[0], B[0])
    return J, A, B
