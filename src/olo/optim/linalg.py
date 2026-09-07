"""Linear algebra for the ALS layer sub-problems.

Two exact solvers for the same Tikhonov least-squares problem, differing only in whether
the contexts are shared across the batch:

    shared:      min_dW  ||A dW B - R||_F^2 + lam ||dW||_F^2
    per-sample:  min_dW  (1/B) sum_b ||A_b dW B_b - R_b||_F^2 + lam ||dW||_F^2

The shared case decouples in the SVD bases of the context Grams and costs O(d^3); the
per-sample case is a *sum* of Kronecker products, does not decouple, and costs O((n m)^3).
Both are exact -- no separable approximation is made anywhere in this file, which is the
whole point of the method.

Numerical care taken here is not incidental: at depth, context Grams routinely have many
exactly-zero or denormal singular values (dead ReLU units, collapsed products), which
makes LAPACK's symmetric eigensolver fail outright. Every fallback below exists because it
was needed.
"""
from __future__ import annotations

import torch


def svd_psd(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigendecomposition of a symmetric PSD matrix as (vectors, values), descending.

    Uses the general SVD rather than `eigh`: deep rectifier networks produce Grams with
    many repeated near-zero singular values, for which LAPACK's `dsyevd` fails to converge
    while `dgesdd` succeeds. Falls back to a ridged retry, then to a CPU eigensolve.
    """
    try:
        U, s, _ = torch.linalg.svd(X)
        return U, s
    except torch._C._LinAlgError:
        pass

    scale = max(float(X.diagonal().abs().mean()), 1e-30)
    eye = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    for mult in (1e-6, 1e-4, 1e-2):
        try:
            U, s, _ = torch.linalg.svd(X + (scale * mult) * eye)
            return U, s
        except torch._C._LinAlgError:
            continue

    X_cpu = X.detach().to("cpu", torch.float64)
    evals, evecs = torch.linalg.eigh(
        X_cpu + scale * 1e-6 * torch.eye(X_cpu.shape[0], dtype=torch.float64)
    )
    order = torch.argsort(evals.clamp(min=0.0), descending=True)
    return (evecs[:, order].to(X.device, X.dtype), evals[order].to(X.device, X.dtype))


def solve_sylvester(
    M: torch.Tensor, N: torch.Tensor, C: torch.Tensor, lam: float
) -> torch.Tensor:
    """Solve  M X N + lam X = C  for X, with M, N symmetric PSD.

    With M = V diag(a) V^T and N = U diag(b) U^T, the change of variables Y = V^T X U
    makes the equation elementwise:

        Y_ij = (V^T C U)_ij / (a_i b_j + lam)

    This is the modewise filter of the paper: each mode of the projected residual is
    divided by the product of its context singular values. Modes whose contexts have
    collapsed sit below lam and are suppressed -- which is a property of the problem, not
    of this solver, and is what `olo.diagnostics.conditioning` measures.
    """
    V, a = svd_psd(_jitter(M))
    U, b = svd_psd(_jitter(N))
    F = V.T @ C @ U
    denom = a.unsqueeze(1) * b.unsqueeze(0) + lam

    if lam == 0.0:
        # No regularizer to pick a null-space representative: drop modes that carry no
        # context, rather than dividing by (numerical) zero.
        tol = max(float(a[0] * b[0]) * 1e-12, 1e-300)
        Y = torch.where(denom >= tol, F / denom.clamp(min=tol), torch.zeros_like(F))
    else:
        Y = F / denom.clamp(min=1e-300)
    return V @ Y @ U.T


def solve_kron_sum(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, lam: float
) -> torch.Tensor:
    """Exact solve of the per-sample normal equations.

        [ (1/S) sum_s (B_s B_s^T) kron (A_s^T A_s) + lam I ] vec(X) = vec(C)

    A is (S, p, n), B is (S, m, q), C is (n, m); returns X of shape (n, m).

    The system matrix is a *sum* of Kronecker products, one per sample, and does not
    factor as a single Kronecker product unless the per-sample Grams are proportional.
    Approximating it by one (the K-FAC / ALS-MN move) is exactly what this repo declines
    to do, so the system is formed and solved densely, in float64, at O((nm)^3).

    The sum over samples is a single contraction, not a loop. Writing
    `kron(B_s, A_s)[i*n+k, j*n+l] = B_s[i,j] A_s[k,l]`, the batch mean is an einsum whose
    `s` index is contracted away, so the whole system is formed in one call. At a batch of
    128 that replaces 128 Python-level `kron` calls per layer per sweep, which at depth is
    the difference between an experiment and an overnight job.
    """
    S = A.shape[0]
    n, m = C.shape
    A64, B64 = A.to(torch.float64), B.to(torch.float64)

    ATA = A64.transpose(1, 2) @ A64                # (S, n, n)
    BBT = B64 @ B64.transpose(1, 2)                # (S, m, m)
    H = torch.einsum("sij,skl->ikjl", BBT, ATA).reshape(n * m, n * m) / S

    H.diagonal().add_(max(lam, 0.0) + _jitter_scale(H))
    rhs = C.T.reshape(-1).to(torch.float64).unsqueeze(1)
    try:
        x = torch.linalg.solve(H, rhs).squeeze(1)
    except RuntimeError:
        x = torch.linalg.lstsq(H, rhs, rcond=1e-12).solution.squeeze(1)
    return x.reshape(m, n).T.to(C.dtype)


def kron_sum_flat_dim(n: int, m: int) -> int:
    """Side length of the dense system `solve_kron_sum` would build for an (n, m) layer."""
    return n * m


def _jitter(X: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    return X + _jitter_scale(X) * eye


def _jitter_scale(X: torch.Tensor) -> float:
    """Symmetric jitter at the matrix's own scale: enough for LAPACK, below any signal."""
    eps = float(torch.finfo(X.dtype).eps)
    return 1024.0 * eps * max(float(X.diagonal().abs().mean()), 1e-30)
