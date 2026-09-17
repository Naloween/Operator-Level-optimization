"""Does the dichotomic solver scale, and does the depth-proportional split stay exact?

Checks the scaling question left open by `studies/certificate.py`, which ran only at d=6, L<=8.
ALS is gone: the dichotomy is the solver.

Two things are measured, and they need different references because the joint augmented-Lagrangian
reference of `certificate.py` does not itself scale (it is already imperfect at L=8):

1. **From a degenerate base point** the optimum of (P) is known in closed form --
   `min sum_l ||W_l||_F^2 = L * sum_i sigma_i(P*)^{2/L}`, the Schatten-2/L value -- so the
   solver can be scored against an exact number at any size, with no reference solver at all.
   This is where the depth-proportional claim gets tested at depth 128.

2. **From a general base point** there is no closed form, so we report the KKT residual

       rho = min_Lambda sum_l ||dW_l - A'_l^T Lambda B'_l^T||^2 / sum_l ||dW_l||^2

   which is a certificate, not a comparison. It is computed matrix-free: the normal equations are
   `T(Lambda) = sum_l A'_l dW_l B'_l` with `T(Lambda) = sum_l A'_l A'_l^T Lambda B'_l^T B'_l` the
   transfer operator, solved by conjugate gradients. Forming the dense map would need a
   `d^2`-column matrix of height `L d^2` -- 1 GB at d=32, L=128 -- which is why the d=6 code
   could not be reused.

Writes `runs/theory/scaling.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def product(Ws):
    J = Ws[0]
    for W in Ws[1:]:
        J = W @ J
    return J


def contexts(Ws):
    L = len(Ws)
    B = [np.eye(Ws[0].shape[1])]
    for l in range(L - 1):
        B.append(Ws[l] @ B[-1])
    A = [np.eye(Ws[-1].shape[0])]
    for l in range(L - 1, 0, -1):
        A.append(A[-1] @ Ws[l])
    return A[::-1], B


# --------------------------------------------------------------------------- certificate


def rho_cg(dWs, Ws_new, iters=300, tol=1e-12):
    """KKT residual, matrix-free. T is PSD self-adjoint for the Frobenius inner product."""
    A, B = contexts(Ws_new)
    L = len(Ws_new)
    AA = [a @ a.T for a in A]
    BB = [b.T @ b for b in B]

    def T(X):
        return sum(AA[l] @ X @ BB[l] for l in range(L))

    rhs = sum(A[l] @ dWs[l] @ B[l] for l in range(L))
    X = np.zeros_like(rhs)
    r = rhs - T(X)
    pdir = r.copy()
    rs = float(np.sum(r * r))
    rs0 = rs
    for _ in range(iters):
        if rs <= tol * tol * max(rs0, 1e-300):
            break
        Tp = T(pdir)
        den = float(np.sum(pdir * Tp))
        if den <= 0:
            break
        al = rs / den
        X += al * pdir
        r -= al * Tp
        rs_new = float(np.sum(r * r))
        pdir = r + (rs_new / rs) * pdir
        rs = rs_new
    num = sum(float(np.sum((dWs[l] - A[l].T @ X @ B[l].T) ** 2)) for l in range(L))
    den = sum(float(np.sum(d * d)) for d in dWs)
    return num / den if den > 0 else 0.0


# --------------------------------------------------------------------------- the solver


def svd_split(Z, t):
    """Z = Y X with Y = U S^t, X = S^{1-t} V^T."""
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    return U @ np.diag(S ** t), np.diag(S ** (1.0 - t)) @ Vt


def split2(Y, X, Z, t, budget, mu0=1e6, tol=1e-12):
    """min ||Y'-Y||^2 + ||X'-X||^2 s.t. Y'X' = Z, started from the closed-form split.

    `t` is the SVD exponent of that start: 1/2 is the symmetric split, a/(a+b) the
    depth-proportional one. When the base point is degenerate the start IS the answer and the
    polish has nothing to do.
    """
    Y0, X0 = svd_split(Z, t)
    if budget == 0:
        return Y0, X0
    ny, nx = Y.shape, X.shape
    Lam = np.zeros_like(Z)
    mu = mu0
    ref = max(float(np.linalg.norm(Z)), 1e-300)

    def fg(v):
        dY = v[:ny[0] * ny[1]].reshape(ny)
        dX = v[ny[0] * ny[1]:].reshape(nx)
        Yp, Xp = Y + dY, X + dX
        C = Yp @ Xp - Z
        S = mu * C - Lam
        f = (float(np.sum(dY * dY)) + float(np.sum(dX * dX))
             - float(np.sum(Lam * C)) + 0.5 * mu * float(np.sum(C * C)))
        return f, np.concatenate([(2 * dY + S @ Xp.T).ravel(), (2 * dX + Yp.T @ S).ravel()])

    best, bestn = (Y0, X0), float(np.sum(Y0 * Y0)) + float(np.sum(X0 * X0))
    for v0 in (np.concatenate([(Y0 - Y).ravel(), (X0 - X).ravel()]),
               np.zeros(ny[0] * ny[1] + nx[0] * nx[1])):
        v = v0
        for _ in range(budget):
            v = minimize(fg, v, jac=True, method="L-BFGS-B",
                         options={"maxiter": 500, "ftol": 1e-18, "gtol": 1e-14}).x
            dY = v[:ny[0] * ny[1]].reshape(ny)
            dX = v[ny[0] * ny[1]:].reshape(nx)
            C = (Y + dY) @ (X + dX) - Z
            if float(np.linalg.norm(C)) < tol * ref:
                break
            Lam = Lam - mu * C
            mu *= 2.0
        dY = v[:ny[0] * ny[1]].reshape(ny)
        dX = v[ny[0] * ny[1]:].reshape(nx)
        if float(np.linalg.norm((Y + dY) @ (X + dX) - Z)) < 1e-8 * ref:
            n = float(np.sum(dY * dY)) + float(np.sum(dX * dX))
            if n < bestn:
                best, bestn = (Y + dY, X + dX), n
    return best


def dichotomy(Ws, Pstar, split="prop", budget=8):
    """Split the chain at its midpoint, solve the two-factor problem, recurse."""
    L = len(Ws)
    New = [None] * L

    def rec(a, b, Z):
        if a == b:
            New[a] = Z
            return
        m = (a + b) // 2
        nlo, nhi = m - a + 1, b - m
        t = 0.5 if split == "sym" else nhi / (nlo + nhi)
        X = product(Ws[a:m + 1])
        Y = product(Ws[m + 1:b + 1])
        degenerate = max(float(np.linalg.norm(Y)), float(np.linalg.norm(X))) == 0.0
        Yp, Xp = split2(Y, X, Z, t, 0 if degenerate else budget)
        rec(m + 1, b, Yp)
        rec(a, m, Xp)

    rec(0, L - 1, Pstar)
    return [n - W for n, W in zip(New, Ws)], New


def schatten_optimum(Pstar, L):
    """L * sum_i sigma_i^{2/L}: the exact min of sum_l ||W_l||_F^2 over all factorisations."""
    s = np.linalg.svd(Pstar, compute_uv=False)
    s = s[s > 0]
    return float(L * np.sum(s ** (2.0 / L)))


# --------------------------------------------------------------------------- experiment


def cell(d, L, init, split, seed, eta, budget) -> dict:
    rng = np.random.default_rng(1000 * seed + L)
    if init == "zero":
        Ws = [np.zeros((d, d)) for _ in range(L)]
    elif init == "tiny":
        eps = 1e-2 ** (1.0 / L)
        Ws = [eps * np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(L)]
    else:
        Ws = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(L)]

    Atgt = rng.standard_normal((d, d)) / np.sqrt(d)
    J = product(Ws)
    Pstar = J - eta * (J - Atgt)
    if not np.all(np.isfinite(Pstar)):
        return {"d": d, "L": L, "init": init, "split": split, "seed": seed, "fail": "nonfinite"}

    t0 = time.time()
    dW, New = dichotomy(Ws, Pstar, split, budget)
    secs = time.time() - t0

    got = product(New)
    ref = max(float(np.linalg.norm(Pstar)), 1e-300)
    out = {
        "d": d, "L": L, "init": init, "split": split, "seed": seed, "seconds": secs,
        "rel_residual": float(np.linalg.norm(got - Pstar) / ref),
        "sq_norm": float(sum(float(np.sum(x * x)) for x in New)),   # sum_l ||W_l||^2 achieved
        "dW_norm": float(np.sqrt(sum(float(np.sum(x * x)) for x in dW))),
        "Pstar_norm": ref,
    }
    if init == "zero":
        opt = schatten_optimum(Pstar, L)
        out["optimum"] = opt
        out["ratio_to_optimum"] = out["sq_norm"] / opt if opt > 0 else float("nan")
    else:
        out["rho"] = rho_cg(dW, New)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 32, 64, 128])
    ap.add_argument("--inits", nargs="+", default=["zero", "tiny", "xavier"])
    ap.add_argument("--splits", nargs="+", default=["sym", "prop"])
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--budget", type=int, default=8, help="AL outer iterations per split")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/scaling.json")
    a = ap.parse_args()

    rows, t0 = [], time.time()
    for init in a.inits:
        for L in a.depths:
            for split in a.splits:
                for s in a.seeds:
                    r = cell(a.d, L, init, split, s, a.eta, a.budget)
                    rows.append(r)
                    extra = (f"ratio={r['ratio_to_optimum']:.6f}" if "ratio_to_optimum" in r
                             else f"rho={r.get('rho', float('nan')):.3e}")
                    print(f"d={a.d} L={L:<4} {init:<7} {split:<5} s={s} | "
                          f"res={r.get('rel_residual', float('nan')):.2e} {extra} "
                          f"sum|W|^2={r.get('sq_norm', float('nan')):.4e} "
                          f"{r.get('seconds', 0):.1f}s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}  ({len(rows)} cells, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
