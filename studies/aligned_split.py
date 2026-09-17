"""A closed-form cost-to-go for (P) at a GENERAL base point, under alignment.

This is direction (a) of `theory/13-paper-plan.md` §6: make the exactness theorem unconditional.

Where we are. The recursion solves (P) exactly from a DEGENERATE base point, because there the
cost-to-go is the Schatten quasi-norm `V_a(Y) = a sum_i sigma_i(Y)^{2/a}` and the split has the
closed form `Y = U S^{a/L}, X = S^{b/L} V^T`. At a general base point no closed form is known, the
split falls back on an augmented Lagrangian (5-40 s per step at d=32), and the greedy surrogate is
measurably wrong: rho = 0.052 at L=8 and 0.231 at L=32, saturated in the sub-solve budget.

The idea. The two-factor split

    min ||Y' - Y||^2 + ||X' - X||^2   s.t.  Y' X' = Z

has no closed form for general Y, X, Z -- but it DECOUPLES if the three matrices share a frame.
Write Z = U Sz V^T. If Y = U Sy Q^T and X = Q Sx V^T for a common inner basis Q, the problem
separates into `d` independent SCALAR problems

    min (y - y0)^2 + (x - x0)^2   s.t.  y x = z

whose stationarity `2(y-y0) = lam x`, `2(x-x0) = lam y` reduces to the quartic

    y^4 - y0 y^3 + x0 z y - z^2 = 0                                    (*)

solved exactly by radicals. So under alignment the split is `d` quartics instead of an L-BFGS over
`2d^2` variables, and the cost-to-go is available per mode.

Why alignment is a reasonable hypothesis and not a wish: it is the conclusion of Theorem 6.1 of
the ICML paper (spectral separation forces singular-vector alignment), and the regime where the
operator separates is exactly the deep regime this project cares about. The hypothesis is
therefore *checkable*, and this script checks it rather than assuming it: every call reports the
frame misalignment alongside the answer, and the aligned solution is scored against the
augmented-Lagrangian reference.

Writes `runs/theory/aligned_split.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scaling import contexts, product, split2  # the AL reference


def scalar_split(y0, x0, z):
    """Exact solution of min (y-y0)^2 + (x-x0)^2 s.t. y x = z, via the quartic (*)."""
    if z == 0.0:
        # y x = 0: take whichever factor is cheaper to zero out
        return (0.0, x0) if y0 * y0 <= x0 * x0 else (y0, 0.0)
    roots = np.roots([1.0, -y0, 0.0, x0 * z, -z * z])
    best, bf = None, np.inf
    for r in roots:
        if abs(r.imag) > 1e-9 * max(1.0, abs(r.real)) or r.real == 0.0:
            continue
        y = float(r.real)
        x = z / y
        f = (y - y0) ** 2 + (x - x0) ** 2
        if f < bf:
            best, bf = (y, x), f
    if best is None:                       # fall back to the balanced split
        s = np.sqrt(abs(z))
        return (np.sign(z) * s, s)
    return best


def frame(Y, X, Z):
    """Common frame (U, Q, V) for the three matrices, plus the misalignment it costs.

    U, V come from Z (the constraint must be met exactly in that basis). Q is the inner basis;
    we take it from the product Y^T Y + X X^T, the choice that treats both factors symmetrically.
    """
    U, Sz, Vt = np.linalg.svd(Z, full_matrices=False)
    M = Y.T @ Y + X @ X.T
    w, Q = np.linalg.eigh(M)
    Q = Q[:, ::-1]                          # descending
    y0 = np.diag(U.T @ Y @ Q)
    x0 = np.diag(Q.T @ X @ Vt.T)
    # what the diagonal approximation throws away
    Yd, Xd = U.T @ Y @ Q, Q.T @ X @ Vt.T
    off = (np.linalg.norm(Yd - np.diag(np.diag(Yd))) ** 2
           + np.linalg.norm(Xd - np.diag(np.diag(Xd))) ** 2)
    tot = float(np.linalg.norm(Y) ** 2 + np.linalg.norm(X) ** 2)
    return U, Q, Vt, Sz, y0, x0, (off / tot if tot > 0 else 0.0)


def split2_aligned(Y, X, Z):
    """The split, solved per mode under the alignment hypothesis. Returns (Y', X', misalignment)."""
    U, Q, Vt, Sz, y0, x0, mis = frame(Y, X, Z)
    n = len(Sz)
    ys, xs = np.empty(n), np.empty(n)
    for i in range(n):
        ys[i], xs[i] = scalar_split(float(y0[i]), float(x0[i]), float(Sz[i]))
    return U @ np.diag(ys) @ Q.T, Q @ np.diag(xs) @ Vt, mis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[6, 16, 32])
    ap.add_argument("--inits", nargs="+", default=["aligned", "tiny", "xavier"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--out", default="runs/theory/aligned_split.json")
    a = ap.parse_args()

    rows = []
    for d in a.dims:
        for init in a.inits:
            for s in a.seeds:
                rng = np.random.default_rng(100 * s + d)
                if init == "aligned":
                    # Y and X built to share an inner basis: the hypothesis holds by construction
                    U = np.linalg.qr(rng.standard_normal((d, d)))[0]
                    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
                    V = np.linalg.qr(rng.standard_normal((d, d)))[0]
                    Y = U @ np.diag(np.sort(rng.random(d))[::-1] + 0.2) @ Q.T
                    X = Q @ np.diag(np.sort(rng.random(d))[::-1] + 0.2) @ V.T
                elif init == "tiny":
                    Y = 0.1 * np.linalg.qr(rng.standard_normal((d, d)))[0]
                    X = 0.1 * np.linalg.qr(rng.standard_normal((d, d)))[0]
                else:
                    Y = rng.standard_normal((d, d)) / np.sqrt(d)
                    X = rng.standard_normal((d, d)) / np.sqrt(d)

                Z = Y @ X
                Z = Z - 0.05 * (Z - rng.standard_normal((d, d)) / np.sqrt(d))   # an operator step

                t0 = time.time()
                Ya, Xa, mis = split2_aligned(Y, X, Z)
                t_al = time.time() - t0
                t0 = time.time()
                Yr, Xr = split2(Y, X, Z, 0.5, a.budget)
                t_ref = time.time() - t0

                def score(Yp, Xp):
                    return (float(np.sum((Yp - Y) ** 2) + np.sum((Xp - X) ** 2)),
                            float(np.linalg.norm(Yp @ Xp - Z) / max(np.linalg.norm(Z), 1e-300)))

                na, ra = score(Ya, Xa)
                nr, rr = score(Yr, Xr)
                row = {"d": d, "init": init, "seed": s, "misalignment": mis,
                       "aligned_cost": na, "aligned_res": ra, "aligned_secs": t_al,
                       "ref_cost": nr, "ref_res": rr, "ref_secs": t_ref,
                       "cost_ratio": na / nr if nr > 0 else float("nan"),
                       "speedup": t_ref / max(t_al, 1e-9)}
                rows.append(row)
                print(f"d={d:<3} {init:<8} s={s} | misalign={mis:.3e}  "
                      f"cost {na:.5f} vs ref {nr:.5f} (x{row['cost_ratio']:.4f})  "
                      f"res {ra:.1e} vs {rr:.1e}  "
                      f"{t_al*1e3:.1f}ms vs {t_ref*1e3:.0f}ms (x{row['speedup']:.0f})", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
