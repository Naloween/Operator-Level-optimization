"""How far is gradient descent from the BEST ACHIEVABLE operator step?

`studies/nonlinear.py` showed the ideal step `-eta G` is largely unreachable in a ReLU network.
That is not a defect to correct, it is a property of the gates: the reachable set is
`range(M)`, `M(dW) = sum_l A_l(x) dW_l B_l(x)`. So the honest reference is not the ideal but the
**achievable ideal** `Pi_{range M}(-eta G)`, and both GD's step and the best step live in
`range(M)`, so the unreachable part cancels.

This matters for what we measured before: `theory/12` compared `dJ` against `G` itself and found
`sin -> 1`. If most of `G` is unreachable then `sin ~ 1` is FORCED and says nothing about GD.

Algebra makes the comparison sharp. GD's weight step is `dW_l = -lr sum_x A_l^T G(x) B_l^T`,
i.e. exactly `M^T` applied to the target. Hence

    GD's induced step  =  M M^T D*            (one Landweber / Richardson step)
    achievable ideal   =  Pi D* = M (M^T M)^+ M^T D*   (the converged solve)

so the gap between them is exactly the ill-conditioning of `M M^T` on its range. We report
`cos(M M^T D*, Pi D*)`, the achievable fraction `||Pi D*|| / ||D*||`, and the spectrum of M.

Writes `runs/theory/achievable.json`.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
from nonlinear import init_net, forward, operators, make_operator


def cell(d, L, n, init, seed, eta):
    rng = np.random.default_rng(1000 * seed + 17 * L + n)
    Ws = init_net(d, d, d, L, init, rng)
    X = rng.standard_normal((n, d))
    Wt = rng.standard_normal((d, d)) / np.sqrt(d)
    Y = X @ Wt.T
    out, gates = forward(Ws, X)
    J, A, B = operators(Ws, gates, n)
    G = np.einsum("ni,nj->nij", out - Y, X)
    Dstar = (-eta * G).ravel()
    shapes = [W.shape for W in Ws]
    M = make_operator(A, B, shapes, n)
    k, m = M.shape[1], M.shape[0]

    Md = np.column_stack([M @ e for e in np.eye(k)])          # dense: k matvecs
    U, s, Vt = np.linalg.svd(Md, full_matrices=False)
    tol = max(m, k) * np.finfo(float).eps * s[0]
    r = int((s > tol).sum())
    Pi = U[:, :r] @ (U[:, :r].T @ Dstar)                      # achievable ideal
    gd = Md @ (Md.T @ Dstar)                                  # M M^T D*  = GD's induced step

    nD, nPi, ngd = (float(np.linalg.norm(x)) for x in (Dstar, Pi, gd))
    return {
        "d": d, "L": L, "n": n, "init": init, "seed": seed,
        "rank": r, "unknowns": k, "constraints": m,
        "achievable_frac": nPi / max(nD, 1e-300),
        "cos_gd_ideal": float(gd @ Dstar / max(ngd * nD, 1e-300)),
        "cos_gd_achievable": float(gd @ Pi / max(ngd * nPi, 1e-300)),
        "cond_range": float(s[0] / s[r - 1]) if r > 0 else float("inf"),
        "sv_decay": [float(x) for x in (s[:r][:: max(1, r // 8)])[:8]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32])
    ap.add_argument("--inits", nargs="+", default=["xavier"])
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/theory/achievable.json")
    a = ap.parse_args()
    rows, t0 = [], time.time()
    for init in a.inits:
        for L in a.depths:
            for n in a.batches:
                for s in a.seeds:
                    r = cell(a.d, L, n, init, s, a.eta)
                    rows.append(r)
                    if s == a.seeds[0]:
                        print(f"{init} L={L:<2} n={n:<3} | rank {r['rank']:>4}/{min(r['unknowns'],r['constraints']):<5}"
                              f" achievable={r['achievable_frac']:.3f}"
                              f"  cos(GD, ideal)={r['cos_gd_ideal']:.3f}"
                              f"  cos(GD, ACHIEVABLE)={r['cos_gd_achievable']:.3f}"
                              f"  cond={r['cond_range']:.2e}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
