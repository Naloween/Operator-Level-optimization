"""The gain exponent, measured with the CORRECT probe, for linear and nonlinear architectures.

The earlier measurement (`gainlaw.py`) probed with a random target D and took
`u_k^T T(D) v_k / u_k^T D v_k`. That equals the diagonal gain only when `A_l A_l^T` and
`B_l^T B_l` are diagonalised by the singular vectors of P -- true under balancedness, FALSE at a
standard initialisation, where the ratio picks up off-diagonal contamination. The correct probe is
the specific rank-one direction `u_k v_k^T`, which gives the gain exactly:

    c_k = <u_k, T(u_k v_k^T) v_k> = sum_l ||A_l^T u_k||^2 ||B_l v_k||^2 .
"""
from __future__ import annotations
import argparse, json
import numpy as np
import dln as DL, crelu as C, nonlinear as R

MODS = {"dln": DL, "relu": R, "crelu": C}


def gains_for(mod, Ws, X, i, floor=1e-10):
    """c_k and s_k for the operator at input i, using the exact rank-one probe."""
    if mod is DL:
        J, A, B = mod.operators(Ws, [], 1)
    else:
        _, g = mod.forward(Ws, X[i:i + 1]); J, A, B = mod.operators(Ws, g, 1)
    U, s, Vt = np.linalg.svd(J[0])
    L = len(Ws)
    out = []
    for k in range(len(s)):
        if s[k] <= floor * s[0]:
            continue
        c = sum(float(np.sum((A[l][0].T @ U[:, k]) ** 2))
                * float(np.sum((B[l][0] @ Vt[k]) ** 2)) for l in range(L))
        if c > 0:
            out.append((s[k], c))
    return np.array(out)


def slope(m):
    if len(m) < 4:
        return float("nan"), float("nan")
    x, y = np.log(m[:, 0]), np.log(m[:, 1])
    xc = x - x.mean(); v = float(xc @ xc)
    if v < 1e-12:
        return float("nan"), float("nan")
    sl = float(xc @ (y - y.mean()) / v)
    return sl, float(np.exp(y.mean() - sl * x.mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n-inputs", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/theory/exponent_fixed.json")
    a = ap.parse_args()
    from olo.tasks.mnist1d import MNIST1D
    X = MNIST1D(n_train=64, n_val=8, n_test=8, seed=0).Xtr.numpy().astype(float)[:a.n_inputs]
    rows = []
    print(f"{'arch':<6}{'L':>4} | {'exponent':>18} {'2-2/L':>8} {'gap':>8} | {'intercept':>10} {'L':>5}")
    for arch in ("dln", "crelu", "relu"):
        mod = MODS[arch]
        for L in a.depths:
            es, its = [], []
            for sd in a.seeds:
                rng = np.random.default_rng(100 * sd + L)
                d_in = X.shape[1]
                Ws = mod.init_net(d_in, a.width, 10, L, "xavier", rng)
                Xi = X if mod is not DL else X
                for i in range(a.n_inputs if mod is not DL else 1):
                    sl, it = slope(gains_for(mod, Ws, Xi, i))
                    if np.isfinite(sl):
                        es.append(sl); its.append(it)
            if not es:
                continue
            e, sd_ = float(np.mean(es)), float(np.std(es))
            rows.append({"arch": arch, "L": L, "exponent": e, "sd": sd_,
                         "target": 2 - 2 / L, "intercept": float(np.mean(its))})
            print(f"{arch:<6}{L:>4} | {e:>10.4f}±{sd_:<7.4f} {2-2/L:>8.4f} {e-(2-2/L):>+8.4f} | "
                  f"{np.mean(its):>10.2f} {L:>5}", flush=True)
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
