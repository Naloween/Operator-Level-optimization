"""Fixed input versus fixed gate pattern: two operators, two exponents.

For a ReLU or CReLU network there are two distinct objects worth measuring.

  P(x)   the operator the network actually applies to input x, with the gates that x induces.
         This is what determines learning for that sample, but it is discontinuous in W: the
         gates flip when x crosses a region boundary.

  J_eps  the operator of a FIXED gate pattern eps, realised by that input or not. This is a
         polynomial in the weights, smooth everywhere, and by gate absorption
         (M_l = D_l W_l D_{l-1}) it is exactly a deep linear chain restricted to a coordinate
         subspace.

That last point makes a sharp prediction: at a fixed pattern the gain exponent should behave like
a DEEP LINEAR network's, because it is one. If the realised-input exponent then departs from the
fixed-pattern exponent, the departure is attributable to the data SELECTING patterns, not to the
presence of gates.

Reports the exponent for: realised inputs, patterns realised by other inputs, and uniformly random
patterns.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import crelu as C, nonlinear as R


def slope(x, y):
    xc = x - x.mean(); v = float(xc @ xc)
    return float(xc @ (y - y.mean()) / v) if v > 1e-12 else float("nan")


def ops_from_gates(Ws, gates, arch):
    """Contexts for an ARBITRARY gate pattern (a list of per-layer gate vectors), batch of 1."""
    L = len(Ws)
    d_in, d_out = Ws[0].shape[1], Ws[-1].shape[0]
    B = [np.eye(d_in)[None]]
    for l in range(L - 1):
        WB = Ws[l] @ B[-1][0]
        if arch == "crelu":
            z = gates[l]
            pos = (z > 0).astype(float)[:, None]; neg = (z < 0).astype(float)[:, None]
            B.append(np.concatenate([pos * WB, -neg * WB], 0)[None])
        else:
            B.append((gates[l].astype(float)[:, None] * WB)[None])
    A = [np.eye(d_out)[None]]
    for l in range(L - 1, 0, -1):
        AW = A[-1][0] @ Ws[l]
        if arch == "crelu":
            z = gates[l - 1]; w = AW.shape[1] // 2
            A.append((AW[:, :w] * (z > 0).astype(float) - AW[:, w:] * (z < 0).astype(float))[None])
        else:
            A.append((AW * gates[l - 1].astype(float))[None])
    A = A[::-1]
    return (A[0][0] @ Ws[0] @ B[0][0])[None], A, B


def exponent(Ws, gates, arch, floor=1e-10):
    J, A, B = ops_from_gates(Ws, gates, arch)
    if not np.all(np.isfinite(J)):
        return float("nan")
    U, s, Vt = np.linalg.svd(J[0]); L = len(Ws)
    sk, ck = [], []
    for k in range(len(s)):
        if s[k] <= floor * s[0]:
            continue
        c = sum(float(np.sum((A[l][0].T @ U[:, k]) ** 2))
                * float(np.sum((B[l][0] @ Vt[k]) ** 2)) for l in range(L))
        if c > 0:
            sk.append(s[k]); ck.append(c)
    return slope(np.log(sk), np.log(ck)) if len(sk) >= 4 else float("nan")


def realised_gates(Ws, x, arch):
    if arch == "crelu":
        _, zs = C.forward(Ws, x[None]); return [z[0] for z in zs]
    _, gs = R.forward(Ws, x[None]); return [g[0] for g in gs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["crelu", "relu"])
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="runs/theory/patterns_exp.json")
    a = ap.parse_args()
    from olo.tasks.mnist1d import MNIST1D
    X = MNIST1D(n_train=64, n_val=8, n_test=8, seed=0).Xtr.numpy().astype(float)
    rows = []
    print(f"{'arch':<6}{'L':>4} | {'realised input':>18} {'random pattern':>18} {'2-2/L':>8}")
    for arch in a.archs:
        mod = C if arch == "crelu" else R
        for L in a.depths:
            re_, ra_ = [], []
            for sd in a.seeds:
                rng = np.random.default_rng(100 * sd + L)
                Ws = mod.init_net(X.shape[1], a.width, 10, L, "xavier", rng)
                for i in range(a.n):
                    e = exponent(Ws, realised_gates(Ws, X[i], arch), arch)
                    if np.isfinite(e): re_.append(e)
                    # A ReLU gate is a 0/1 MASK; a continuous vector would be a random diagonal
                    # scaling, which never zeroes a coordinate and is a different object entirely.
                    # A CReLU gate reads only the sign, so a sign vector is the right sample there.
                    g = ([(rng.random(a.width) < 0.5).astype(float) for _ in range(L - 1)]
                         if arch == "relu" else
                         [rng.standard_normal(a.width) for _ in range(L - 1)])
                    e = exponent(Ws, g, arch)
                    if np.isfinite(e): ra_.append(e)
            rows.append({"arch": arch, "L": L, "realised": float(np.mean(re_)),
                         "realised_sd": float(np.std(re_)), "random": float(np.mean(ra_)),
                         "random_sd": float(np.std(ra_)), "target": 2 - 2 / L})
            print(f"{arch:<6}{L:>4} | {np.mean(re_):>10.4f}±{np.std(re_):<6.3f} "
                  f"{np.mean(ra_):>10.4f}±{np.std(ra_):<6.3f} {2-2/L:>8.4f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
