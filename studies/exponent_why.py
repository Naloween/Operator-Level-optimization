"""Why is the measured gain exponent about 1, and is it universal?

`gainlaw.py` found that at standard initialisation the exponent `d log c / d log s` sits near 1
rather than at `2 - 2/L`, identically for deep linear, CReLU and ReLU. Before claiming that as a
finding we need to know whether it is derivable, and whether "about 1" is a constant or an
accident of the sizes we happened to run.

An exact decomposition makes the question sharp. With `a = A_l^T u_k`, `b = B_l v_k`,
Cauchy-Schwarz on `s_k = a^T W_l b` gives `||a|| ||b|| >= s_k / ||W_l||`, so writing

    r_{l,k} := ||a|| ||b|| ||W_l|| / s_k   >= 1          (1 = perfectly aligned)

the mode gain is EXACTLY

    c_k = sum_l ||a||^2 ||b||^2 = s_k^2 * sum_l ( r_{l,k} / ||W_l|| )^2

and therefore, with no approximation,

    exponent = d log c / d log s = 2 + d log( sum_l (r_{l,k}/||W_l||)^2 ) / d log s .

So the exponent is 2 minus however fast the misalignment grows as the singular value shrinks.
An exponent of 1 means `sum_l r_{l,k}^2` scales like `1/s_k`: the leading directions are nearly
aligned and the trailing ones are not. This script measures the two terms separately, and sweeps
width, depth and initialisation scale to see whether the result is universal or drifts.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def chain(d, L, init, scale, rng):
    if init == "xavier":
        return [rng.standard_normal((d, d)) * np.sqrt(2.0 / d) * scale for _ in range(L)]
    if init == "gaussian":
        return [rng.standard_normal((d, d)) / np.sqrt(d) * scale for _ in range(L)]
    if init == "balanced":
        Us = [np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(L + 1)]
        S = np.diag(np.exp(np.linspace(0.8, -0.8, d)) ** (1.0 / L))
        return [Us[l + 1] @ S @ Us[l].T for l in range(L)]
    raise ValueError(init)


def measure(Ws, floor=1e-10):
    L = len(Ws)
    B = [np.eye(Ws[0].shape[1])]
    for l in range(L - 1):
        B.append(Ws[l] @ B[-1])
    A = [np.eye(Ws[-1].shape[0])]
    for l in range(L - 1, 0, -1):
        A.append(A[-1] @ Ws[l])
    A = A[::-1]
    J = A[0] @ Ws[0] @ B[0]
    U, s, Vt = np.linalg.svd(J)
    wn = [np.linalg.norm(W, 2) for W in Ws]
    rows = []
    for k in range(len(s)):
        if s[k] <= floor * s[0]:
            continue
        c, rr = 0.0, []
        for l in range(L):
            na = np.linalg.norm(A[l].T @ U[:, k]); nb = np.linalg.norm(B[l] @ Vt[k])
            c += (na * nb) ** 2
            rr.append(na * nb * wn[l] / max(s[k], 1e-300))
        rows.append((s[k], c, float(np.sum(np.square(rr))), float(np.min(rr))))
    return np.array(rows)


def slope(x, y):
    xc = x - x.mean()
    v = float(xc @ xc)
    return float(xc @ (y - y.mean()) / v) if v > 1e-12 else float("nan")


def cell(d, L, init, scale, seed):
    m = measure(chain(d, L, init, scale, np.random.default_rng(seed)))
    if len(m) < 4:
        return None
    ls, lc = np.log(m[:, 0]), np.log(m[:, 1])
    lR = np.log(m[:, 2])                       # sum_l r^2 : the misalignment term
    return {"d": d, "L": L, "init": init, "scale": scale, "seed": seed, "n": len(m),
            "exponent": slope(ls, lc), "target": 2 - 2 / L,
            "misalign_slope": slope(ls, lR),   # exponent = 2 + this, exactly
            "r_min_median": float(np.median(m[:, 3])),
            "check": slope(ls, lc) - (2 + slope(ls, lR))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--inits", nargs="+", default=["xavier", "balanced"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/theory/exponent_why.json")
    a = ap.parse_args()
    rows = []
    print(f"{'init':<9}{'d':>5}{'L':>4}{'scale':>7} | {'exponent':>16} {'2-2/L':>7} | "
          f"{'2+misalign':>11} {'identity':>9} | {'min r':>7}")
    for init in a.inits:
        for d in a.widths:
            for L in a.depths:
                for sc in a.scales:
                    rs = [cell(d, L, init, sc, s) for s in a.seeds]
                    rs = [r for r in rs if r]
                    if not rs:
                        continue
                    rows += rs
                    e = np.mean([r["exponent"] for r in rs]); sd = np.std([r["exponent"] for r in rs])
                    ms = np.mean([r["misalign_slope"] for r in rs])
                    ck = np.mean([abs(r["check"]) for r in rs])
                    rm = np.mean([r["r_min_median"] for r in rs])
                    print(f"{init:<9}{d:>5}{L:>4}{sc:>7g} | {e:>9.4f}±{sd:<6.4f} {2-2/L:>7.4f} | "
                          f"{2+ms:>11.4f} {ck:>9.1e} | {rm:>7.3f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
