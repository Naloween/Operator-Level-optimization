"""Is the per-input operator problem non-trivial for a ReLU MLP, and does lambda buy gate stability?

Two kill-or-continue checks before building anything on the nonlinear setting.

**Why this setting and not deep linear.** In a balanced deep linear network the induced operator
step is `T_W(G) = sum_j (J J^T)^{(L-j)/L} G (J^T J)^{(j-1)/L}` -- a function of `J` and `L` alone
(Arora, Cohen & Hazan 2018, Thm 1). So the whole trajectory can be integrated in operator space
with no network and no solver, and the weights are bookkeeping. That is why every comparison we
ran there reproduced known results. With ReLU there is no single `J`: there is a family `J(x)`,
one per input, and the weights are the only object shared across them. The reduction fails, and
the solver is the only way to ask the question.

**The problem.** With `J(x) = W_L D_{L-1}(x) ... D_1(x) W_1` and per-sample operator gradient
`G(x) = (f(x) - y) x^T` (rank one), a single weight update must serve `n` incompatible per-input
operator steps:

    min_{dW}   sum_x || sum_l A_l(x) dW_l B_l(x)  +  eta G(x) ||^2   +   lambda sum_l ||dW_l||^2

**Check 1 (is it non-trivial?).** With `lambda = 0` this is a linear least squares. Report the
irreducible relative residual `||M dW* - D*|| / ||D*||`, as a function of batch size, depth and
width. Near 0 means the constraints are effectively compatible and we are back in the reducible
case; near 1 means no trade-off helps. The direction is only interesting in between. Naive
dimension counting says the transition sits at `n ~ L` (unknowns `L d^2`, constraints `n d^2`),
so the informative question is whether the measured residual beats that count -- i.e. whether the
per-input operators share enough structure for one update to serve many inputs.

**Check 2 (does lambda buy gate stability?).** Small `||dW||` should mean small pre-activation
change, hence no gate flips, hence the fixed-gate algebra used to build `A_l, B_l` is actually
valid for the step taken. Measure gate-flip fraction against `lambda`, at matched operator
progress. If it does not hold the bridge to the piecewise-linear analysis collapses.

Writes `runs/theory/nonlinear.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr


# --------------------------------------------------------------------------- the model


def init_net(d_in, d, d_out, L, init, rng):
    """ReLU MLP without biases: f(x) = W_L relu(W_{L-1} ... relu(W_1 x))."""
    dims = [d_in] + [d] * (L - 1) + [d_out]
    if init == "xavier":
        return [rng.standard_normal((dims[l + 1], dims[l])) * np.sqrt(2.0 / dims[l])
                for l in range(L)]
    if init == "orth":
        Ws = []
        for l in range(L):
            m, n = dims[l + 1], dims[l]
            Q = np.linalg.qr(rng.standard_normal((max(m, n), max(m, n))))[0]
            Ws.append(np.sqrt(2.0) * Q[:m, :n])
        return Ws
    raise ValueError(init)


def forward(Ws, X):
    """X is (n, d_in). Returns per-sample gates and the pre-activations."""
    L = len(Ws)
    H = X
    gates = []
    for l in range(L - 1):
        Z = H @ Ws[l].T
        D = (Z > 0).astype(float)
        gates.append(D)
        H = Z * D
    out = H @ Ws[-1].T
    return out, gates


def operators(Ws, gates, n):
    """J(x) and the contexts A_l(x), B_l(x) with J = A_l W_l B_l, for every sample."""
    L = len(Ws)
    d_in = Ws[0].shape[1]
    B = [np.tile(np.eye(d_in), (n, 1, 1))]                     # B_1 = I
    for l in range(L - 1):
        WB = np.einsum("ij,njk->nik", Ws[l], B[-1])            # W_l B_l
        B.append(gates[l][:, :, None] * WB)                    # D_l (W_l B_l)

    d_out = Ws[-1].shape[0]
    A = [np.tile(np.eye(d_out), (n, 1, 1))]                    # A_L = I
    for l in range(L - 1, 0, -1):
        AW = np.einsum("nij,jk->nik", A[-1], Ws[l])            # A_{l+1} W_{l+1}
        A.append(AW * gates[l - 1][:, None, :])                # (A W) D_l
    A = A[::-1]
    J = np.einsum("nij,jk,nkl->nil", A[0], Ws[0], B[0])
    return J, A, B


# --------------------------------------------------------------------------- the linear map


def make_operator(A, B, shapes, n):
    """M: dW -> (sum_l A_l(x) dW_l B_l(x))_x, and its adjoint. Matrix-free."""
    L = len(shapes)
    sizes = [s[0] * s[1] for s in shapes]
    offs = np.cumsum([0] + sizes)
    d_out, d_in = A[0].shape[1], B[0].shape[2]
    m = n * d_out * d_in

    # Batched matmul, not a 3-operand einsum: the latter is not BLAS-backed and its cost blows
    # up as n*L grows, which is exactly the regime under test.
    At = [a.transpose(0, 2, 1).copy() for a in A]
    Bt = [b.transpose(0, 2, 1).copy() for b in B]

    def mv(v):
        out = np.zeros((n, d_out, d_in))
        for l in range(L):
            dW = v[offs[l]:offs[l + 1]].reshape(shapes[l])
            out += (A[l] @ dW) @ B[l]
        return out.ravel()

    def rmv(u):
        H = u.reshape(n, d_out, d_in)
        return np.concatenate([((At[l] @ H) @ Bt[l]).sum(axis=0).ravel() for l in range(L)])

    return LinearOperator((m, int(offs[-1])), matvec=mv, rmatvec=rmv, dtype=float)


# --------------------------------------------------------------------------- checks


def cell(d_in, d, d_out, L, n, init, seed, eta, lams, iters) -> dict:
    rng = np.random.default_rng(1000 * seed + 17 * L + n)
    Ws = init_net(d_in, d, d_out, L, init, rng)
    X = rng.standard_normal((n, d_in))
    Wt = rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)
    Y = X @ Wt.T

    out, gates = forward(Ws, X)
    J, A, B = operators(Ws, gates, n)
    # per-sample operator gradient of 1/2||J(x)x - y||^2 : rank one
    R = out - Y                                   # (n, d_out)
    G = np.einsum("ni,nj->nij", R, X)             # (n, d_out, d_in)
    Dstar = (-eta * G).ravel()
    nrm = float(np.linalg.norm(Dstar))

    shapes = [W.shape for W in Ws]
    M = make_operator(A, B, shapes, n)

    res = {"d_in": d_in, "d": d, "d_out": d_out, "L": L, "n": n, "init": init, "seed": seed,
           "eta": eta, "unknowns": int(sum(s[0] * s[1] for s in shapes)),
           "constraints": int(n * d_out * d_in), "loss": float(0.5 * np.sum(R * R) / n)}

    # ---- check 1: irreducible residual at lambda = 0
    # LSQR stalls well short of the true minimum on these operators, so densify M when it fits
    # (one matvec per column) and take the exact least-squares residual instead.
    k, m = M.shape[1], M.shape[0]
    if m * k <= 4e7:
        Md = np.column_stack([M @ e for e in np.eye(k)])
        sol_x, *_ = np.linalg.lstsq(Md, Dstar, rcond=None)
        res["floor_exact"] = True
    else:
        s0 = lsqr(M, Dstar, atol=1e-12, btol=1e-12, conlim=1e13, iter_lim=iters)
        sol_x = s0[0]
        res["floor_exact"] = False
        res["floor_istop"] = int(s0[1])
    res["floor"] = float(np.linalg.norm(M @ sol_x - Dstar) / max(nrm, 1e-300))
    res["floor_dW"] = float(np.linalg.norm(sol_x))
    # what a generic over-determined system of the same shape would give on a random target
    res["floor_generic"] = float(np.sqrt(max(0.0, 1.0 - min(1.0, k / m))))
    res["structure_gain"] = res["floor_generic"] - res["floor"]

    # ---- check 2: gate stability against lambda
    rows = []
    g0 = np.concatenate([g.ravel() for g in gates])
    for lam in lams:
        s = lsqr(M, Dstar, damp=lam, atol=1e-11, btol=1e-11, conlim=1e12, iter_lim=iters)
        v = s[0]
        dWs = []
        off = 0
        for sh in shapes:
            k = sh[0] * sh[1]
            dWs.append(v[off:off + k].reshape(sh))
            off += k
        New = [W + dw for W, dw in zip(Ws, dWs)]
        _, g_new = forward(New, X)
        g1 = np.concatenate([g.ravel() for g in g_new])
        realised = M @ v
        rows.append({
            "lam": lam,
            "dW_norm": float(np.linalg.norm(v)),
            "residual": float(np.linalg.norm(realised - Dstar) / max(nrm, 1e-300)),
            "cos": float(realised @ Dstar / max(np.linalg.norm(realised) * nrm, 1e-300)),
            "gate_flips": float(np.mean(g0 != g1)),
        })
    res["lam_sweep"] = rows
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--inits", nargs="+", default=["xavier"])
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/theory/nonlinear.json")
    a = ap.parse_args()

    rows, t0 = [], time.time()
    for init in a.inits:
        for L in a.depths:
            for n in a.batches:
                for s in a.seeds:
                    r = cell(a.d, a.d, a.d, L, n, init, s, a.eta, a.lams, a.iters)
                    rows.append(r)
                    if s == a.seeds[0]:
                        sw = r["lam_sweep"]
                        print(f"{init} L={L:<2} n={n:<4} | unk={r['unknowns']:<5} "
                              f"cons={r['constraints']:<6} floor={r['floor']:.4f} "
                              f"(generic {r['floor_generic']:.4f}, gain {r['structure_gain']:+.4f})"
                              f"{'' if r['floor_exact'] else ' ~'} | "
                              + "  ".join(f"lam={q['lam']:g}:flip={q['gate_flips']:.3f}"
                                          for q in sw[::2]), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}  ({len(rows)} cells, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
