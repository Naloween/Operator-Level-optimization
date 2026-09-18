"""Measure (D1) reachability and (D2) efficiency along training, for ReLU and CReLU.

Quantities (notation as in the draft, section "Notation"):
    M           the step map, (M dW)_x = sum_l A_l(x) dW_l B_l(x)
    D*          the target, (-eta G(x))_x, G = dloss/dJ(x)
    Pi D*       the achievable ideal, projection of D* on range(M)
    D1  alpha = ||Pi D*|| / ||D*||                  how much of the ideal is reachable at all
    D2  eps   = cos(M M^T D*, Pi D*)                how much of the ACHIEVABLE step GD takes

`M M^T D*` is GD's induced operator change (the coordinate gradient is exactly `M^T D*`).
`Pi D*` is obtained as `M x*` with `x*` the LSQR solution of `min ||M x - D*||`; the achieved
least-squares residual is reported so alpha is known to be a lower bound when LSQR has not
converged. `M` is never formed: at L=8, width 64 it would be 5.4 GB.

The point of running ReLU against CReLU at the same depth and initialisation: the CReLU gate is an
isometry (`D^T D = I`), so the two architectures differ in how the gate conditions the contexts,
not in whether a gate is present. Any systematic difference in (D2) is attributable to that.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

import crelu as C
import nonlinear as R


def make_M(A, B, shapes, n):
    L = len(shapes)
    sizes = [s[0] * s[1] for s in shapes]
    offs = np.cumsum([0] + sizes)
    d_out, d_in = A[0].shape[1], B[0].shape[2]
    At = [a.transpose(0, 2, 1).copy() for a in A]
    Bt = [b.transpose(0, 2, 1).copy() for b in B]

    def mv(v):
        out = np.zeros((n, d_out, d_in))
        for l in range(L):
            out += (A[l] @ v[offs[l]:offs[l + 1]].reshape(shapes[l])) @ B[l]
        return out.ravel()

    def rmv(u):
        H = u.reshape(n, d_out, d_in)
        return np.concatenate([((At[l] @ H) @ Bt[l]).sum(axis=0).ravel() for l in range(L)])

    return LinearOperator((n * d_out * d_in, int(offs[-1])), matvec=mv, rmatvec=rmv, dtype=float)


def softmax_grad(out, Y):
    n = out.shape[0]
    z = out - out.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    loss = float(-np.log(np.maximum(p[np.arange(n), Y], 1e-300)).mean())
    Rm = p.copy(); Rm[np.arange(n), Y] -= 1.0
    return loss, Rm / n, float((out.argmax(axis=1) == Y).mean())


def diagnose(mod, Ws, X, Y, eta, kmax):
    out, gates = mod.forward(Ws, X)
    n = X.shape[0]
    J, A, B = mod.operators(Ws, gates, n)
    loss, Rm, acc = softmax_grad(out, Y)
    G = np.einsum("ni,nj->nij", Rm, X)
    D = (-eta * G).ravel()
    M = make_M(A, B, [W.shape for W in Ws], n)

    gd = M @ (M.rmatvec(D))                       # M M^T D*  : GD's induced operator change
    sol = lsqr(M, D, atol=1e-12, btol=1e-12, conlim=1e13, iter_lim=kmax)
    Pi = M @ sol[0]                               # Pi D*    : achievable ideal (LSQR estimate)
    nD, nPi, ngd = (float(np.linalg.norm(z)) for z in (D, Pi, gd))

    sv = np.linalg.svd(J, compute_uv=False)       # (n, r)
    pr = (sv.sum(axis=1) ** 2 / np.maximum((sv ** 2).sum(axis=1), 1e-300))
    return {
        "loss": loss, "acc": acc,
        "alpha": nPi / max(nD, 1e-300),
        "eps": float(gd @ Pi / max(ngd * nPi, 1e-300)),
        "cos_gd_ideal": float(gd @ D / max(ngd * nD, 1e-300)),
        "lsqr_res": float(np.linalg.norm(M @ sol[0] - D) / max(nD, 1e-300)),
        "lsqr_itn": int(sol[2]), "lsqr_istop": int(sol[1]),
        "pr_mean": float(pr.mean()),
        "sv_max": float(sv.max()), "sv_min": float(sv.min()),
        "cond_J": float(sv.max() / max(sv.min(), 1e-300)),
    }


def run(arch, init, Xtr, Ytr, L, width, steps, batch, lr, eta, kmax, every, seed):
    mod = C if arch == "crelu" else R
    rng = np.random.default_rng(seed)
    d_in, d_out = Xtr.shape[1], int(Ytr.max()) + 1
    if arch == "crelu":
        Ws = mod.init_net(d_in, width, d_out, L, init, rng)
    else:
        Ws = mod.init_net(d_in, width, d_out, L, "xavier" if init == "xavier" else "orth", rng)
    m = [np.zeros_like(W) for W in Ws]; v = [np.zeros_like(W) for W in Ws]
    trace, t0 = [], time.time()
    n = Xtr.shape[0]
    for step in range(steps + 1):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        Xb, Yb = Xtr[idx], Ytr[idx]
        if step % every == 0 or step == steps:
            d = diagnose(mod, Ws, Xb, Yb, eta, kmax)
            d["step"] = step
            trace.append(d)
            print(f"  {arch}/{init} L={L} step {step:>4} | loss {d['loss']:.4f} acc {d['acc']:.3f}"
                  f" | alpha {d['alpha']:.4f} eps {d['eps']:.4f} (naive {d['cos_gd_ideal']:.4f})"
                  f" | PR {d['pr_mean']:.2f} cond(J) {d['cond_J']:.2e}"
                  f" | lsqr res {d['lsqr_res']:.2e} itn {d['lsqr_itn']}", flush=True)
        if step == steps:
            break
        out, gates = mod.forward(Ws, Xb)
        _, Rm, _ = softmax_grad(out, Yb)
        J, A, B = mod.operators(Ws, gates, Xb.shape[0])
        G = np.einsum("ni,nj->nij", Rm, Xb)
        g = [np.einsum("nji,njk,nlk->il", A[l], G, B[l]) for l in range(L)]
        b1, b2, eps_ = 0.9, 0.999, 1e-8
        for i, gi in enumerate(g):
            m[i] = b1 * m[i] + (1 - b1) * gi
            v[i] = b2 * v[i] + (1 - b2) * gi * gi
            Ws[i] = Ws[i] - lr * (m[i] / (1 - b1 ** (step + 1))) / (np.sqrt(v[i] / (1 - b2 ** (step + 1))) + eps_)
    return {"arch": arch, "init": init, "L": L, "width": width, "lr": lr, "seed": seed,
            "seconds": time.time() - t0, "trace": trace}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--kmax", type=int, default=800)
    ap.add_argument("--archs", nargs="+", default=["relu", "crelu"])
    ap.add_argument("--inits", nargs="+", default=["xavier"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/diagnose.json")
    a = ap.parse_args()

    from olo.tasks.mnist1d import MNIST1D
    rows = []
    for s in a.seeds:
        t = MNIST1D(n_train=a.n_train, n_val=64, n_test=256, seed=s)
        Xtr = t.Xtr.numpy().astype(float); Ytr = t.ytr.numpy().astype(int)
        for arch in a.archs:
            for init in a.inits:
                rows.append(run(arch, init, Xtr, Ytr, a.depth, a.width, a.steps,
                                a.batch, a.lr, a.eta, a.kmax, a.every, s))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
