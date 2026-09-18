"""Where does each method stop training, as depth grows?

Arms: Adam and plain gradient descent (both with a learning-rate grid) against S2, the truncated
Krylov operator step (grid over the target scale `eta` and the iteration budget `k`). ALS is not
run -- it is no longer the method, and its depth behaviour is already known from the previous
submission; it appears in the appendix only.

Why S2 is expected to go deeper than a block method: ALS inverts the per-layer Gram
`A_l^T A_l  x  B_l B_l^T`, which collapses with depth, whereas S2 only applies `M` and `M^T` and
uses truncation at `k` as the regulariser. On the same normal system, block Gauss-Seidel converges
like `cond(M)^2` and Krylov like `cond(M)`. Whether that is enough in practice is the question
here; nothing below is assumed.

Reports, per (architecture, depth, arm): best final train loss and test accuracy over the
hyper-parameter grid, and whether the run made any progress at all.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import lsqr

import crelu as C
import nonlinear as R
from diagnose import make_M, softmax_grad


def step_op(mod, Ws, X, Rm, eta, k):
    n = X.shape[0]
    _, gates = mod.forward(Ws, X)
    J, A, B = mod.operators(Ws, gates, n)
    G = np.einsum("ni,nj->nij", Rm, X)
    shapes = [W.shape for W in Ws]
    M = make_M(A, B, shapes, n)
    sol = lsqr(M, (-eta * G).ravel(), atol=0.0, btol=0.0, conlim=0.0, iter_lim=k)
    out, off = [], 0
    for sh in shapes:
        q = sh[0] * sh[1]
        out.append(sol[0][off:off + q].reshape(sh)); off += q
    return out


def coord_grad(mod, Ws, X, Rm):
    _, gates = mod.forward(Ws, X)
    J, A, B = mod.operators(Ws, gates, X.shape[0])
    G = np.einsum("ni,nj->nij", Rm, X)
    return [np.einsum("nji,njk,nlk->il", A[l], G, B[l]) for l in range(len(Ws))]


def train(arch, arm, hyper, Xtr, Ytr, Xte, Yte, L, width, steps, batch, seed):
    mod = C if arch == "crelu" else R
    rng = np.random.default_rng(seed)
    d_in, d_out = Xtr.shape[1], int(Ytr.max()) + 1
    Ws = mod.init_net(d_in, width, d_out, L, "xavier", rng)
    m = [np.zeros_like(W) for W in Ws]; v = [np.zeros_like(W) for W in Ws]
    best, t0 = {"train": np.inf, "acc": 0.0}, time.time()
    l0 = None
    for step in range(steps + 1):
        idx = rng.choice(Xtr.shape[0], size=min(batch, Xtr.shape[0]), replace=False)
        Xb, Yb = Xtr[idx], Ytr[idx]
        out, _ = mod.forward(Ws, Xb)
        loss, Rm, _ = softmax_grad(out, Yb)
        if l0 is None:
            l0 = loss
        if step % 20 == 0 or step == steps:
            ote, _ = mod.forward(Ws, Xte)
            tl, _, ta = softmax_grad(ote, Yte)
            if loss < best["train"]:
                best = {"train": loss, "acc": ta, "test": tl, "at": step}
        if step == steps or not np.isfinite(loss):
            break
        if arm == "op":
            eta, k = hyper
            dWs = step_op(mod, Ws, Xb, Rm, eta, k)
        else:
            g = coord_grad(mod, Ws, Xb, Rm)
            if arm == "gd":
                dWs = [-hyper * x for x in g]
            else:
                b1, b2, e = 0.9, 0.999, 1e-8
                dWs = []
                for i, gi in enumerate(g):
                    m[i] = b1 * m[i] + (1 - b1) * gi
                    v[i] = b2 * v[i] + (1 - b2) * gi * gi
                    dWs.append(-hyper * (m[i] / (1 - b1 ** (step + 1)))
                               / (np.sqrt(v[i] / (1 - b2 ** (step + 1))) + e))
        Ws = [W + dw for W, dw in zip(Ws, dWs)]
        if not all(np.all(np.isfinite(W)) for W in Ws):
            break
    return {"arch": arch, "arm": arm, "hyper": hyper, "L": L, "seed": seed,
            "loss0": l0, **best, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["relu", "crelu"])
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.001, 0.005, 0.02, 0.1])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.5, 2.0])
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 60])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/depth.json")
    a = ap.parse_args()

    from olo.tasks.mnist1d import MNIST1D
    rows = []
    for s in a.seeds:
        t = MNIST1D(n_train=a.n_train, n_val=64, n_test=256, seed=s)
        Xtr, Ytr = t.Xtr.numpy().astype(float), t.ytr.numpy().astype(int)
        Xte, Yte = t.Xte.numpy().astype(float), t.yte.numpy().astype(int)
        for arch in a.archs:
            for L in a.depths:
                jobs = [("gd", lr) for lr in a.lrs] + [("adam", lr) for lr in a.lrs] \
                       + [("op", (e, k)) for e in a.etas for k in a.ks]
                for arm, h in jobs:
                    r = train(arch, arm, h, Xtr, Ytr, Xte, Yte, L, a.width, a.steps, a.batch, s)
                    rows.append(r)
                    print(f"{arch:<5} L={L:<3} {arm:<5} {str(h):<12} | loss {r['loss0']:.3f} -> "
                          f"{r['train']:.4f}  acc {r['acc']:.3f}  ({r['seconds']:.0f}s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
