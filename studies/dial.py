"""The biased-to-unbiased dial: truncated LSQR on the operator step, on a real dataset.

`studies/achievable.py` established the algebra. GD's weight step is exactly `M^T` applied to the
target, so

    GD's induced operator step  =  M M^T D*                 (one Landweber step)
    best achievable step        =  Pi D* = M (M^T M)^+ M^T D*   (the converged solve)

Truncating LSQR at `k` iterations therefore **interpolates between them**: `k=1` is essentially
gradient descent, `k -> inf` is the unbiased achievable step. That is a single principled dial
(Krylov regularisation) for the biased-vs-unbiased comparison, and it is cheaper than the damped
form because small `k` is exactly what makes large problems tractable.

Model: ReLU MLP without biases, `f(x) = W_L relu(W_{L-1} ... relu(W_1 x))`, so `f(x) = J(x) x`
with `J(x) = W_L D_{L-1}(x) ... D_1(x) W_1`.

Tasks: `mnist1d` (40 -> 10, cross-entropy) and `teacher` (random ReLU teacher, squared loss).
For cross-entropy the per-sample operator gradient is still rank one,
`G(x) = (softmax(f(x)) - onehot(y)) x^T`.

Writes `runs/theory/dial.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import lsqr

from nonlinear import forward, init_net, make_operator, operators


def load_task(name, n_train, n_test, width, L, seed):
    if name == "mnist1d":
        from olo.tasks.mnist1d import MNIST1D
        t = MNIST1D(n_train=n_train, n_val=64, n_test=n_test, seed=seed)
        Xtr = t.Xtr.numpy().astype(float)
        Ytr = t.ytr.numpy().astype(int)
        Xte = t.Xte.numpy().astype(float)
        Yte = t.yte.numpy().astype(int)
        return Xtr, Ytr, Xte, Yte, t.d_in, t.d_out, "ce"
    rng = np.random.default_rng(seed + 991)
    d_in = d_out = width
    teacher = init_net(d_in, width, d_out, L, "xavier", rng)
    r2 = np.random.default_rng(seed)
    Xtr = r2.standard_normal((n_train, d_in))
    Xte = r2.standard_normal((n_test, d_in))
    Ytr, _ = forward(teacher, Xtr)
    Yte, _ = forward(teacher, Xte)
    return Xtr, Ytr, Xte, Yte, d_in, d_out, "mse"


def loss_and_grad(out, Y, kind):
    """Returns (loss, dL/dout, accuracy or nan). dL/dout is per-sample, unnormalised by n."""
    n = out.shape[0]
    if kind == "ce":
        z = out - out.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.maximum(p[np.arange(n), Y], 1e-300)).mean())
        R = p.copy()
        R[np.arange(n), Y] -= 1.0
        acc = float((out.argmax(axis=1) == Y).mean())
        return loss, R / n, acc
    R = out - Y
    return float(0.5 * np.sum(R * R) / n), R / n, float("nan")


def step_op(Ws, X, R, eta, k):
    """Truncated-LSQR realisation of the operator step. k=1 ~ gradient descent."""
    n = X.shape[0]
    out_dummy, gates = forward(Ws, X)
    J, A, B = operators(Ws, gates, n)
    G = np.einsum("ni,nj->nij", R, X)
    shapes = [W.shape for W in Ws]
    M = make_operator(A, B, shapes, n)
    target = (-eta * G).ravel()
    sol = lsqr(M, target, atol=0.0, btol=0.0, conlim=0.0, iter_lim=k)
    v = sol[0]
    out, off = [], 0
    for sh in shapes:
        q = sh[0] * sh[1]
        out.append(v[off:off + q].reshape(sh))
        off += q
    realised = M @ v
    nt = float(np.linalg.norm(target))
    return out, {
        "dW_norm": float(np.linalg.norm(v)),
        "cos_target": float(realised @ target / max(np.linalg.norm(realised) * nt, 1e-300)),
        "rel_progress": float(np.linalg.norm(realised) / max(nt, 1e-300)),
    }


def coord_grad(Ws, X, R):
    n = X.shape[0]
    _, gates = forward(Ws, X)
    J, A, B = operators(Ws, gates, n)
    G = np.einsum("ni,nj->nij", R, X)
    return [np.einsum("nji,njk,nlk->il", A[l], G, B[l]) for l in range(len(Ws))]


def run(arm, hyper, Xtr, Ytr, Xte, Yte, d_in, d_out, kind, width, L, steps, batch, seed, log_every):
    rng = np.random.default_rng(seed)
    Ws = init_net(d_in, width, d_out, L, "xavier", rng)
    m = [np.zeros_like(W) for W in Ws]
    vv = [np.zeros_like(W) for W in Ws]
    trace, t0, last_info = [], time.time(), {}
    n = Xtr.shape[0]
    for step in range(steps + 1):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        Xb, Yb = Xtr[idx], Ytr[idx]
        out, _ = forward(Ws, Xb)
        loss, R, acc = loss_and_grad(out, Yb, kind)

        if step % log_every == 0 or step == steps:
            ote, _ = forward(Ws, Xte)
            tl, _, ta = loss_and_grad(ote, Yte, kind)
            trace.append({"step": step, "train_loss": loss, "test_loss": tl,
                          "train_acc": acc, "test_acc": ta})
        if step == steps:
            break

        if arm == "op":
            eta, k = hyper
            dWs, info = step_op(Ws, Xb, R, eta, k)
            last_info = info
            if step % log_every == 0:
                trace[-1].update(info)
        else:
            g = coord_grad(Ws, Xb, R)
            lr = hyper
            if arm == "gd":
                dWs = [-lr * x for x in g]
            else:
                b1, b2, eps = 0.9, 0.999, 1e-8
                dWs = []
                for i, gi in enumerate(g):
                    m[i] = b1 * m[i] + (1 - b1) * gi
                    vv[i] = b2 * vv[i] + (1 - b2) * gi * gi
                    dWs.append(-lr * (m[i] / (1 - b1 ** (step + 1)))
                               / (np.sqrt(vv[i] / (1 - b2 ** (step + 1))) + eps))
        Ws = [W + dw for W, dw in zip(Ws, dWs)]
        if not all(np.all(np.isfinite(W)) for W in Ws):
            trace[-1]["diverged"] = True
            break
    return {"arm": arm, "hyper": hyper, "L": L, "width": width, "batch": batch, "seed": seed,
            "seconds": time.time() - t0, "trace": trace,
            "final": {**last_info, **trace[-1]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mnist1d")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 15, 50, 150])
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.01, 0.05, 0.2, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="runs/theory/dial.json")
    a = ap.parse_args()

    rows = []
    for s in a.seeds:
        Xtr, Ytr, Xte, Yte, d_in, d_out, kind = load_task(a.task, a.n_train, a.n_test,
                                                          a.width, a.depth, s)
        common = (Xtr, Ytr, Xte, Yte, d_in, d_out, kind, a.width, a.depth,
                  a.steps, a.batch, s, a.log_every)
        for arm, hypers in (("gd", a.lrs), ("adam", a.lrs)):
            for h in hypers:
                r = run(arm, h, *common)
                rows.append(r)
                f = r["final"]
                print(f"[{a.task}] s={s} {arm:<5} lr={h:<6g} | train {f['train_loss']:.4f} "
                      f"test {f['test_loss']:.4f} acc {f['test_acc']:.4f} ({r['seconds']:.0f}s)",
                      flush=True)
        for k in a.ks:
            r = run("op", (a.eta, k), *common)
            rows.append(r)
            f = r["final"]
            print(f"[{a.task}] s={s} op    k={k:<6} | train {f['train_loss']:.4f} "
                  f"test {f['test_loss']:.4f} acc {f['test_acc']:.4f} "
                  f"cos={f.get('cos_target', float('nan')):.3f} "
                  f"prog={f.get('rel_progress', float('nan')):.3f} ({r['seconds']:.0f}s)",
                  flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
