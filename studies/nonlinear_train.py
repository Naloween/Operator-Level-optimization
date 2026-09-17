"""Does a lambda-regularised operator step actually train a ReLU MLP?

The legitimacy precondition for everything downstream. `studies/nonlinear.py` established that the
per-input operator gradient step is structurally unreachable in a ReLU network -- 33--90% of its
norm lies outside the reachable set, even for a single input, because the gates make the contexts
rank-deficient. So the step we can take is a compromise:

    dW = argmin  sum_x || sum_l A_l(x) dW_l B_l(x) + eta G(x) ||^2  +  lambda sum_l ||dW_l||^2

solved by LSQR with damping. The question here is narrow and must be answered before any analysis
is worth doing: **does iterating that compromise reduce the loss at all, and does it get to a
comparable place as ordinary gradient descent?** If it does not, then any later claim about *what
the operator step learns differently* is confounded by it simply not learning.

Arms:
    op     the lambda-regularised operator step
    gd     plain gradient descent, dW_l = -lr * sum_x A_l(x)^T G(x) B_l(x)^T
    adam   Adam on the same gradients

Also records, per step, the **per-sample residual** `||dJ(x) + eta G(x)|| / ||eta G(x)||` -- which
inputs the compromise serves and which it sacrifices. That allocation is the candidate object of
study if this check passes.

Writes `runs/theory/nonlinear_train.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import lsqr

from nonlinear import forward, init_net, make_operator, operators


def unpack(v, shapes):
    out, off = [], 0
    for sh in shapes:
        k = sh[0] * sh[1]
        out.append(v[off:off + k].reshape(sh))
        off += k
    return out


def grads(Ws, X, Y):
    """Per-sample operator gradient and the coordinate gradient it induces."""
    out, gates = forward(Ws, X)
    n = X.shape[0]
    J, A, B = operators(Ws, gates, n)
    R = out - Y
    G = np.einsum("ni,nj->nij", R, X) / n            # (n, d_out, d_in), rank one per sample
    gW = [np.einsum("nji,njk,nlk->il", A[l], G, B[l]) for l in range(len(Ws))]
    loss = float(0.5 * np.sum(R * R) / n)
    return loss, G, A, B, gates, gW


def op_step(Ws, A, B, G, eta, lam, iters):
    shapes = [W.shape for W in Ws]
    n = G.shape[0]
    M = make_operator(A, B, shapes, n)
    target = (-eta * G).ravel()
    sol = lsqr(M, target, damp=lam, atol=1e-10, btol=1e-10, conlim=1e12, iter_lim=iters)
    v = sol[0]
    realised = (M @ v).reshape(G.shape)
    want = (-eta * G)
    per_sample = (np.linalg.norm((realised - want).reshape(n, -1), axis=1)
                  / np.maximum(np.linalg.norm(want.reshape(n, -1), axis=1), 1e-300))
    return unpack(v, shapes), float(np.linalg.norm(v)), per_sample


def run(arm, d, L, n, steps, eta, lam, lr, iters, seed, iters_log) -> dict:
    rng = np.random.default_rng(seed)
    teacher = init_net(d, d, d, L, "xavier", np.random.default_rng(seed + 991))
    X = rng.standard_normal((n, d))
    Y, _ = forward(teacher, X)
    Ws = init_net(d, d, d, L, "xavier", rng)

    m = [np.zeros_like(W) for W in Ws]
    v = [np.zeros_like(W) for W in Ws]
    trace, t0 = [], time.time()
    for step in range(steps + 1):
        loss, G, A, B, gates, gW = grads(Ws, X, Y)
        rec = {"step": step, "loss": loss}
        if step % iters_log == 0 or step == steps:
            trace.append(rec)
        if step == steps:
            break

        if arm == "op":
            dWs, nrm, per_sample = op_step(Ws, A, B, G, eta, lam, iters)
            rec["dW_norm"] = nrm
            rec["res_med"] = float(np.median(per_sample))
            rec["res_min"] = float(per_sample.min())
            rec["res_max"] = float(per_sample.max())
            rec["res_spread"] = float(per_sample.max() - per_sample.min())
        elif arm == "gd":
            dWs = [-lr * g for g in gW]
            rec["dW_norm"] = float(np.sqrt(sum(float(np.sum(x * x)) for x in dWs)))
        elif arm == "adam":
            b1, b2, eps = 0.9, 0.999, 1e-8
            dWs = []
            for i, g in enumerate(gW):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1 ** (step + 1))
                vh = v[i] / (1 - b2 ** (step + 1))
                dWs.append(-lr * mh / (np.sqrt(vh) + eps))
            rec["dW_norm"] = float(np.sqrt(sum(float(np.sum(x * x)) for x in dWs)))
        else:
            raise ValueError(arm)

        g_before = np.concatenate([g.ravel() for g in gates])
        Ws = [W + dw for W, dw in zip(Ws, dWs)]
        _, g_after = forward(Ws, X)
        rec["gate_flips"] = float(np.mean(g_before != np.concatenate([g.ravel() for g in g_after])))
        if not all(np.all(np.isfinite(W)) for W in Ws):
            rec["diverged"] = True
            break

    return {"arm": arm, "d": d, "L": L, "n": n, "eta": eta, "lam": lam, "lr": lr,
            "seed": seed, "seconds": time.time() - t0,
            "loss0": trace[0]["loss"], "loss_end": trace[-1]["loss"], "trace": trace}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[4])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eta", type=float, nargs="+", default=[0.5])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 1.0])
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.05])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--out", default="runs/theory/nonlinear_train.json")
    a = ap.parse_args()

    rows = []
    for L in a.depths:
        for s in a.seeds:
            for lr in a.lrs:
                for arm in ("gd", "adam"):
                    r = run(arm, a.d, L, a.n, a.steps, 0.0, 0.0, lr, a.iters, s, a.log_every)
                    rows.append(r)
                    print(f"L={L} s={s} {arm:<5} lr={lr:<7g} | loss {r['loss0']:.4e} -> "
                          f"{r['loss_end']:.4e}  ({r['seconds']:.0f}s)", flush=True)
            for eta in a.eta:
                for lam in a.lams:
                    r = run("op", a.d, L, a.n, a.steps, eta, lam, 0.0, a.iters, s, a.log_every)
                    rows.append(r)
                    last = [t for t in r["trace"] if "res_med" in t]
                    rm = last[-1]["res_med"] if last else float("nan")
                    print(f"L={L} s={s} op    eta={eta:<5g} lam={lam:<5g} | loss "
                          f"{r['loss0']:.4e} -> {r['loss_end']:.4e}  res_med={rm:.3f}  "
                          f"({r['seconds']:.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
