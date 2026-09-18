"""Pilot for the endpoint experiment: can the arms interpolate, at what cost, and do they differ?

Design questions this answers before a grid is sized (see the discussion in the session log):

  1. Can the operator arm reach a low training loss at all? Its step realises only part of the
     target, so it may never interpolate -- and if it cannot, matching arms ON TRAINING LOSS is
     impossible and the whole endpoint design fails.
  2. How many steps, and what does a step cost as a function of the Krylov budget k?
  3. Is any endpoint difference visible at all?

Method. Implicit bias is only defined when the data does not pin the solution, so the network must
interpolate a SMALL training set and we ask which interpolant each arm picks. Every arm trains
until the training loss (on the whole training set, not a minibatch) crosses `--target`, then
stops -- matching on loss rather than on steps, which is what made the earlier comparison
uninterpretable. `k` is used as a continuum (k=1 is gradient descent), not as a binary.

Endpoint measurements: test loss and accuracy, effective rank of J(x), and the gain exponent.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import lsqr

import crelu as C, nonlinear as R
from diagnose import make_M, softmax_grad
from gainlaw import realised, fit, pairs_from

MODS = {"relu": R, "crelu": C}


def endpoint_stats(mod, Ws, Xte, Yte, Xpr, rng, k):
    out, g = mod.forward(Ws, Xte)
    tl, _, ta = softmax_grad(out, Yte)
    _, gp = mod.forward(Ws, Xpr)
    J, A, B = mod.operators(Ws, gp, Xpr.shape[0])
    sv = np.linalg.svd(J, compute_uv=False)
    pr = (sv.sum(axis=1) ** 2 / np.maximum((sv ** 2).sum(axis=1), 1e-300))
    ex = []
    for i in range(Xpr.shape[0]):
        Xi = Xpr[i:i + 1]
        _, gi = mod.forward(Ws, Xi)
        Ji, Ai, Bi = mod.operators(Ws, gi, 1)
        D = rng.standard_normal((1, Ji.shape[1], Ji.shape[2]))
        sl, _, _ = fit(pairs_from(Ji, realised(Ai, Bi, D), D, 0))
        if np.isfinite(sl):
            ex.append(sl)
    return {"test_loss": tl, "test_acc": ta, "pr_mean": float(pr.mean()),
            "exponent": float(np.mean(ex)) if ex else float("nan")}


def run(arch, arm, hyper, data, L, width, batch, target, max_steps, seed, every):
    Xtr, Ytr, Xte, Yte = data
    mod = MODS[arch]
    rng = np.random.default_rng(seed)
    Ws = mod.init_net(Xtr.shape[1], width, int(Ytr.max()) + 1, L, "xavier", rng)
    m = [np.zeros_like(W) for W in Ws]; v = [np.zeros_like(W) for W in Ws]
    t0, trace, stop, reached = time.time(), [], "max_steps", None
    step_times = []
    for step in range(max_steps + 1):
        if step % every == 0 or step == max_steps:
            o, _ = mod.forward(Ws, Xtr)
            full, _, tra = softmax_grad(o, Ytr)      # training loss on the WHOLE training set
            trace.append({"step": step, "train": full, "train_acc": tra})
            if full <= target:
                stop, reached = "target", step; break
            if not np.isfinite(full) or full > 50:
                stop = "exploded"; break
        if step == max_steps:
            break
        idx = rng.choice(Xtr.shape[0], size=min(batch, Xtr.shape[0]), replace=False)
        Xb, Yb = Xtr[idx], Ytr[idx]
        o, _ = mod.forward(Ws, Xb)
        _, Rm, _ = softmax_grad(o, Yb)
        ts = time.time()
        _, gates = mod.forward(Ws, Xb)
        J, A, B = mod.operators(Ws, gates, Xb.shape[0])
        G = np.einsum("ni,nj->nij", Rm, Xb)
        if arm == "op":
            eta, k = hyper
            M = make_M(A, B, [W.shape for W in Ws], Xb.shape[0])
            sol = lsqr(M, (-eta * G).ravel(), atol=0.0, btol=0.0, conlim=0.0, iter_lim=int(k))
            dWs, off = [], 0
            for sh in [W.shape for W in Ws]:
                q = sh[0] * sh[1]; dWs.append(sol[0][off:off + q].reshape(sh)); off += q
        else:
            gg = [np.einsum("nji,njk,nlk->il", A[l], G, B[l]) for l in range(L)]
            if arm == "gd":
                dWs = [-hyper * x for x in gg]
            else:
                b1, b2, e = 0.9, 0.999, 1e-8
                dWs = []
                for i, gi in enumerate(gg):
                    m[i] = b1 * m[i] + (1 - b1) * gi
                    v[i] = b2 * v[i] + (1 - b2) * gi * gi
                    dWs.append(-hyper * (m[i] / (1 - b1 ** (step + 1)))
                               / (np.sqrt(v[i] / (1 - b2 ** (step + 1))) + e))
        step_times.append(time.time() - ts)
        Ws = [W + dw for W, dw in zip(Ws, dWs)]
        if not all(np.all(np.isfinite(W)) for W in Ws):
            stop = "exploded"; break
    st = endpoint_stats(mod, Ws, Xte, Yte, Xtr[:8], np.random.default_rng(3), hyper)
    return {"arch": arch, "arm": arm, "hyper": hyper, "L": L, "width": width,
            "stop": stop, "reached_at": reached, "steps": trace[-1]["step"],
            "train_final": trace[-1]["train"], "sec_per_step": float(np.mean(step_times)) if step_times else 0,
            "seconds": time.time() - t0, **st, "trace": trace}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="crelu")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--target", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--eta", type=float, default=8.0)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 10, 30])
    ap.add_argument("--adam-lrs", type=float, nargs="+", default=[1e-3, 5e-3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/theory/pilot.json")
    a = ap.parse_args()
    from olo.tasks.mnist1d import MNIST1D
    t = MNIST1D(n_train=a.n_train, n_val=32, n_test=512, seed=a.seed)
    data = (t.Xtr.numpy().astype(float), t.ytr.numpy().astype(int),
            t.Xte.numpy().astype(float), t.yte.numpy().astype(int))
    rows = []
    jobs = [("adam", lr) for lr in a.adam_lrs] + [("op", (a.eta, k)) for k in a.ks]
    for arm, h in jobs:
        r = run(a.arch, arm, h, data, a.depth, a.width, a.batch, a.target,
                a.max_steps, a.seed, a.every)
        rows.append(r)
        print(f"{arm:<5}{str(h):<12} | train {r['train_final']:.2e} {r['stop']:<9}"
              f"@{r['steps']:<5} | test {r['test_loss']:.3f}/{r['test_acc']:.3f}"
              f" | PR {r['pr_mean']:.2f} exp {r['exponent']:+.3f}"
              f" | {r['sec_per_step']*1e3:.0f}ms/step {r['seconds']:.0f}s", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
