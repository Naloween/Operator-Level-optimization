"""Adam versus the reachable ideal, on standard MNIST-1D, compared at matched training loss.

Binary by design: gradient descent as practised (Adam, at the reference recipe for this dataset)
against a step that realises the best reachable operator change. The intermediate values of the
Krylov budget k were useful for calibrating the dial, but the comparison of interest is the one
between the two ends, so k is set large enough to BE the ideal rather than a point on a path.

Standard setting: MNIST-1D ships 5000 samples with an 80/20 split, so 4000 train and 1000 test,
and the reference recipe is Adam with lr 1e-2 and batch 100. We sweep lr around it so the baseline
is tuned rather than merely standard.

Every arm records the full training loss, test loss and accuracy, effective rank and gain exponent
along its trajectory, so arms are compared AT MATCHED TRAINING LOSS afterwards rather than at a
matched step count -- the confound that made earlier comparisons unreadable.
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


def probe_stats(mod, Ws, Xpr, rng):
    _, g = mod.forward(Ws, Xpr)
    J, A, B = mod.operators(Ws, g, Xpr.shape[0])
    sv = np.linalg.svd(J, compute_uv=False)
    pr = (sv.sum(axis=1) ** 2 / np.maximum((sv ** 2).sum(axis=1), 1e-300))
    ex = []
    for i in range(min(6, Xpr.shape[0])):
        Xi = Xpr[i:i + 1]
        _, gi = mod.forward(Ws, Xi)
        Ji, Ai, Bi = mod.operators(Ws, gi, 1)
        D = rng.standard_normal((1, Ji.shape[1], Ji.shape[2]))
        sl, _, _ = fit(pairs_from(Ji, realised(Ai, Bi, D), D, 0))
        if np.isfinite(sl):
            ex.append(sl)
    return float(pr.mean()), (float(np.mean(ex)) if ex else float("nan"))


def run(arch, arm, hyper, data, L, width, batch, steps, every, seed):
    Xtr, Ytr, Xte, Yte = data
    mod = MODS[arch]
    rng = np.random.default_rng(seed)
    Ws = mod.init_net(Xtr.shape[1], width, int(Ytr.max()) + 1, L, "xavier", rng)
    m = [np.zeros_like(W) for W in Ws]; v = [np.zeros_like(W) for W in Ws]
    Xpr = Xtr[:8]
    trace, t0, stop = [], time.time(), "max_steps"
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            o, _ = mod.forward(Ws, Xtr); trl, _, tra = softmax_grad(o, Ytr)
            o, _ = mod.forward(Ws, Xte); tel, _, tea = softmax_grad(o, Yte)
            pr, ex = probe_stats(mod, Ws, Xpr, np.random.default_rng(3))
            trace.append({"step": step, "train": trl, "train_acc": tra,
                          "test": tel, "test_acc": tea, "pr": pr, "exponent": ex})
            if not np.isfinite(trl) or trl > 50:
                stop = "exploded"; break
        if step == steps:
            break
        idx = rng.choice(Xtr.shape[0], size=batch, replace=False)
        Xb, Yb = Xtr[idx], Ytr[idx]
        o, _ = mod.forward(Ws, Xb); _, Rm, _ = softmax_grad(o, Yb)
        _, gates = mod.forward(Ws, Xb)
        J, A, B = mod.operators(Ws, gates, batch)
        G = np.einsum("ni,nj->nij", Rm, Xb)
        if arm == "op":
            eta, k = hyper
            M = make_M(A, B, [W.shape for W in Ws], batch)
            sol = lsqr(M, (-eta * G).ravel(), atol=0.0, btol=0.0, conlim=0.0, iter_lim=int(k))
            dWs, off = [], 0
            for sh in [W.shape for W in Ws]:
                q = sh[0] * sh[1]; dWs.append(sol[0][off:off + q].reshape(sh)); off += q
        else:
            gg = [np.einsum("nji,njk,nlk->il", A[l], G, B[l]) for l in range(L)]
            b1, b2, e = 0.9, 0.999, 1e-8
            dWs = []
            for i, gi in enumerate(gg):
                m[i] = b1 * m[i] + (1 - b1) * gi
                v[i] = b2 * v[i] + (1 - b2) * gi * gi
                dWs.append(-hyper * (m[i] / (1 - b1 ** (step + 1)))
                           / (np.sqrt(v[i] / (1 - b2 ** (step + 1))) + e))
        Ws = [W + dw for W, dw in zip(Ws, dWs)]
        if not all(np.all(np.isfinite(W)) for W in Ws):
            stop = "exploded"; break
    b = min(trace, key=lambda z: z["train"])
    return {"arch": arch, "arm": arm, "hyper": hyper, "L": L, "width": width, "seed": seed,
            "stop": stop, "seconds": time.time() - t0, "best": b, "trace": trace}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["crelu"])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--adam-lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--etas", type=float, nargs="+", default=[8.0, 32.0])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/endpoint2.json")
    a = ap.parse_args()
    from olo.tasks.mnist1d import MNIST1D
    rows = []
    for s in a.seeds:
        t = MNIST1D(n_train=4000, n_val=64, n_test=1000, seed=s)   # the standard 80/20 split
        data = (t.Xtr.numpy().astype(float), t.ytr.numpy().astype(int),
                t.Xte.numpy().astype(float), t.yte.numpy().astype(int))
        for arch in a.archs:
            jobs = [("adam", lr) for lr in a.adam_lrs] + [("op", (e, a.k)) for e in a.etas]
            for arm, h in jobs:
                r = run(arch, arm, h, data, a.depth, a.width, a.batch, a.steps, a.every, s)
                rows.append(r); b = r["best"]
                print(f"{arch:<5}{arm:<5}{str(h):<12}s={s} | train {b['train']:.4f} "
                      f"test {b['test']:.3f}/{b['test_acc']:.3f} | PR {b['pr']:.2f} "
                      f"exp {b['exponent']:+.3f} | {r['stop']:<9} {r['seconds']:.0f}s", flush=True)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
