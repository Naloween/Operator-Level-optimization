"""Depth sweep, resumable: checkpointed, early-stopped, and extendable without recomputation.

Replaces `depth.py`. Three things that one lacked:

  * **Checkpoints.** Every cell writes weights, optimiser moments, step, stop reason and config to
    `runs/theory/depth_ckpt/`. Extending the step budget or adding hyper-parameters re-uses them.
  * **Resume with the right condition.** A cell is skipped only if it actually FINISHED
    (converged or exploded) or already reached the requested `--steps`. Skipping every recorded
    cell -- the obvious implementation -- means a warm start never fires and raising the budget
    silently does nothing.
  * **Early stopping.** Stop when the loss has stopped improving, or when it explodes. At depth
    most of the grid is dead on arrival and running it to the cap wastes the budget.

Learning-rate grid extends down to 1e-5: in very deep networks that is the range where Adam still
makes progress, so a grid that bottoms out at 1e-3 mis-reports the baseline.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import lsqr

import crelu as C, nonlinear as R
from diagnose import make_M, softmax_grad

MODS = {"relu": R, "crelu": C}


def tag(arch, L, arm, hyper, seed):
    h = ("%g" % hyper) if not isinstance(hyper, (tuple, list)) else "e%g_k%d" % tuple(hyper)
    return f"{arch}__L{L}__{arm}__{h}__s{seed}"


def save(path, Ws, m, v, step, stop, cfg, trace):
    np.savez_compressed(
        path,
        **{f"W{i}": W for i, W in enumerate(Ws)},
        **{f"m{i}": x for i, x in enumerate(m)},
        **{f"v{i}": x for i, x in enumerate(v)},
        step=step, stop=stop, cfg=json.dumps(cfg), trace=json.dumps(trace), n=len(Ws))


def load(path):
    d = np.load(path, allow_pickle=False)
    n = int(d["n"])
    return ([d[f"W{i}"] for i in range(n)], [d[f"m{i}"] for i in range(n)],
            [d[f"v{i}"] for i in range(n)], int(d["step"]), str(d["stop"]),
            json.loads(str(d["trace"])))


def best_of(trace):
    """Best-train-loss entry. Used on BOTH paths: a resumed cell must report what a fresh one
    would, or the two are silently incomparable."""
    return min(trace, key=lambda z: z["train"]) if trace else None


def step_dW(mod, Ws, Xb, Rm, arm, hyper, m, v, step, L):
    _, gates = mod.forward(Ws, Xb)
    J, A, B = mod.operators(Ws, gates, Xb.shape[0])
    G = np.einsum("ni,nj->nij", Rm, Xb)
    if arm == "op":
        eta, k = hyper
        M = make_M(A, B, [W.shape for W in Ws], Xb.shape[0])
        sol = lsqr(M, (-eta * G).ravel(), atol=0.0, btol=0.0, conlim=0.0, iter_lim=int(k))
        out, off = [], 0
        for sh in [W.shape for W in Ws]:
            q = sh[0] * sh[1]; out.append(sol[0][off:off + q].reshape(sh)); off += q
        return out
    g = [np.einsum("nji,njk,nlk->il", A[l], G, B[l]) for l in range(L)]
    if arm == "gd":
        return [-hyper * x for x in g]
    b1, b2, e = 0.9, 0.999, 1e-8
    out = []
    for i, gi in enumerate(g):
        m[i][...] = b1 * m[i] + (1 - b1) * gi
        v[i][...] = b2 * v[i] + (1 - b2) * gi * gi
        out.append(-hyper * (m[i] / (1 - b1 ** (step + 1)))
                   / (np.sqrt(v[i] / (1 - b2 ** (step + 1))) + e))
    return out


def train(arch, arm, hyper, data, L, width, steps, batch, seed, ckdir, resume,
          patience, tol, every):
    Xtr, Ytr, Xte, Yte = data
    mod = MODS[arch]
    path = Path(ckdir) / (tag(arch, L, arm, hyper, seed) + ".npz")
    cfg = {"arch": arch, "arm": arm, "hyper": hyper, "L": L, "width": width,
           "batch": batch, "seed": seed}
    rng = np.random.default_rng(seed)
    d_in, d_out = Xtr.shape[1], int(Ytr.max()) + 1

    start, trace, stop = 0, [], "max_steps"
    if resume and path.exists():
        Ws, m, v, start, stop, trace = load(path)
        if stop != "max_steps" or start >= steps:
            return {**cfg, "resumed": "skipped", "stop": stop, "steps": start,
                    "trace": trace, "final": best_of(trace)}
        rng = np.random.default_rng(seed + 10_000 + start)   # fresh batch stream after resume
    else:
        Ws = mod.init_net(d_in, width, d_out, L, "xavier", rng)
        m = [np.zeros_like(W) for W in Ws]; v = [np.zeros_like(W) for W in Ws]

    t0, best, bad = time.time(), np.inf, 0
    for step in range(start, steps + 1):
        idx = rng.choice(Xtr.shape[0], size=min(batch, Xtr.shape[0]), replace=False)
        Xb, Yb = Xtr[idx], Ytr[idx]
        out, _ = mod.forward(Ws, Xb)
        loss, Rm, _ = softmax_grad(out, Yb)
        if not np.isfinite(loss) or loss > 50:
            stop = "exploded"; break
        if step % every == 0 or step == steps:
            ote, _ = mod.forward(Ws, Xte)
            tl, _, ta = softmax_grad(ote, Yte)
            trace.append({"step": step, "train": loss, "test": tl, "acc": ta})
            if loss < best * (1 - tol):
                best, bad = loss, 0
            else:
                bad += 1
                if bad >= patience:
                    stop = "converged"; break
        if step == steps:
            break
        dWs = step_dW(mod, Ws, Xb, Rm, arm, hyper, m, v, step, L)
        Ws = [W + dw for W, dw in zip(Ws, dWs)]
        if not all(np.all(np.isfinite(W)) for W in Ws):
            stop = "exploded"; break

    Path(ckdir).mkdir(parents=True, exist_ok=True)
    save(path, Ws, m, v, step, stop, cfg, trace)
    return {**cfg, "resumed": "ran", "stop": stop, "steps": step, "seconds": time.time() - t0,
            "trace": trace, "final": best_of(trace)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["relu", "crelu"])
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=8, help="evals without improvement -> stop")
    ap.add_argument("--tol", type=float, default=1e-3, help="relative improvement that counts")
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[1e-5, 1e-4, 1e-3, 5e-3, 2e-2, 1e-1])
    ap.add_argument("--etas", type=float, nargs="+", default=[0.5, 2.0])
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 60])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ckdir", default="runs/theory/depth_ckpt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="runs/theory/depth2.json")
    a = ap.parse_args()

    from olo.tasks.mnist1d import MNIST1D
    rows, t0 = [], time.time()
    Path(a.ckdir).mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        t = MNIST1D(n_train=a.n_train, n_val=64, n_test=256, seed=s)
        data = (t.Xtr.numpy().astype(float), t.ytr.numpy().astype(int),
                t.Xte.numpy().astype(float), t.yte.numpy().astype(int))
        for arch in a.archs:
            for L in a.depths:
                jobs = ([("gd", lr) for lr in a.lrs] + [("adam", lr) for lr in a.lrs]
                        + [("op", (e, k)) for e in a.etas for k in a.ks])
                for arm, h in jobs:
                    r = train(arch, arm, h, data, L, a.width, a.steps, a.batch, s,
                              a.ckdir, a.resume, a.patience, a.tol, a.every)
                    rows.append(r)
                    f = r.get("final") or {}
                    print(f"{arch:<5} L={L:<3} {arm:<5} {str(h):<12} s={s} | "
                          f"train {f.get('train', float('nan')):.4f} acc {f.get('acc', float('nan')):.3f}"
                          f" | {r['stop']:<9} @{r['steps']:<5} {r['resumed']}"
                          f" ({r.get('seconds', 0):.0f}s)", flush=True)
                    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}  ({len(rows)} cells, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
