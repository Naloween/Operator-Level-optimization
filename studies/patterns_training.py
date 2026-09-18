"""Realised versus fixed gate patterns, ALONG TRAINING.

At initialisation the two agree, and trivially so: the weights are independent of the data, so no
pattern is special. The question only has content once training has shaped the weights.

The comparison is between two trajectories of the same network:

  P(x, t)   the operator at a realised input. Its gates flip as the weights move, so the
            trajectory is discontinuous -- and the flips are driven by the data.
  J_eps(t)  the operator at a gate pattern FIXED at t = 0 and held. A polynomial in the weights,
            so the trajectory is smooth, and by gate absorption it is a deep linear chain
            throughout.

Training is driven by the realised patterns in both cases; only the measurement differs. If the
realised exponent departs from the fixed-pattern exponent as training proceeds, the departure is
attributable to the data reshaping which patterns are active -- an effect with no deep linear
analogue. We also track how far the realised patterns have drifted from their own initial value,
so a null result can be distinguished from "the patterns never moved".
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

import crelu as C, nonlinear as R
from diagnose import softmax_grad
from patterns_exp import exponent, realised_gates

MODS = {"relu": R, "crelu": C}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["crelu", "relu"])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="runs/theory/patterns_training.json")
    a = ap.parse_args()
    from olo.tasks.mnist1d import MNIST1D
    rows = []
    for s in a.seeds:
        t = MNIST1D(n_train=4000, n_val=64, n_test=256, seed=s)
        Xtr, Ytr = t.Xtr.numpy().astype(float), t.ytr.numpy().astype(int)
        for arch in a.archs:
            mod = MODS[arch]
            L, d = a.depth, a.width
            rng = np.random.default_rng(s)
            Ws = mod.init_net(Xtr.shape[1], d, int(Ytr.max()) + 1, L, "xavier", rng)
            m = [np.zeros_like(W) for W in Ws]; v = [np.zeros_like(W) for W in Ws]
            # patterns fixed at t=0 and held for the whole run
            fixed = [[(rng.random(d) < 0.5).astype(float) for _ in range(L - 1)]
                     if arch == "relu" else
                     [rng.standard_normal(d) for _ in range(L - 1)]
                     for _ in range(a.n_probe)]
            g0 = [realised_gates(Ws, Xtr[i], arch) for i in range(a.n_probe)]
            trace, t0 = [], time.time()
            for step in range(a.steps + 1):
                if step % a.every == 0 or step == a.steps:
                    re_ = [exponent(Ws, realised_gates(Ws, Xtr[i], arch), arch)
                           for i in range(a.n_probe)]
                    fx_ = [exponent(Ws, f, arch) for f in fixed]
                    gt = [realised_gates(Ws, Xtr[i], arch) for i in range(a.n_probe)]
                    drift = float(np.mean([np.mean([( (z>0) != (z0>0) ).mean()
                                                    for z, z0 in zip(g, h)])
                                           for g, h in zip(gt, g0)]))
                    o, _ = mod.forward(Ws, Xtr[:512])
                    loss, _, acc = softmax_grad(o, Ytr[:512])
                    f = lambda q: float(np.nanmean(q))
                    trace.append({"step": step, "loss": loss, "acc": acc,
                                  "realised": f(re_), "fixed": f(fx_),
                                  "gap": f(re_) - f(fx_), "gate_drift": drift})
                    print(f"  {arch:<6}s={s} step {step:>5} | loss {loss:.4f} acc {acc:.3f} | "
                          f"realised {f(re_):.4f}  fixed {f(fx_):.4f}  gap {f(re_)-f(fx_):+.4f} | "
                          f"gate drift {drift:.3f}", flush=True)
                if step == a.steps:
                    break
                idx = rng.choice(Xtr.shape[0], size=a.batch, replace=False)
                Xb, Yb = Xtr[idx], Ytr[idx]
                out, g = mod.forward(Ws, Xb)
                _, Rm, _ = softmax_grad(out, Yb)
                J, A, B = mod.operators(Ws, g, a.batch)
                G = np.einsum("ni,nj->nij", Rm, Xb)
                gg = [((A[l].transpose(0, 2, 1) @ G) @ B[l].transpose(0, 2, 1)).sum(0)
                      for l in range(L)]
                b1, b2, e_ = 0.9, 0.999, 1e-8
                for i, gi in enumerate(gg):
                    m[i] = b1 * m[i] + (1 - b1) * gi
                    v[i] = b2 * v[i] + (1 - b2) * gi * gi
                    Ws[i] = Ws[i] - a.lr * (m[i] / (1 - b1 ** (step + 1))) / (
                        np.sqrt(v[i] / (1 - b2 ** (step + 1))) + e_)
                if not all(np.all(np.isfinite(W)) for W in Ws):
                    break
            rows.append({"arch": arch, "L": L, "seed": s, "seconds": time.time() - t0,
                         "trace": trace})
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
