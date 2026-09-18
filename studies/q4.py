"""Q4: is the gain exponent a property of the architecture, or of gradient descent's step?

`studies/gainlaw.py` measured, at initialisation, that the exponent
`d log c / d log s` saturates near 1.0 under Xavier -- identically for deep linear, CReLU and
ReLU, and not at `2 - 2/L`. Two things that measurement cannot settle, and this one can:

  (a) ALONG TRAINING. The initialisation-time exponent may not survive training. If it drifts to
      `2 - 2/L`, the E2 result is an artefact of measuring at step 0. This is the check that can
      hurt the contribution, so it is run first.

  (b) IS IT GD's CHOICE? Gradient descent is `k = 1` on the Krylov dial: its induced operator
      change is `M M^T D*`, one Landweber step. The achievable ideal is the converged solve. If a
      trajectory driven by the ideal step has a DIFFERENT exponent, then the exponent is a
      property of the update rule, not of the architecture -- and there is no way to learn that
      except by taking the other step. In a deep linear network the counterfactual collapses to
      convex gradient descent and says nothing; with input-dependent gates there is no
      operator-space flow to collapse to, so the solver is the only route. **This is the question
      that requires the machinery.**

Each arm trains from the same initialisation; at intervals we measure the intrinsic gain exponent
(one input at a time, as in `gainlaw.py`) of the arm's OWN induced step.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import lsqr

import crelu as C, nonlinear as R, dln as DL
from diagnose import make_M, softmax_grad
from gainlaw import realised, fit, pairs_from

MODS = {"dln": DL, "relu": R, "crelu": C}


def exponent_now(mod, Ws, X, eta, k, n_probe, rng):
    """Intrinsic exponent of the step this ARM would take, one input at a time.

    k = 1 reproduces the gradient step exactly (the first Krylov iterate is the gradient
    direction); larger k moves toward the achievable ideal.
    """
    ex, ins = [], []
    for i in range(min(n_probe, X.shape[0])):
        Xi = X[i:i + 1]
        out, g = mod.forward(Ws, Xi)
        J, A, B = mod.operators(Ws, g, 1)
        D = rng.standard_normal((1, J.shape[1], J.shape[2]))       # probe all directions
        if k <= 1:
            dJ = realised(A, B, D)                                  # M M^T D*  = GD
        else:
            M = make_M(A, B, [W.shape for W in Ws], 1)
            sol = lsqr(M, D.ravel(), atol=0.0, btol=0.0, conlim=0.0, iter_lim=k)
            dJ = (M @ sol[0]).reshape(D.shape)
        sl, it, _ = fit(pairs_from(J, dJ, D, 0))
        if np.isfinite(sl):
            ex.append(sl); ins.append(it)
    return (float(np.mean(ex)) if ex else float("nan"),
            float(np.std(ex)) if ex else float("nan"),
            float(np.mean(ins)) if ins else float("nan"))


def run(arch, arm_k, Xtr, Ytr, L, width, steps, batch, lr, eta, every, n_probe, seed):
    mod = MODS[arch]
    rng = np.random.default_rng(seed)
    d_in, d_out = Xtr.shape[1], int(Ytr.max()) + 1
    Ws = mod.init_net(d_in, width, d_out, L, "xavier", rng)
    m = [np.zeros_like(W) for W in Ws]; v = [np.zeros_like(W) for W in Ws]
    trace, t0 = [], time.time()
    for step in range(steps + 1):
        idx = rng.choice(Xtr.shape[0], size=min(batch, Xtr.shape[0]), replace=False)
        Xb, Yb = Xtr[idx], Ytr[idx]
        out, _ = mod.forward(Ws, Xb)
        loss, Rm, acc = softmax_grad(out, Yb)
        if step % every == 0 or step == steps:
            e, esd, ic = exponent_now(mod, Ws, Xb, eta, arm_k, n_probe,
                                      np.random.default_rng(7 + step))
            trace.append({"step": step, "loss": loss, "acc": acc,
                          "exponent": e, "exp_sd": esd, "intercept": ic})
            print(f"  {arch} L={L} k={arm_k:<4} step {step:>4} | loss {loss:.4f} "
                  f"| exponent {e:.4f} +-{esd:.3f}  (2-2/L = {2-2/L:.4f})  intercept {ic:.2f}",
                  flush=True)
        if step == steps:
            break
        _, gates = mod.forward(Ws, Xb)
        J, A, B = mod.operators(Ws, gates, Xb.shape[0])
        G = np.einsum("ni,nj->nij", Rm, Xb)
        if arm_k <= 1:                                   # plain gradient descent
            g = [np.einsum("nji,njk,nlk->il", A[l], G, B[l]) for l in range(L)]
            dWs = [-lr * x for x in g]
        else:                                            # operator step, k Krylov iterations
            M = make_M(A, B, [W.shape for W in Ws], Xb.shape[0])
            sol = lsqr(M, (-eta * G).ravel(), atol=0.0, btol=0.0, conlim=0.0, iter_lim=arm_k)
            dWs, off = [], 0
            for sh in [W.shape for W in Ws]:
                q = sh[0] * sh[1]; dWs.append(sol[0][off:off + q].reshape(sh)); off += q
        Ws = [W + dw for W, dw in zip(Ws, dWs)]
        if not all(np.all(np.isfinite(W)) for W in Ws):
            break
    return {"arch": arch, "k": arm_k, "L": L, "lr": lr, "eta": eta, "seed": seed,
            "seconds": time.time() - t0, "trace": trace}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["crelu", "relu"])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 30])
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--n-probe", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/q4.json")
    a = ap.parse_args()
    from olo.tasks.mnist1d import MNIST1D
    rows = []
    for s in a.seeds:
        t = MNIST1D(n_train=a.n_train, n_val=32, n_test=128, seed=s)
        Xtr, Ytr = t.Xtr.numpy().astype(float), t.ytr.numpy().astype(int)
        for arch in a.archs:
            for k in a.ks:
                rows.append(run(arch, k, Xtr, Ytr, a.depth, a.width, a.steps,
                                a.batch, a.lr, a.eta, a.every, a.n_probe, s))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
