"""E2 along training: does the architecture-independence of the gain exponent survive training?

At initialisation the exponent is essentially the same for deep linear, CReLU and ReLU (1.318,
1.309, 1.319 at L=8), sitting below 2-2/L by an amount the Cauchy-Schwarz decomposition accounts
for exactly. That is the spine of the paper's claim -- that the gain law is set by balance and
alignment rather than by the nonlinearity -- and it is currently an INITIALISATION-TIME statement
only.

The earlier attempt to measure it along training (`q4.py`) probed the gain with a random target
and is void: that ratio equals c_k only when the context Grams are diagonalised by the singular
vectors of P, which holds under balancedness and fails otherwise. This script uses the correct
rank-one probe throughout,

    c_k = <u_k, T(u_k v_k^T) v_k> = sum_l ||A_l^T u_k||^2 ||B_l v_k||^2 ,

and reports the exponent, the intercept, and the misalignment slope d log R / d log s -- which by
the identity c_k = s_k^2 R_k must equal exponent - 2 at every step.

Training is Adam on MNIST-1D, identical for all three architectures so the comparison is about the
architecture and nothing else.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

import dln as DL, crelu as C, nonlinear as R
from diagnose import softmax_grad
def slope(x, y):
    xc = x - x.mean()
    v = float(xc @ xc)
    return float(xc @ (y - y.mean()) / v) if v > 1e-12 else float("nan")

MODS = {"dln": DL, "relu": R, "crelu": C}


def contexts_for(mod, Ws, X, i):
    if mod is DL:
        return mod.operators(Ws, [], 1)
    _, g = mod.forward(Ws, X[i:i + 1])
    return mod.operators(Ws, g, 1)


def measure(mod, Ws, X, n_probe, floor=1e-10):
    """Exponent, intercept and misalignment slope, averaged over probe inputs."""
    L = len(Ws)
    wn = [np.linalg.norm(W, 2) for W in Ws]
    ex, it, ms = [], [], []
    for i in range(1 if mod is DL else n_probe):
        J, A, B = contexts_for(mod, Ws, X, i)
        if not np.all(np.isfinite(J)):
            continue
        U, s, Vt = np.linalg.svd(J[0])
        sk, ck, Rk = [], [], []
        for k in range(len(s)):
            if s[k] <= floor * s[0]:
                continue
            na = [np.linalg.norm(A[l][0].T @ U[:, k]) for l in range(L)]
            nb = [np.linalg.norm(B[l][0] @ Vt[k]) for l in range(L)]
            c = sum((na[l] * nb[l]) ** 2 for l in range(L))
            if c <= 0:
                continue
            r = [na[l] * nb[l] * wn[l] / max(s[k], 1e-300) for l in range(L)]
            sk.append(s[k]); ck.append(c)
            Rk.append(sum((r[l] / wn[l]) ** 2 for l in range(L)))
        if len(sk) < 4:
            continue
        ls = np.log(np.array(sk))
        e = slope(ls, np.log(np.array(ck)))
        if not np.isfinite(e):
            continue
        ex.append(e)
        it.append(float(np.exp(np.log(ck).mean() - e * ls.mean())))
        ms.append(slope(ls, np.log(np.array(Rk))))
    f = lambda v: (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2
    return f(ex), f(it), f(ms)


def run(arch, L, width, data, steps, every, lr, batch, n_probe, seed):
    Xtr, Ytr = data
    mod = MODS[arch]
    rng = np.random.default_rng(seed)
    Ws = mod.init_net(Xtr.shape[1], width, int(Ytr.max()) + 1, L, "xavier", rng)
    m = [np.zeros_like(W) for W in Ws]; v = [np.zeros_like(W) for W in Ws]
    trace, t0 = [], time.time()
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            (e, esd), (it, _), (ms, _) = measure(mod, Ws, Xtr, n_probe)
            o, _ = mod.forward(Ws, Xtr[:512]) if mod is not DL else (Xtr[:512] @ mod.operators(Ws, [], 1)[0][0].T, None)
            loss, _, acc = softmax_grad(o, Ytr[:512])
            trace.append({"step": step, "loss": loss, "acc": acc, "exponent": e,
                          "exp_sd": esd, "intercept": it, "misalign_slope": ms,
                          "identity": e - (2 + ms)})
            print(f"  {arch:<6}L={L:<3}s={seed} step {step:>5} | loss {loss:.4f} | "
                  f"exp {e:.4f}±{esd:.3f} (2-2/L={2-2/L:.4f}) | misalign {ms:+.4f} | "
                  f"identity {e-(2+ms):+.1e}", flush=True)
        if step == steps:
            break
        idx = rng.choice(Xtr.shape[0], size=batch, replace=False)
        Xb, Yb = Xtr[idx], Ytr[idx]
        if mod is DL:
            J, A, B = mod.operators(Ws, [], batch)
            out = Xb @ J[0].T
        else:
            out, g = mod.forward(Ws, Xb); J, A, B = mod.operators(Ws, g, batch)
        _, Rm, _ = softmax_grad(out, Yb)
        G = np.einsum("ni,nj->nij", Rm, Xb)
        # batched matmul, not a 3-operand einsum: the latter is not BLAS-backed and dominates
        gg = [((A[l].transpose(0, 2, 1) @ G) @ B[l].transpose(0, 2, 1)).sum(0) for l in range(L)]
        b1, b2, e_ = 0.9, 0.999, 1e-8
        for i, gi in enumerate(gg):
            m[i] = b1 * m[i] + (1 - b1) * gi
            v[i] = b2 * v[i] + (1 - b2) * gi * gi
            Ws[i] = Ws[i] - lr * (m[i] / (1 - b1 ** (step + 1))) / (np.sqrt(v[i] / (1 - b2 ** (step + 1))) + e_)
        if not all(np.all(np.isfinite(W)) for W in Ws):
            break
    return {"arch": arch, "L": L, "width": width, "lr": lr, "seed": seed,
            "seconds": time.time() - t0, "trace": trace}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["dln", "crelu", "relu"])
    ap.add_argument("--depths", type=int, nargs="+", default=[8])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="runs/theory/e2_training.json")
    a = ap.parse_args()
    from olo.tasks.mnist1d import MNIST1D
    rows = []
    for s in a.seeds:
        t = MNIST1D(n_train=4000, n_val=64, n_test=256, seed=s)
        data = (t.Xtr.numpy().astype(float), t.ytr.numpy().astype(int))
        for L in a.depths:
            for arch in a.archs:
                rows.append(run(arch, L, a.width, data, a.steps, a.every, a.lr,
                                a.batch, a.n_probe, s))
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
