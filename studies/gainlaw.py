"""Q1: does the per-input gain law hold, and with what exponent?

In a deep linear network with balanced initialisation the whole implicit bias reduces to one
exponent (Arora, Cohen & Hazan 2018, Thm 1): for the induced operator change `dJ = -eta T(G)`,
the component on singular direction k of J satisfies

    c_k := (u_k^T dJ v_k) / (u_k^T (-eta G) v_k)  =  L * s_k^{2 - 2/L},

so `d log c / d log s = 2 - 2/L` and the intercept is `L`. Verified here to 1e-15 for L = 2..32,
which is what makes the measurement trustworthy; anything that fails it is a broken pipeline, not
a finding.

Everything known about the nonlinear case concerns ENDPOINTS, LIMITS or RANKS -- gradient flow on
ReLU need not minimise rank (Timor, Vardi & Shamir 2023), the representation cost tends to the
bottleneck rank as depth grows with regularisation (Jacot 2023), the first escape is low-rank in
deeper layers (Bantzis, Simon & Jacot 2025), effective rank falls with depth empirically (Huh et
al. 2021). The finite-time RATE with input-dependent gates is what is untouched, so that is what
is measured here.

Two quantities, because debugging the deep linear case showed they differ:

  intrinsic   one input at a time (n=1). The gain law of the architecture at that input, with no
              interference from other inputs. This is the direct analogue of the linear law.
  batch       the same ratio when a single weight step must serve the whole batch. The gap
              between the two is the cross-input interference, which has no analogue in the
              linear theory since there J is shared.

Writes `runs/theory/gainlaw.json`.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

import dln as DL, crelu as C, nonlinear as R

MODS = {"dln": DL, "relu": R, "crelu": C}


def realised(A, B, D):
    """M M^T D : the operator change gradient descent induces, per sample."""
    L = len(A)
    gW = [np.einsum("nji,njk,nlk->il", A[l], D, B[l]) for l in range(L)]
    return sum(A[l] @ gW[l] @ B[l] for l in range(L))


def fit(pairs):
    p = np.array([q for q in pairs if q[1] > 0])
    if len(p) < 4:
        return float("nan"), float("nan"), len(p)
    x, y = np.log(p[:, 0]), np.log(p[:, 1])
    xc = x - x.mean()
    v = float(xc @ xc)
    if v < 1e-12:                       # flat spectrum: no exponent is identifiable
        return float("nan"), float("nan"), len(p)
    sl = float(xc @ (y - y.mean()) / v)
    return sl, float(np.exp(y.mean() - sl * x.mean())), len(p)


def pairs_from(J, dJ, D, idx, floor=1e-10):
    U, s, Vt = np.linalg.svd(J[idx])
    out = []
    for k in range(len(s)):
        if s[k] <= floor * s[0]:
            continue
        dem = float(U[:, k] @ D[idx] @ Vt[k])
        rea = float(U[:, k] @ dJ[idx] @ Vt[k])
        if abs(dem) > 1e-14:
            out.append((s[k], rea / dem))
    return out


def cell(arch, init, L, d, X, rng, probe):
    mod = MODS[arch]
    d_in, d_out = X.shape[1], X.shape[1]
    if arch == "crelu" and init == "balanced":
        # W_l = [O_l | -O_l] with O_l = U_l S^{1/L} U_{l-1}^T: the gate absorbs (W D(z) = O_l for
        # every z) so J = U_L S U_0^T is balanced AND non-flat -- the control that shows the
        # nonlinear pipeline recovers the linear law when balance holds.
        Us = [np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(L + 1)]
        S = np.diag(np.exp(np.linspace(0.8, -0.8, d)) ** (1.0 / L))
        Ws = []
        for l in range(L):
            O = Us[l + 1] @ S @ Us[l].T
            Ws.append(O if l == 0 else np.concatenate([O, -O], axis=1))
        X = X[:, :d]
    elif arch == "dln" and init == "balanced":
        Ws = DL.balanced_net(d, L, np.exp(np.linspace(0.8, -0.8, d)), rng)
        X = X[:, :d]
        d_in = d_out = d
    else:
        Ws = mod.init_net(d_in, d, d_out, L, init, rng)

    # ---- intrinsic: one input at a time
    ex, ins = [], []
    for i in range(X.shape[0]):
        Xi = X[i:i + 1]
        out, g = mod.forward(Ws, Xi)
        J, A, B = mod.operators(Ws, g, 1)
        D = (rng.standard_normal((1, J.shape[1], J.shape[2])) if probe == "random"
             else np.einsum("ni,nj->nij", out - rng.standard_normal(out.shape), Xi))
        sl, it, n = fit(pairs_from(J, realised(A, B, D), D, 0))
        if np.isfinite(sl):
            ex.append(sl); ins.append(it)

    # ---- batch: one step serving all inputs
    out, g = mod.forward(Ws, X)
    J, A, B = mod.operators(Ws, g, X.shape[0])
    D = (rng.standard_normal(J.shape) if probe == "random"
         else np.einsum("ni,nj->nij", out - rng.standard_normal(out.shape), X))
    dJ = realised(A, B, D)
    bex = [fit(pairs_from(J, dJ, D, i))[0] for i in range(X.shape[0])]
    bex = [q for q in bex if np.isfinite(q)]

    return {"arch": arch, "init": init, "L": L, "target": 2 - 2 / L,
            "intrinsic_mean": float(np.mean(ex)) if ex else float("nan"),
            "intrinsic_sd": float(np.std(ex)) if ex else float("nan"),
            "intercept_mean": float(np.mean(ins)) if ins else float("nan"),
            "batch_mean": float(np.mean(bex)) if bex else float("nan"),
            "batch_sd": float(np.std(bex)) if bex else float("nan"),
            "n_inputs": len(ex)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--n-inputs", type=int, default=16)
    ap.add_argument("--probe", default="random", choices=["random", "loss"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/theory/gainlaw.json")
    a = ap.parse_args()

    from olo.tasks.mnist1d import MNIST1D
    t = MNIST1D(n_train=256, n_val=32, n_test=32, seed=0)
    Xfull = t.Xtr.numpy().astype(float)

    rows = []
    print(f"{'arch':<6}{'init':<10}{'L':>3} | {'intrinsic':>18} {'2-2/L':>7} {'err':>8} | "
          f"{'batch':>18} | {'intercept':>10} {'L':>4}")
    for arch, init in (("dln", "balanced"), ("crelu", "balanced"),
                       ("dln", "xavier"), ("crelu", "xavier"), ("relu", "xavier")):
        for L in a.depths:
            rs = [cell(arch, init, L, a.width, Xfull[:a.n_inputs],
                       np.random.default_rng(100 * s + L), a.probe) for s in a.seeds]
            im = np.nanmean([r["intrinsic_mean"] for r in rs])
            isd = np.nanmean([r["intrinsic_sd"] for r in rs])
            bm = np.nanmean([r["batch_mean"] for r in rs])
            bsd = np.nanmean([r["batch_sd"] for r in rs])
            ic = np.nanmean([r["intercept_mean"] for r in rs])
            rows += rs
            tgt = 2 - 2 / L
            print(f"{arch:<6}{init:<10}{L:>3} | {im:>10.4f} +-{isd:>6.3f} {tgt:>7.4f} "
                  f"{abs(im-tgt):>8.1e} | {bm:>10.4f} +-{bsd:>6.3f} | {ic:>10.3f} {L:>4}",
                  flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
