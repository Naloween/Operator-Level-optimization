"""Mode A: score every optimiser's step against the reference target, from the SAME weights.

The endpoint comparisons (`endpoint_gpu.py`, `rank_sweep.py`) let each optimiser run its own
course and compare where they land. That answers "do they end up somewhere different" but cannot
answer "is this step better than that step", because by step 100 the two arms sit at different
weights and every per-step number is confounded by position.

This probe removes that confound. One optimiser drives a single reference trajectory. At probe
steps we stop, and from *those* weights ask what each candidate would do, scoring all of them
against the same target. Nothing differs between candidates except the update rule.

What is measured, at weights W with per-sample operator gradient G(x) (N3):

    ideal target        D*     = -eta G(x)            what the loss asks the operator to do
    reachable ideal     Pi D*  = M M^+ D*             the most of it the architecture admits
    ideal weight step   dW_id  = M^+ D*               the minimum-norm weight update realising it

and for each candidate step dW_m:

    cos_op   = cos( M(dW_m), Pi D* )      how well its realised OPERATOR change points at the ideal
    cos_w    = cos( dW_m, dW_id )         how well its WEIGHT update points at the ideal one
    ratio_op = ||M(dW_m)|| / ||Pi D*||    and the same in norm, which unlike cos depends on lr
    ratio_w  = ||dW_m|| / ||dW_id||

**The cosines are scale-invariant, so they do not depend on the learning rate at all.** That is the
point of reporting them: the direction comparison is hyperparameter-free, so it cannot be
attacked the way a tuned-vs-untuned endpoint comparison can. Only the two norm ratios carry a
hyperparameter, and they are reported separately for that reason.

Adam is stateful, so its step is not a function of W alone. Its moments are carried along the
reference trajectory at *every* step, so `adam` here means "what Adam would do having followed this
path" -- well defined, and exactly equal to the real thing when Adam is itself the reference.

LIMITATION, measured, do not ignore: `cos_op` is reliable at every depth tested; `cos_w` is NOT.
At L=16 two solves that realise the same operator change to cos 0.976-0.997 can have near-orthogonal
weight vectors (linear L=16: cos_w = 0.045). Both are valid solutions differing by an element of
ker(M), so cos_w is then comparing an arbitrary choice within the solution set rather than the
methods. Note this is NOT caught by `solve_ne`: that same cell has solve_ne = 2.8e-9, an excellent
least-squares residual. Measured clean (cos_w = 1.000000 for all four architectures) at L in {4,8}.
So: read `cos_op` as the directional result, and treat `cos_w` as informative only at shallow depth.

Writes `runs/theory/probe_steps.json`.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import gpu

K_IDEAL = 8000          # cap only: the solve stops itself on stagnation (see gpu.lsqr `stall`)
ATOL_IDEAL = 0.0
STALL = 1e-10           # REQUIRED: without it LSQR diverges past convergence and returns garbage


def flat(dWs):
    return torch.cat([d.reshape(-1) for d in dWs])


def cos(a, b):
    na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def task(name, n_train, n_test, d, seed, dev):
    if name == "mnist1d":
        from olo.tasks.mnist1d import MNIST1D
        t = MNIST1D(n_train=n_train, n_val=64, n_test=n_test, seed=seed)
        return (t.Xtr.to(dev), t.ytr.to(dev), t.Xte.to(dev), t.yte.to(dev), "ce")
    g = torch.Generator().manual_seed(seed + 7919)
    W1 = (torch.randn(d, d, generator=g) / d ** 0.5).to(dev)
    W2 = (torch.randn(d, d, generator=g) / d ** 0.5).to(dev)
    tf = lambda X: torch.relu(X @ W1.T) @ W2.T
    gg = torch.Generator().manual_seed(seed + 11)
    Xtr = torch.randn(n_train, d, generator=gg).to(dev)
    Xte = torch.randn(n_test, d, generator=gg).to(dev)
    return (Xtr, tf(Xtr), Xte, tf(Xte), "mse")


def residual(out, Y, kind):
    """Per-sample dL/df, unnormalised by n -- the operator target must not carry the 1/n."""
    if kind == "ce":
        z = out - out.max(1, keepdim=True).values
        p = torch.softmax(z, 1)
        R = p.clone()
        R[torch.arange(out.shape[0], device=out.device), Y] -= 1.0
        loss = float(-torch.log(p[torch.arange(out.shape[0]), Y].clamp_min(1e-30)).mean())
        return loss, R, float((out.argmax(1) == Y).float().mean())
    R = out - Y
    return float((R * R).sum(1).mean()), R, float("nan")


def build(Ws, X, R, eta, arch):
    """The step map at the current weights, the ideal target, and the converged ideal solve.

    Promoted to float64: gpu.py trains in float32 by design, but its own docstring reserves float64
    for the diagnostics, and this solve is the one place where that matters most.
    """
    Ws = [w.double() for w in Ws]
    X, R = X.double(), R.double()
    _, gs = gpu.forward(Ws, X, arch)
    A, B = gpu.contexts(Ws, gs, X, arch)
    shapes = [tuple(w.shape) for w in Ws]
    M = gpu.StepMap(A, B, shapes)
    G = R.unsqueeze(2) * X.unsqueeze(1)
    Dstar = (-eta * G).reshape(-1)
    # Damped (Tikhonov) solve. The undamped iteration drifts into ker(M), and the step map's
    # nullity is largest exactly for the input-independent architectures -- measured at L=6,
    # width 8: 19% CReLU, 53% ReLU, but 83% deep linear and 93% FGLN, because every layer there
    # moves the same shared operator so the rank is capped at d_out*d_in regardless of the
    # parameter count. Undamped, FGLN returned alpha > 1 (impossible for a projection ratio) on
    # roughly half its probes; damped, ker is trivial and it cannot drift.
    dW_id = gpu.solve_min_norm(M, Dstar, iters=K_IDEAL, lam_rel=1e-7, stall=STALL)
    # Quality of the ideal solve, recorded at every probe so a bad one is never trusted silently.
    # Everything downstream is measured against dW_id and Pi D*, so if this is not small the whole
    # record is meaningless -- deep CReLU in particular does not always converge within the cap.
    ne = float(torch.linalg.vector_norm(M.rmv(M.mv(dW_id) - Dstar))
               / torch.linalg.vector_norm(M.rmv(Dstar)).clamp_min(1e-300))
    return M, Dstar, dW_id, M.mv(dW_id), shapes, ne


def candidates(Ws, X, R, eta, arch, M, shapes, adam_state, lr, ks, step):
    """Every candidate step, from the SAME weights. Returns {name: flat dW}."""
    out = {}
    g = gpu.coord_grad(Ws, X, R / X.shape[0], arch)
    out["gd"] = flat([-lr * x for x in g])
    m, v = adam_state
    b1, b2, e = 0.9, 0.999, 1e-8
    dWs = []
    for i, gi in enumerate(g):
        mh = m[i] / (1 - b1 ** (step + 1))
        vh = v[i] / (1 - b2 ** (step + 1))
        dWs.append(-lr * mh / (vh.sqrt() + e))
    out["adam"] = flat(dWs)
    # `stall` here too: the candidate is "truncated LSQR at k", and past convergence LSQR does not
    # stagnate, it diverges -- so an unguarded large-k candidate scores its own numerical blow-up
    # rather than the method. At the small k that form the dial the guard never fires.
    Dstar = (-eta * (R.unsqueeze(2) * X.unsqueeze(1))).reshape(-1)
    for k in ks:
        out[f"op{k}"] = gpu.lsqr(M, Dstar, k, 0.0, stall=1e-6)   # the dial: k as given, undamped
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="relu",
                    choices=["linear", "fgln", "relu", "leaky", "crelu"])
    ap.add_argument("--init", default="he",
                    choices=["he", "xavier", "orthogonal", "looks_linear"])
    ap.add_argument("--task", default="teacher", choices=["teacher", "mnist1d"])
    ap.add_argument("--ref", default="adam", choices=["adam", "gd", "op"],
                    help="which optimiser drives the single reference trajectory")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20, 50, 200])
    ap.add_argument("--ref-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/theory/probe_steps.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, Ytr, Xte, Yte, kind = task(a.task, a.n_train, a.n_test, a.dim, a.seed, dev)
    d_out = (int(Ytr.max()) + 1) if kind == "ce" else Ytr.shape[1]
    Ws = gpu.init_net(Xtr.shape[1], a.width, d_out, a.depth, a.arch, a.seed, dev,
                      init=a.init)
    m = [torch.zeros_like(w) for w in Ws]
    v = [torch.zeros_like(w) for w in Ws]
    g = torch.Generator().manual_seed(a.seed)
    rows, t0 = [], time.time()

    for step in range(a.steps + 1):
        idx = torch.randint(0, Xtr.shape[0], (a.batch,), generator=g).to(dev)
        Xb, Yb = Xtr[idx], Ytr[idx]
        with torch.no_grad():
            out, _ = gpu.forward(Ws, Xb, a.arch)
            loss, R, _ = residual(out, Yb, kind)

            if step % a.probe_every == 0:
                M, Dstar, dW_id, PiD, shapes, ne = build(Ws, Xb, R, a.eta, a.arch)
                nD = torch.linalg.vector_norm(Dstar)
                nPi = torch.linalg.vector_norm(PiD)
                cand = candidates([w.double() for w in Ws], Xb.double(), R.double(), a.eta,
                                  a.arch, M, shapes,
                                  ([x.double() for x in m], [x.double() for x in v]),
                                  a.lr, a.ks, step)
                rec = {"step": step, "loss": loss, "solve_ne": ne,
                       "alpha": float(nPi / nD.clamp_min(1e-30))}
                for name, dW in cand.items():
                    realised = M.mv(dW)
                    rec[name] = {
                        "cos_op": cos(realised, PiD),
                        "cos_w": cos(dW, dW_id),
                        "ratio_op": float(torch.linalg.vector_norm(realised) / nPi.clamp_min(1e-30)),
                        "ratio_w": float(torch.linalg.vector_norm(dW)
                                         / torch.linalg.vector_norm(dW_id).clamp_min(1e-30)),
                    }
                with torch.no_grad():
                    o, _ = gpu.forward(Ws, Xte, a.arch)
                    tl, _, ta = residual(o, Yte, kind)
                rec["test_loss"], rec["test_acc"] = tl, ta
                rows.append(rec)
                print(f"step {step:>5} loss {loss:.4f} alpha {rec['alpha']:.3f} "
                      f"ne {ne:.1e} | " +
                      "  ".join(f"{n} {rec[n]['cos_op']:+.3f}" for n in cand), flush=True)

            # the reference trajectory itself
            gg = gpu.coord_grad(Ws, Xb, R / a.batch, a.arch)
            b1, b2, e = 0.9, 0.999, 1e-8
            for i, gi in enumerate(gg):                      # moments carried at EVERY step
                m[i].mul_(b1).add_(gi, alpha=1 - b1)
                v[i].mul_(b2).addcmul_(gi, gi, value=1 - b2)
            if a.ref == "adam":
                dWs = [-a.lr * (m[i] / (1 - b1 ** (step + 1)))
                       / ((v[i] / (1 - b2 ** (step + 1))).sqrt() + e) for i in range(len(Ws))]
            elif a.ref == "gd":
                dWs = [-a.lr * x for x in gg]
            else:
                dWs = gpu.op_step(Ws, Xb, R / R.norm().clamp_min(1e-12), a.eta, a.ref_k, a.arch)
            Ws = [w + dw for w, dw in zip(Ws, dWs)]
        if not all(torch.isfinite(w).all() for w in Ws):
            print("diverged"); break

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
