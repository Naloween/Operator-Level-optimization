"""When does the low-rank bias help, and when does it cost? A sweep over the target's rank.

The endpoint experiment showed that replacing gradient descent's step with the reachable ideal
gives higher-rank solutions at matched training loss and essentially equal accuracy. That says the
bias is a low-rank prior but not when the prior is worth having. A single sweep answers both: fix
everything and vary only the RANK OF THE TARGET.

Teacher: y = W2 relu(W1 x) with W1 of shape (r, d_in) and W2 of shape (d_out, r), so the teacher's
Jacobian has rank at most r. Sweeping r moves the target from strongly low-rank to full-rank.

The prediction going in was that the biased arm (Adam) should win at small r and lose at large r, a
crossover being stronger evidence than two hand-picked settings. A crossover is what came out, but
with the OPPOSITE SIGN: Adam is worse on low-rank targets and only wins once the target is
high-rank. Nor is the gap explained by rank -- the two arms' participation ratios agree at every r
-- so whatever drives it is not the bias that the endpoint experiment measures. Recorded here so
the file does not keep asserting a prediction the data contradicts; the mechanism is still open.

Why this needs a NONLINEAR student. In a deep linear network P is shared across all inputs, so
enough samples pin the minimiser and both arms must agree -- we measured a gap of 1e-9 in that
case. With input-dependent gates each sample constrains only P(x_i) x_i = y_i, a rank-one slice of
its own operator, so the operator keeps enormous freedom and the bias has room to act.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import gpu


def teacher(d_in, d_out, r, seed, dev):
    g = torch.Generator().manual_seed(seed + 7919)
    W1 = (torch.randn(r, d_in, generator=g) / d_in ** 0.5).to(dev)
    W2 = (torch.randn(d_out, r, generator=g) / r ** 0.5).to(dev)
    return lambda X: torch.relu(X @ W1.T) @ W2.T


def mnist1d_denoise(r, n_train, n_test, sigma, seed, dev):
    """The same rank axis, on real data: denoise MNIST-1D onto its top-r principal subspace.

    Input is a noisy signal, target is the rank-r projection of the clean one, so the map is not a
    function of the input alone and the optimal predictor is genuinely nonlinear -- a linear network
    would not do. Only the target's rank moves across the sweep; the data, the shapes, the loss and
    the noise are identical, which is what makes it the real-data counterpart of `teacher` rather
    than a second, differently-confounded experiment.
    """
    from olo.tasks.mnist1d import MNIST1D
    t = MNIST1D(n_train=n_train, n_val=64, n_test=n_test, seed=seed)
    Str = t.Xtr.to(dev).double().float()
    Ste = t.Xte.to(dev).double().float()
    U = torch.linalg.svd(Str - Str.mean(0), full_matrices=False)[2]       # (40, 40), rows = PCs
    Ur = U[:r]
    g = torch.Generator().manual_seed(seed + 31)
    noise = lambda S: S + sigma * torch.randn(S.shape, generator=g).to(dev)
    proj = lambda S: (S @ Ur.T) @ Ur
    return noise(Str), proj(Str), noise(Ste), proj(Ste)


def mse(out, Y):
    R = out - Y
    return float((R * R).sum(1).mean()), R          # per-sample residual, unnormalised


def probe_pr(Ws, Xpr, arch):
    """Mean participation ratio of P(x), averaged over the inputs where P(x) is not zero.

    A low-rank teacher such as y = w2 relu(w1.x) is identically zero on a positive-measure set of
    inputs, and a student that matches it there by killing its gates has P(x) = 0 exactly. Those
    samples carry no spectrum, and folding their zero into the mean drags the average below 1 --
    below the value PR takes on a rank-one operator -- which is not a rank statement at all. So they
    are excluded and counted, and the dead fraction is reported next to the ratio as its own signal.
    """
    W64 = [w.double() for w in Ws]
    _, gs = gpu.forward(W64, Xpr.double(), arch)
    A, B = gpu.contexts(W64, gs, Xpr.double(), arch)
    P = A[0] @ W64[0] @ B[0]
    sv = torch.linalg.svdvals(P)
    live = sv[:, 0] > 1e-9 * sv[:, 0].max().clamp_min(1e-300)
    dead = float((~live).float().mean())
    if not bool(live.any()):
        return float("nan"), dead
    s = sv[live]
    return float(((s.sum(1) ** 2) / (s ** 2).sum(1)).mean()), dead


def probe_gates(Ws, Xpr, arch):
    """Unit-level gate statistics, which are NOT what `probe_pr`'s dead fraction measures.

    `probe_pr` reports the fraction of *inputs* whose operator is identically zero. This reports the
    fraction of *hidden units* that are off for every probe input -- units the network has given up,
    which shrink its effective width -- and the mean fraction of gates open. The two come apart: a
    network can have every unit alive somewhere yet still annihilate particular inputs, or carry
    dead units while no single input is fully blocked.

    Note what `leaky` does and does not remove. A leaky unit can still sit in its negative regime
    for every input, so this statistic does NOT go to zero there -- it measures the sign pattern,
    not connectivity. What leaky removes is the *consequence*: such a unit still contributes
    alpha*z, so its Jacobian entry is never zero and no path is ever cut. That is precisely what
    makes it a control -- the sign statistic is free to behave as it does under ReLU while the
    annihilation it normally causes is switched off.
    """
    if arch == "crelu":
        return float("nan"), float("nan")
    _, gs = gpu.forward(Ws, Xpr, arch)
    on = [g > 0.5 for g in gs]                       # 0/1 for relu, 0.1/1 for leaky
    dead = sum(float((~o.any(0)).sum()) for o in on)
    total = sum(o.shape[1] for o in on)
    density = float(torch.cat([o.reshape(-1) for o in on]).float().mean())
    return dead / total, density


def run(arch, arm, hyper, data, L, width, batch, steps, every, seed, dev):
    Xtr, Ytr, Xte, Yte = data
    Ws = gpu.init_net(Xtr.shape[1], width, Ytr.shape[1], L, arch, seed, dev)
    m = [torch.zeros_like(w) for w in Ws]; v = [torch.zeros_like(w) for w in Ws]
    g = torch.Generator().manual_seed(seed)
    trace, t0 = [], time.time()
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            with torch.no_grad():
                o, _ = gpu.forward(Ws, Xtr, arch); tr, _ = mse(o, Ytr)
                o, _ = gpu.forward(Ws, Xte, arch); te, _ = mse(o, Yte)
                pr, dead = probe_pr(Ws, Xtr[:64], arch)
                du, dens = probe_gates(Ws, Xtr[:256], arch)
            trace.append({"step": step, "train": tr, "test": te, "pr": pr, "dead": dead,
                          "dead_units": du, "density": dens})
            if not np.isfinite(tr) or tr > 1e6:
                break
        if step == steps:
            break
        idx = torch.randint(0, Xtr.shape[0], (batch,), generator=g).to(dev)
        Xb, Yb = Xtr[idx], Ytr[idx]
        with torch.no_grad():
            o, _ = gpu.forward(Ws, Xb, arch)
            _, R = mse(o, Yb)
            if arm == "op":
                eta, k = hyper
                dWs = gpu.op_step(Ws, Xb, R / R.norm().clamp_min(1e-12), eta, int(k), arch)
            else:
                gg = gpu.coord_grad(Ws, Xb, R / batch, arch)
                b1, b2, e = 0.9, 0.999, 1e-8
                dWs = []
                for i, gi in enumerate(gg):
                    m[i].mul_(b1).add_(gi, alpha=1 - b1)
                    v[i].mul_(b2).addcmul_(gi, gi, value=1 - b2)
                    dWs.append(-hyper * (m[i] / (1 - b1 ** (step + 1)))
                               / ((v[i] / (1 - b2 ** (step + 1))).sqrt() + e))
            Ws = [w + dw for w, dw in zip(Ws, dWs)]
        if not all(torch.isfinite(w).all() for w in Ws):
            break
    b = min(trace, key=lambda z: z["train"])
    return {"arch": arch, "arm": arm, "hyper": hyper, "rank": None, "seed": seed,
            "seconds": time.time() - t0, "best": b, "trace": trace}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="teacher", choices=["teacher", "mnist1d"])
    ap.add_argument("--sigma", type=float, default=0.3, help="mnist1d input noise")
    ap.add_argument("--arch", default="relu")
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--adam-lrs", type=float, nargs="*", default=[1e-3, 3e-3])
    ap.add_argument("--etas", type=float, nargs="*", default=[0.1, 0.3])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/rank_sweep.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    print(f"{'r':>3}{'arm':>6}{'hyper':>10} | {'train':>10}{'test':>10}{'PR':>7}{'deadX':>7}{'deadU':>7}{'dens':>7}{'s':>4}")
    for s in a.seeds:
        for r in a.ranks:
            if a.task == "mnist1d":
                data = mnist1d_denoise(r, a.n_train, a.n_test, a.sigma, s, dev)
            else:
                tf = teacher(a.dim, a.dim, r, s, dev)
                gg = torch.Generator().manual_seed(s + 11)
                Xtr = torch.randn(a.n_train, a.dim, generator=gg).to(dev)
                Xte = torch.randn(a.n_test, a.dim, generator=gg).to(dev)
                data = (Xtr, tf(Xtr), Xte, tf(Xte))
            for arm, hs in (("adam", a.adam_lrs), ("op", [(e, a.k) for e in a.etas])):
                for h in hs:
                    row = run(a.arch, arm, h, data, a.depth, a.width, a.batch,
                              a.steps, a.every, s, dev)
                    row["rank"] = r
                    rows.append(row); b = row["best"]
                    hs_ = f"{h:g}" if not isinstance(h, tuple) else f"({h[0]:g})"
                    print(f"{r:>3}{arm:>6}{hs_:>10} | {b['train']:>10.4f}{b['test']:>10.4f}"
                          f"{b['pr']:>7.2f}{b['dead']:>7.2f}{b['dead_units']:>7.2f}"
                          f"{b['density']:>7.2f}{s:>4}", flush=True)
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
