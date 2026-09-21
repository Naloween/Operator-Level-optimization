"""Find a configuration in which the model actually learns MNIST-1D, before comparing optimisers.

The endpoint comparison is only meaningful between two runs that have learned the task. Measured
on the existing runs, they had not: the configurations that drive the training loss to 1e-4 reach
0.44-0.57 test accuracy, while the only one reaching 0.63-0.66 never fits the training set and was
therefore excluded from the matched-loss comparison. That comparison was between two over-fitted
runs, and no conclusion about implicit bias survives it.

This sweeps width, depth and step size to locate the ceiling of the bias-free architecture the
framework requires (f(x) = P(x) x needs no biases), selecting on a held-out validation split and
reporting test accuracy at the selected point, so the number is not the maximum over a trajectory.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch, gpu


def data(n_train, n_val, n_test, seed, dev):
    from olo.tasks.mnist1d import MNIST1D
    t = MNIST1D(n_train=n_train, n_val=n_val, n_test=n_test, seed=seed)
    return (t.Xtr.to(dev), t.ytr.to(dev), t.Xva.to(dev), t.yva.to(dev),
            t.Xte.to(dev), t.yte.to(dev))


def ce(out, Y):
    p = torch.softmax(out - out.max(1, keepdim=True).values, 1)
    n = out.shape[0]
    loss = float(-torch.log(p[torch.arange(n, device=out.device), Y].clamp_min(1e-30)).mean())
    R = p.clone(); R[torch.arange(n, device=out.device), Y] -= 1.0
    return loss, R, float((out.argmax(1) == Y).float().mean())


def run(arch, arm, hyper, D, L, width, batch, steps, every, seed, dev, k=50):
    Xtr, Ytr, Xva, Yva, Xte, Yte = D
    Ws = gpu.init_net(Xtr.shape[1], width, int(Ytr.max()) + 1, L, arch, seed, dev)
    m = [torch.zeros_like(w) for w in Ws]; v = [torch.zeros_like(w) for w in Ws]
    g = torch.Generator().manual_seed(seed)
    best = (-1.0, -1.0, 0, 0.0)                    # val acc, test acc, step, train loss
    t0 = time.time()
    for step in range(steps + 1):
        if step % every == 0:
            with torch.no_grad():
                tl, _, _ = ce(gpu.forward(Ws, Xtr, arch)[0], Ytr)
                _, _, va = ce(gpu.forward(Ws, Xva, arch)[0], Yva)
                _, _, te = ce(gpu.forward(Ws, Xte, arch)[0], Yte)
            if va > best[0]:
                best = (va, te, step, tl)
        if step == steps:
            break
        idx = torch.randint(0, Xtr.shape[0], (batch,), generator=g).to(dev)
        with torch.no_grad():
            out, _ = gpu.forward(Ws, Xtr[idx], arch)
            _, R, _ = ce(out, Ytr[idx])
            if arm == "op":
                dWs = gpu.op_step(Ws, Xtr[idx], R / R.norm().clamp_min(1e-12), hyper, k, arch)
            else:
                gg = gpu.coord_grad(Ws, Xtr[idx], R / batch, arch)
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
    return {"arch": arch, "arm": arm, "hyper": hyper, "L": L, "width": width, "seed": seed,
            "val": best[0], "test": best[1], "at_step": best[2], "train": best[3],
            "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["crelu", "relu"])
    ap.add_argument("--arm", default="adam", choices=["adam", "op"])
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--hypers", type=float, nargs="+", default=[3e-3, 1e-2, 3e-2])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/tune_mnist.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    print(f"{'arch':>6}{'L':>3}{'w':>5}{'hyper':>8} | {'val':>6}{'test':>6}{'step':>7}{'train':>8}{'s':>6}")
    for s in a.seeds:
        D = data(a.n_train, 1000, 1000, s, dev)
        for arch in a.archs:
            for L in a.depths:
                for w in a.widths:
                    for h in a.hypers:
                        r = run(arch, a.arm, h, D, L, w, a.batch, a.steps, a.every, s, dev, a.k)
                        rows.append(r)
                        print(f"{arch:>6}{L:>3}{w:>5}{h:>8g} | {r['val']:>6.3f}{r['test']:>6.3f}"
                              f"{r['at_step']:>7}{r['train']:>8.3f}{r['seconds']:>6.0f}", flush=True)
                        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                        Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
