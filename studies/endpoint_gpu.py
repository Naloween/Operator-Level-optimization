"""Adam versus the reachable ideal on standard MNIST-1D -- GPU implementation.

Same experiment as `endpoint2.py`, on the `gpu.py` step map. Binary by design: Adam at the
reference recipe for this dataset against a step that realises the best reachable operator change
(k large enough to BE the ideal, not a point on a dial). Standard setting: 4000 train / 1000 test,
batch 100, and the reference 6000-step budget, which the CPU version could not afford.

Precision. Training runs in float32; the spectral probes (effective rank and gain exponent) are
computed in float64, since they are measurements rather than steps.

Arms are compared AT MATCHED TRAINING LOSS afterwards, using the recorded trajectories, not at a
matched step count.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

import gpu


def softmax_grad(out, Y):
    p = torch.softmax(out, dim=1)
    n = out.shape[0]
    loss = float(-torch.log(p[torch.arange(n, device=out.device), Y].clamp_min(1e-30)).mean())
    R = p.clone()
    R[torch.arange(n, device=out.device), Y] -= 1.0
    acc = float((out.argmax(1) == Y).to(torch.float32).mean())
    return loss, R / n, acc


def probes(Ws, Xpr, arch, gen):
    """Effective rank of P(x) and the gain exponent, in float64."""
    W64 = [w.double() for w in Ws]
    X64 = Xpr.double()
    _, gs = gpu.forward(W64, X64, arch)
    A, B = gpu.contexts(W64, gs, X64, arch)
    P = A[0] @ W64[0] @ B[0]
    sv = torch.linalg.svdvals(P)
    pr = float(((sv.sum(1) ** 2) / (sv ** 2).sum(1).clamp_min(1e-300)).mean())
    exps = []
    for i in range(X64.shape[0]):
        Xi = X64[i:i + 1]
        _, gi = gpu.forward(W64, Xi, arch)
        Ai, Bi = gpu.contexts(W64, gi, Xi, arch)
        Pi = Ai[0] @ W64[0] @ Bi[0]
        D = torch.randn(Pi.shape, generator=gen, device="cpu", dtype=torch.float64).to(Pi.device)
        M = gpu.StepMap(Ai, Bi, [tuple(w.shape) for w in W64])
        dP = M.mv(M.rmv(D.reshape(-1))).view(Pi.shape)
        U, s, Vh = torch.linalg.svd(Pi[0])
        ss, cc = [], []
        for k in range(s.shape[0]):
            if s[k] <= 1e-10 * s[0]:
                continue
            dem = float(U[:, k] @ D[0] @ Vh[k]); rea = float(U[:, k] @ dP[0] @ Vh[k])
            if abs(dem) > 1e-14 and rea / dem > 0:
                ss.append(float(s[k])); cc.append(rea / dem)
        if len(ss) >= 4:
            x, y = np.log(np.array(ss)), np.log(np.array(cc))
            xc = x - x.mean()
            if float(xc @ xc) > 1e-12:
                exps.append(float(xc @ (y - y.mean()) / (xc @ xc)))
    return pr, (float(np.mean(exps)) if exps else float("nan"))


def run(arch, arm, hyper, data, L, width, batch, steps, every, seed, dev):
    Xtr, Ytr, Xte, Yte = data
    Ws = gpu.init_net(Xtr.shape[1], width, int(Ytr.max()) + 1, L, arch, seed, dev)
    m = [torch.zeros_like(w) for w in Ws]; v = [torch.zeros_like(w) for w in Ws]
    g = torch.Generator().manual_seed(seed)
    gpr = torch.Generator().manual_seed(3)
    Xpr = Xtr[:8]
    trace, t0, stop = [], time.time(), "max_steps"
    for step in range(steps + 1):
        if step % every == 0 or step == steps:
            with torch.no_grad():
                o, _ = gpu.forward(Ws, Xtr, arch); trl, _, tra = softmax_grad(o, Ytr)
                o, _ = gpu.forward(Ws, Xte, arch); tel, _, tea = softmax_grad(o, Yte)
                pr, ex = probes(Ws, Xpr, arch, torch.Generator().manual_seed(3))
            trace.append({"step": step, "train": trl, "train_acc": tra, "test": tel,
                          "test_acc": tea, "pr": pr, "exponent": ex})
            if not np.isfinite(trl) or trl > 50:
                stop = "exploded"; break
        if step == steps:
            break
        idx = torch.randint(0, Xtr.shape[0], (batch,), generator=g).to(dev)
        Xb, Yb = Xtr[idx], Ytr[idx]
        with torch.no_grad():
            o, _ = gpu.forward(Ws, Xb, arch)
            _, R, _ = softmax_grad(o, Yb)
            if arm == "op":
                eta, k = hyper
                # `softmax_grad` returns the residual divided by the batch, which is what the
                # WEIGHT gradient needs. The per-sample OPERATOR target must not carry that 1/n:
                # eta is the operator step size, so it should be read directly as a fraction of
                # ||P(x)||, not as n times it. Undo the division here.
                dWs = gpu.op_step(Ws, Xb, R * batch, eta, int(k), arch)
            else:
                gg = gpu.coord_grad(Ws, Xb, R, arch)
                b1, b2, e = 0.9, 0.999, 1e-8
                dWs = []
                for i, gi in enumerate(gg):
                    m[i].mul_(b1).add_(gi, alpha=1 - b1)
                    v[i].mul_(b2).addcmul_(gi, gi, value=1 - b2)
                    dWs.append(-hyper * (m[i] / (1 - b1 ** (step + 1)))
                               / ((v[i] / (1 - b2 ** (step + 1))).sqrt() + e))
            Ws = [w + dw for w, dw in zip(Ws, dWs)]
        if not all(torch.isfinite(w).all() for w in Ws):
            stop = "exploded"; break
    b = min(trace, key=lambda z: z["train"])
    return {"arch": arch, "arm": arm, "hyper": hyper, "L": L, "width": width, "seed": seed,
            "stop": stop, "seconds": time.time() - t0, "best": b, "trace": trace}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=["crelu", "relu"])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--adam-lrs", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2])
    ap.add_argument("--etas", type=float, nargs="+", default=[8.0, 32.0])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs/theory/endpoint_gpu.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from olo.tasks.mnist1d import MNIST1D
    rows = []
    for s in a.seeds:
        t = MNIST1D(n_train=4000, n_val=64, n_test=1000, seed=s)
        data = tuple(x.to(dev) for x in
                     (t.Xtr.float(), t.ytr.long(), t.Xte.float(), t.yte.long()))
        for arch in a.archs:
            jobs = [("adam", lr) for lr in a.adam_lrs] + [("op", (e, a.k)) for e in a.etas]
            for arm, h in jobs:
                r = run(arch, arm, h, data, a.depth, a.width, a.batch, a.steps, a.every, s, dev)
                rows.append(r); b = r["best"]
                print(f"{arch:<5}{arm:<5}{str(h):<12}s={s} | train {b['train']:.4f} "
                      f"test {b['test']:.3f}/{b['test_acc']:.3f} | PR {b['pr']:.2f} "
                      f"exp {b['exponent']:+.3f} | {r['stop']:<9} {r['seconds']:.0f}s", flush=True)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
