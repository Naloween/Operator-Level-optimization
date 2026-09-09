"""Does separation buy alignment outside the setting the theorem was proved in?

Everything in `theory/04` §2-§3 replaces the exact per-mode velocity by its diagonal part,
which is free exactly when each subproduct is aligned with the whole product. Theorem 6.1 of
Haas et al. (ICML 2026) derives that alignment from spectral separation, in a *fixed-gates
linear* setting and subject to the subproduct being large enough. Whether it survives into
trained and nonlinear networks has not been checked, and it is the one hypothesis the rate
law still stands on.

So: train, and track along the trajectory

* `separation`   -- log(s_1/s_d), the variable the theorem says drives alignment;
* `alignment`    -- worst per-layer diagonality of `A_l A_l^T` in `U` and `B_l^T B_l` in `V`;
* `reduction_error` -- what the diagonal replacement actually costs, `||exact - diag||/||exact||`,
  which is the quantity that matters (alignment only matters through it);
* `p`, `R^2`     -- the task exponent by Proposition 15, and how well the rich-get-richer
  form fits at all.

**The prediction.** If Theorem 6.1 transfers, `reduction_error` falls as `separation` grows,
with a floor where the subproduct condition fails. A null or positive correlation would say
the modal reduction is not bought by separation in this regime, and that the rate law needs
a different justification.

`p` is measured on both a synthetic teacher and MNIST, because file 04 §5.5 found `p < 0`
on a well-conditioned synthetic target -- a task that *opposes* the bias -- and whether real
data does the same decides how much of the depth story is about the task rather than the
architecture.

Writes `runs/theory/alignment.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from olo.models.crelu_mlp import CReLUMLP
from olo.models.deep_linear import DeepLinear
from olo.tasks.teacher_student import TeacherStudent
from olo.tasks.vision import MNIST
from olo.theory.alignment import modal_reduction
from olo.theory.instability import amplification, log_velocity_exponent

MODELS = {"deep_linear": DeepLinear, "crelu_mlp": CReLUMLP}


def rotated_net(depth: int, width: int, r0: float, theta: float, seed: int = 0):
    """`W_l = R(theta) diag(exp(a/L))`: separation set by `r0`, misalignment by `theta`.

    The two are independent knobs, which is what the training runs cannot offer -- there
    separation, misalignment and training time all move together, so a correlation between
    any two of them is uninterpretable.
    """
    net = DeepLinear(d_in=width, d_out=width, width=width, depth=depth).double()
    g = torch.Generator().manual_seed(seed)
    a = torch.linspace(-0.5, 0.5, width, dtype=torch.float64) * r0
    with torch.no_grad():
        for W in net.weights:
            A = torch.randn(width, width, generator=g, dtype=torch.float64)
            R = torch.linalg.matrix_exp(theta * (A - A.T) / np.sqrt(width))
            W.copy_(R @ torch.diag(torch.exp(a / depth)))
    return net


def controlled(depths, width, r0s, thetas, seed: int = 0) -> list[dict]:
    """Sweep separation at fixed misalignment: the clean version of the Thm 6.1 test."""
    G = torch.randn(1, width, width, generator=torch.Generator().manual_seed(seed + 7),
                    dtype=torch.float64)
    X = torch.eye(width, dtype=torch.float64)
    out = []
    for L in depths:
        for theta in thetas:
            for r0 in r0s:
                mc = modal_reduction(rotated_net(L, width, r0, theta, seed), X, G)
                out.append({"part": "controlled", "depth": L, "theta": theta, "r0": r0,
                            "separation": mc.separation, "alignment": mc.alignment,
                            "reduction_error": mc.reduction_error, "cosine": mc.cosine})
    return out


def make_task(name: str, width: int, n: int, seed: int):
    if name == "mnist":
        t = MNIST(n_train=n, n_val=256, n_test=256, seed=seed)
    else:
        t = TeacherStudent(d=width, n=n, target="orthogonal", seed=seed)
    return t.to(torch.device("cpu"), torch.float64)


def snapshot(net, task, X, Y, depth: int) -> dict:
    G = task.operator_gradient(net, X, Y)
    mc = modal_reduction(net, X, G)
    amp = amplification(mc.s, mc.exact)
    phi = log_velocity_exponent(depth)
    return {
        "separation": mc.separation,
        "alignment": mc.alignment,
        "diag_A_min": float(mc.diag_A.min()),
        "diag_B_min": float(mc.diag_B.min()),
        "reduction_error": mc.reduction_error,
        "cosine": mc.cosine,
        "psi": amp.psi,
        "p": amp.psi - phi,
        "r2": amp.r2,
        "rate": amp.rate,
        "uniform": amp.uniform,
        "s_max": float(mc.s.max()),
        "s_min": float(mc.s.min()),
    }


def run(model: str, init: str, task_name: str, depth: int, width: int, steps: int,
        lr: float, batch: int, eval_every: int, seed: int, diag_batch: int) -> dict:
    torch.manual_seed(seed)
    task = make_task(task_name, width, max(512, batch), seed)
    net = MODELS[model](d_in=task.d_in, d_out=task.d_out, width=width, depth=depth).double()
    net.initialize(init, seed=seed)
    opt = torch.optim.SGD(net.parameters(), lr=lr / depth)

    rec = []
    for step in range(steps + 1):
        X, Y = task.train_batch(step, batch)
        if step % eval_every == 0:
            row = snapshot(net, task, X[:diag_batch], Y[:diag_batch], depth)
            row["step"] = step
            with torch.no_grad():
                row["loss"] = float(task.loss(task.forward(net, X), Y))
            rec.append(row)
            if not np.isfinite(row["loss"]) or row["loss"] > 1e10:
                break
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        task.loss(task.forward(net, X), Y).backward()
        opt.step()

    return {"model": model, "init": init, "task": task_name, "depth": depth,
            "width": width, "lr": lr / depth, "seed": seed, "trace": rec}


def correlate(rec: dict) -> dict:
    """Spearman-style rank correlation of reduction_error with separation, along the run."""
    tr = [r for r in rec["trace"]
          if np.isfinite(r["reduction_error"]) and np.isfinite(r["separation"])]
    if len(tr) < 5:
        return {"n": len(tr)}
    sep = np.array([r["separation"] for r in tr])
    err = np.array([r["reduction_error"] for r in tr])
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    rho = float(np.corrcoef(rank(sep), rank(err))[0, 1]) if np.ptp(sep) > 0 else float("nan")
    fin = lambda k: [r[k] for r in tr if np.isfinite(r[k])]
    return {
        "n": len(tr), "rho_sep_vs_error": rho,
        "sep_first": float(sep[0]), "sep_last": float(sep[-1]),
        "err_first": float(err[0]), "err_last": float(err[-1]),
        "err_min": float(err.min()), "err_max": float(err.max()),
        "alignment_last": tr[-1]["alignment"],
        "p_median": float(np.median(fin("p"))) if fin("p") else float("nan"),
        "r2_median": float(np.median(fin("r2"))) if fin("r2") else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["crelu_mlp", "deep_linear"])
    ap.add_argument("--inits", nargs="+", default=["xavier", "looks_linear"])
    ap.add_argument("--tasks", nargs="+", default=["teacher_student", "mnist"])
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--diag-batch", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--controlled-only", action="store_true")
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.0, 0.05, 0.2, 0.5, 1.0])
    ap.add_argument("--r0s", type=float, nargs="+",
                    default=[0.01, 0.1, 0.5, 2.0, 8.0, 20.0])
    ap.add_argument("--out", default="runs/theory/alignment.json")
    a = ap.parse_args()

    ctrl = controlled(a.depths, a.width, a.r0s, a.thetas)
    for theta in a.thetas:
        row = [c for c in ctrl if c["theta"] == theta and c["depth"] == a.depths[-1]]
        print(f"  theta={theta:<5g} err vs r0: " +
              " ".join(f"{c['reduction_error']:.4f}" for c in row), flush=True)
    if a.controlled_only:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "controlled": ctrl,
                                           "runs": []}))
        print(f"wrote {a.out} (controlled only)")
        return

    jobs = [(m, i, t, d, s) for m in a.models for i in a.inits for t in a.tasks
            for d in a.depths for s in a.seeds
            if not (m == "deep_linear" and i == "looks_linear")]
    out, t0 = [], time.time()
    for n, (m, i, t, d, s) in enumerate(jobs):
        rec = run(m, i, t, d, a.width, a.steps, a.lr, a.batch, a.eval_every, s,
                  a.diag_batch)
        rec["summary"] = correlate(rec)
        out.append(rec)
        c = rec["summary"]
        print(f"[{n+1}/{len(jobs)}] {m:<11}{i:<13}{t:<16}L={d:<4}s={s} "
              f"sep {c.get('sep_first', float('nan')):.2f}->{c.get('sep_last', float('nan')):.2f}  "
              f"err {c.get('err_first', float('nan')):.3f}->{c.get('err_last', float('nan')):.3f}  "
              f"rho={c.get('rho_sep_vs_error', float('nan')):+.2f}  "
              f"p={c.get('p_median', float('nan')):+.2f} R2={c.get('r2_median', float('nan')):.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "controlled": ctrl, "runs": out}))
    print(f"wrote {a.out}  ({len(out)} runs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
