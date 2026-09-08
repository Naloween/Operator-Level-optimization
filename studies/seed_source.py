"""Where does the seed come from? Fluctuation, or something the task supplies?

A looks-linear CReLU network is *exactly* linear and *exactly* isometric, so by the
instability law it sits at a fixed point of the low-rank bias at every depth. It does not
stay there. The reason is exact (theory/04, Theorem 12): one gradient step from a
looks-linear configuration splits as

    dS_l     = -(eta/2) * mean_b [ R_b ]
    dDelta_l = -(eta/2) * mean_b [ R_b E_b ],      R_b := A_l^T G_b N_l^T,  E_b = diag(sign z_l(x_b))

so the symmetric part `S` -- the linear network -- moves under the mean residual, while the
*nonlinear* part `Delta`, which is the entire seed, moves under the **correlation between
the residual and the gate sign pattern**. Nothing else can create it.

That gives a sharp dichotomy, and it is measurable as `||dDelta_l|| / ||dS_l||`:

* if `R_b` and `E_b` are uncorrelated (a sign-symmetric input distribution and a force that
  does not know the sign of x), the seed is a mean-zero average of B terms and decays as
  **B^(-1/2)** -- a finite-batch artefact that vanishes in the full-batch limit;
* if the task correlates the residual with the gate pattern, the seed is **systematic** and
  the ratio plateaus at a value the task decides.

Slope -1/2 versus slope 0 on a log-log batch sweep separates the two, with no free
parameter. The prescribed symmetric force is the negative control that must give -1/2;
teacher-student and MNIST are the question.

Writes `runs/theory/seed_source.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from olo.models.crelu_mlp import CReLUMLP
from olo.tasks.teacher_student import TeacherStudent
from olo.tasks.vision import MNIST
from olo.theory.crelu import split_layer


@torch.no_grad()
def seed_ratio(net, X: torch.Tensor, G: torch.Tensor) -> list[dict]:
    """Per gated layer: how much of the gradient goes into `Delta` rather than `S`.

    Model-agnostic and assumption-free -- it splits the *realized* weight gradient, so it
    stays meaningful once the network has left the looks-linear configuration.
    """
    ctx = net.all_contexts(X)
    B = X.shape[0]
    grow = lambda T: T.expand(B, *T.shape[1:]) if T.shape[0] == 1 else T
    Gb = grow(G)
    out = []
    for l in range(1, net.depth):          # layer 0 has no gate, hence no S/Delta split
        A, Bc = ctx[l]
        # mean_b A_b^T G_b Bc_b^T, contracted in one pass: the batch sweep reaches 2048
        # and a Python loop over it would dominate the whole study.
        g = torch.einsum("bij,bik,blk->jl", grow(A), Gb, grow(Bc)) / B
        S, D = split_layer(g)
        out.append({"layer": l, "s_norm": float(S.norm()), "delta_norm": float(D.norm()),
                    "ratio": float(D.norm() / S.norm().clamp_min(1e-300))})
    return out


def prescribed_force(net, X, p: float = 0.0, c: float = 1.0):
    """`G = -c U diag(s^p) V^T`: symmetric in x by construction, so `E[R E] = 0`."""
    J = net.operator(X)
    U, S, Vh = torch.linalg.svd(J)
    k = S.shape[-1]
    return -c * (U[..., :, :k] * S.clamp_min(1e-300).pow(p).unsqueeze(-2)) @ Vh[..., :k, :]


def make(task_name: str, width: int, depth: int, seed: int, n_pool: int, mu: float = 0.0):
    """(net, X_pool, gradient_fn). Every task uses a looks-linear CReLU net."""
    gen = torch.Generator().manual_seed(seed + 17)
    if task_name == "prescribed":
        # x ~ N(mu*1, I). At looks-linear the force is the same matrix R for every input,
        # so the seed is exactly `R * mean_b E_b` and its systematic part is
        # `2*Phi(m_i) - 1` at `m = N_l mu*1` -- a dose-response curve in mu with a closed
        # form, running from a pure B^(-1/2) fluctuation at mu=0 to saturation.
        net = CReLUMLP(d_in=width, d_out=width, width=width, depth=depth).double()
        net.initialize("looks_linear", seed=seed)
        X = torch.randn(n_pool, width, generator=gen, dtype=torch.float64) + mu
        return net, X, (lambda n, x: prescribed_force(n, x))

    if task_name == "teacher_student":
        task = TeacherStudent(d=width, n=n_pool, target="orthogonal", seed=seed)
        task.to(torch.device("cpu"), torch.float64)
        net = CReLUMLP(d_in=width, d_out=width, width=width, depth=depth).double()
        net.initialize("looks_linear", seed=seed)
        idx = {"X": task.X, "Y": task.Y}
        return net, idx["X"], (lambda n, x, t=task: t.operator_gradient(
            n, x, t.Y[: x.shape[0]]))

    if task_name == "mnist":
        task = MNIST(n_train=n_pool, n_val=256, n_test=256, seed=seed)
        task.to(torch.device("cpu"), torch.float64)
        net = CReLUMLP(d_in=task.d_in, d_out=task.d_out, width=width, depth=depth).double()
        net.initialize("looks_linear", seed=seed)
        X, Y = task.train_batch(0, n_pool)
        return net, X, (lambda n, x, t=task, Y=Y: t.operator_gradient(n, x, Y[: x.shape[0]]))

    raise ValueError(f"unknown task {task_name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+",
                    default=["prescribed", "teacher_student", "mnist"])
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--mu", type=float, nargs="+", default=[0.0],
                    help="input mean shift for the prescribed task: 0 is sign-symmetric")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/theory/seed_source.json")
    a = ap.parse_args()

    out, t0 = [], time.time()
    for task_name in a.tasks:
      for mu in (a.mu if task_name == "prescribed" else [0.0]):
        for depth in a.depths:
            for seed in a.seeds:
                net, X, grad_fn = make(task_name, a.width, depth, seed,
                                       max(a.batches), mu)
                rows = []
                for B in a.batches:
                    if B > X.shape[0]:
                        continue
                    Xb = X[:B]
                    r = seed_ratio(net, Xb, grad_fn(net, Xb))
                    rows.append({"batch": B,
                                 "ratio_mean": float(np.mean([v["ratio"] for v in r])),
                                 "ratio_max": float(max(v["ratio"] for v in r)),
                                 "layers": r})
                Bs = np.log([v["batch"] for v in rows])
                ys = np.log([v["ratio_mean"] for v in rows])
                slope = float(np.polyfit(Bs, ys, 1)[0]) if len(rows) > 2 else float("nan")
                out.append({"task": task_name, "depth": depth, "seed": seed, "mu": mu,
                            "width": a.width, "slope": slope, "rows": rows})
                print(f"{task_name:<16} L={depth:<4} mu={mu:<5g} seed={seed}  slope={slope:+.3f}  "
                      f"ratio(B={rows[0]['batch']})={rows[0]['ratio_mean']:.3e} -> "
                      f"ratio(B={rows[-1]['batch']})={rows[-1]['ratio_mean']:.3e} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "runs": out}))
    print(f"wrote {a.out}  ({len(out)} sweeps, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
