"""The comparison grid: what the spectrum and the gradient bias do, across the whole design.

Three architectures x three initialisations x three tasks x three seeds, trained full-batch
at a small learning rate until the loss stops moving. At intervals we record

* the **Jacobian spectrum** of `J(x)` on a fixed probe batch — singular values, the
  participation-ratio effective rank `tr(M)^2/tr(M^2)`, separation `log(s_1/s_d)`;
* the **gradient bias** `Delta J - (-eta G)` with `G = E_x[dL/dJ(x)]`, measured by taking the
  real step and differencing the real operator. Three numbers:
    - `cos(Delta J, -G)`      direction,
    - `||Delta J|| / eta||G||` scale,
    - `sin = sqrt(1 - cos^2)`  the **scale-free** bias, which is what survives after the
      learning rate is tuned and so is the only part that is implicit bias at all
      (theory/10, Thm 3.1).

The three initialisations are matched across architectures so the comparison is about the
architecture and not about the starting point:

    xavier      standard everywhere
    orthogonal  ReLU: Haar | CReLU: looks-linear with Haar blocks | residual: Haar
    identity    ReLU: I    | CReLU: [I | -I]                      | residual: W = 0
                (all three give an operator that is the identity, or an isometry, at step 0)

Writes `runs/theory/grid.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from olo.init.schemes import haar
from olo.models.crelu_mlp import CReLUMLP
from olo.models.relu_mlp import ReLUMLP
from olo.models.residual_mlp import ResidualReLUMLP
from olo.tasks.teacher_student import TeacherStudent
from olo.tasks.mnist1d import MNIST1D

ARCH = {"relu_mlp": ReLUMLP, "crelu_mlp": CReLUMLP, "residual_mlp": ResidualReLUMLP}

#: init name -> the scheme each architecture uses to realise it
INIT = {
    "xavier":     {"relu_mlp": "xavier",   "crelu_mlp": "xavier",       "residual_mlp": "xavier"},
    "orthogonal": {"relu_mlp": "haar",     "crelu_mlp": "looks_linear", "residual_mlp": "haar"},
    "identity":   {"relu_mlp": "identity", "crelu_mlp": "identity",     "residual_mlp": "identity"},
}


def initialize(net, arch: str, init: str, seed: int) -> None:
    """Realise `init` on `arch`, handling non-square first/last layers.

    A plain rectangular identity on a `32 x 784` first layer selects the first 32 pixels of
    an MNIST image — all border, all constant after normalisation — so the network sees no
    signal at all and a ReLU one dies outright. That is a fact about the slice, not about the
    architecture, so for `identity` the non-square edge layers use the orthogonal
    construction instead and only the square inner layers are set to the identity. The
    operator is then an isometry at step 0 in every case, which is what the init is for.
    """
    if init != "identity":
        net.initialize(INIT[init][arch], seed=seed)
        return

    if arch == "residual_mlp":
        net.initialize("identity", seed=seed)            # blocks are square; W=0 gives J=I
        if net.W_in.shape[0] != net.W_in.shape[1] or net.W_out.shape[0] != net.W_out.shape[1]:
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                net.W_in.copy_(haar(*net.W_in.shape, g))
                net.W_out.copy_(haar(*net.W_out.shape, g))
        return

    # ReLU / CReLU: start from the orthogonal realisation (which handles every shape), then
    # set the SQUARE inner layers to the identity version of the same block structure.
    net.initialize(INIT["orthogonal"][arch], seed=seed)
    with torch.no_grad():
        for idx, W in enumerate(net.weights):
            if idx == 0 or idx == net.depth - 1:
                continue                                  # edges keep the orthogonal form
            n_out, n_in = W.shape
            if arch == "crelu_mlp" and n_in == 2 * n_out:
                W.copy_(torch.cat([torch.eye(n_out), -torch.eye(n_out)], dim=1))
            elif arch == "relu_mlp" and n_in == n_out:
                W.copy_(torch.eye(n_out))


def make_task(name: str, width: int, n: int, seed: int):
    if name == "mnist1d":
        t = MNIST1D(n_train=n, n_val=500, n_test=500, seed=seed)
    elif name == "teacher_isotropic":
        t = TeacherStudent(d=width, n=n, target="orthogonal", seed=seed)
    elif name == "teacher_lowrank":
        t = TeacherStudent(d=width, n=n, target="low_rank", rank=max(2, width // 8), seed=seed)
    else:
        raise ValueError(f"unknown task {name!r}")
    return t.to(torch.device("cpu"), torch.float64)


@torch.no_grad()
def spectrum(net, X) -> dict:
    J = net.operator(X)
    J = J if J.shape[0] > 1 else J.expand(X.shape[0], *J.shape[1:])
    out = []
    for b in range(J.shape[0]):
        s = torch.linalg.svdvals(J[b].double()).numpy()
        M = s**2
        pr = float(M.sum() ** 2 / (M**2).sum()) if (M**2).sum() > 1e-300 else 0.0
        nz = s[s > 1e-14]
        out.append((pr, float(np.log(nz.max() / nz.min())) if nz.size > 1 else 0.0,
                    float(s.max()), float(np.exp(np.log(np.clip(s, 1e-300, None)).mean()))))
    a = np.array(out)
    return {"pr": float(a[:, 0].mean()), "separation": float(a[:, 1].mean()),
            "s_max": float(a[:, 2].mean()), "s_geo": float(a[:, 3].mean()),
            "spectrum": [float(v) for v in
                         torch.linalg.svdvals(J[0].double()).numpy()]}


def bias_metrics(dJ: torch.Tensor, G: torch.Tensor, lr: float) -> dict:
    """`Delta J` against the ideal `-lr * G`, in the three numbers that matter."""
    ideal = -lr * G
    a, b = dJ.flatten().double(), ideal.flatten().double()
    na, nb = float(a.norm()), float(b.norm())
    if na == 0 or nb == 0:
        return {"cos": float("nan"), "scale": float("nan"), "sin": float("nan")}
    cos = float((a @ b) / (na * nb))
    return {"cos": cos, "scale": na / nb,
            "sin": float(np.sqrt(max(0.0, 1.0 - cos**2)))}


def ckpt_path(out: str, arch: str, init: str, task_name: str, seed: int) -> Path:
    """One checkpoint per configuration, beside the results file."""
    p = Path(out)
    return p.parent / (p.stem + "_ckpt") / f"{arch}__{init}__{task_name}__s{seed}.pt"


def run(arch: str, init: str, task_name: str, depth: int, width: int, n: int, lr: float,
        max_steps: int, eval_every: int, tol: float, seed: int, probe: int,
        ckpt=None, resume: bool = False) -> dict:
    torch.manual_seed(seed)
    task = make_task(task_name, width, n, seed)
    net = ARCH[arch](d_in=task.d_in, d_out=task.d_out, width=width, depth=depth).double()
    initialize(net, arch, init, seed)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    start = 0
    if resume and ckpt is not None and Path(ckpt).exists():
        state = torch.load(ckpt, weights_only=False)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optim"])
        start = int(state["step"])

    X, Y = task.train_batch(0, n)                      # full batch, fixed
    Xp = X[:probe]
    rec, losses, stop = [], [], "max_steps"

    for step in range(start, max_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = task.loss(task.forward(net, X), Y)
        loss.backward()
        losses.append(float(loss.detach()))

        if step % eval_every == 0:
            row = {"step": step, "loss": losses[-1]} | spectrum(net, Xp)
            # The ideal step is -lr * E_x[dL/dJ(x)] over the WHOLE training set, not the
            # per-sample gradient: a per-sample G is rank one for a regression loss and
            # comparing a full-rank Delta J against it measures the rank gap, not the bias.
            # operator_gradient differentiates internally, so it must not be under no_grad.
            G_full = task.operator_gradient(net, X, Y).detach()
            G_bar = G_full.mean(0, keepdim=True) if G_full.shape[0] > 1 else G_full
            with torch.no_grad():
                J0 = net.operator(Xp).clone()
            opt.step()                                  # the real step
            with torch.no_grad():
                J1 = net.operator(Xp)
                per = [bias_metrics(J1[b] - J0[b], G_bar[0], lr)
                       for b in range(J0.shape[0])]
                row |= {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            rec.append(row)
            if not np.isfinite(row["loss"]) or row["loss"] > 1e10:
                stop = "diverged"
                break
            if len(losses) > 4 * eval_every:            # relative progress over a window
                old = losses[-4 * eval_every]
                if old > 0 and abs(old - losses[-1]) / old < tol:
                    stop = "converged"
                    break
            continue
        opt.step()

    # Save the final state, so a run that hit the step cap -- which at this depth and
    # learning rate is most of them -- can be continued rather than restarted from scratch.
    if ckpt is not None:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(), "optim": opt.state_dict(),
                    "step": rec[-1]["step"], "stop": stop,
                    "config": {"arch": arch, "init": init, "task": task_name,
                               "depth": depth, "width": width, "n": n, "lr": lr,
                               "seed": seed, "probe": probe}}, ckpt)

    return {"arch": arch, "init": init, "task": task_name, "depth": depth, "width": width,
            "lr": lr, "seed": seed, "n": n, "stop": stop, "steps": rec[-1]["step"],
            "loss_final": rec[-1]["loss"], "trace": rec,
            "ckpt": str(ckpt) if ckpt is not None else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", default=list(ARCH))
    ap.add_argument("--inits", nargs="+", default=list(INIT))
    ap.add_argument("--tasks", nargs="+",
                    default=["teacher_isotropic", "teacher_lowrank", "mnist1d"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=200_000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--probe", type=int, default=8)
    ap.add_argument("--out", default="runs/theory/grid.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip configurations already recorded in --out, and warm-start any "
                         "run whose checkpoint exists (so a capped run continues)")
    a = ap.parse_args()

    jobs = [(A, i, t, s) for A in a.archs for i in a.inits for t in a.tasks for s in a.seeds]
    out, t0 = [], time.time()
    prev: dict = {}
    if a.resume and Path(a.out).exists():
        out = json.loads(Path(a.out).read_text())["runs"]
        prev = {(r["arch"], r["init"], r["task"], r["seed"]): r for r in out}
        # Skip a recorded configuration only if it is actually finished: either it stopped
        # on its own, or it already ran at least as far as this invocation asks for.
        # Otherwise re-enter it warm-started from its checkpoint, so raising --max-steps
        # CONTINUES the capped runs instead of silently leaving them where they were.
        done = {k for k, r in prev.items()
                if r["stop"] != "max_steps" or r["steps"] >= a.max_steps}
        print(f"resuming from {a.out}: {len(prev)} recorded, {len(done)} finished, "
              f"{len(prev) - len(done)} to continue, {len(jobs) - len(prev)} new", flush=True)
        out = [r for k, r in prev.items() if k in done]
    else:
        done = set()
    for k, (A, i, t, s) in enumerate(jobs):
        if (A, i, t, s) in done:
            continue
        r = run(A, i, t, a.depth, a.width, a.n, a.lr, a.max_steps, a.eval_every, a.tol, s,
                a.probe, ckpt_path(a.out, A, i, t, s), a.resume)
        old = prev.get((A, i, t, s))
        if old is not None and r["trace"] and r["trace"][0]["step"] > 0:
            # continued run: keep the earlier history so the trace stays contiguous
            r["trace"] = [x for x in old["trace"] if x["step"] < r["trace"][0]["step"]] + r["trace"]
        out.append(r)
        f, l = r["trace"][0], r["trace"][-1]
        print(f"[{k+1}/{len(jobs)}] {A:<13}{i:<11}{t:<18}s={s} {r['stop']:<10}"
              f"steps={r['steps']:<7} loss {f['loss']:.3e}->{l['loss']:.3e}  "
              f"PR {f['pr']:.2f}->{l['pr']:.2f}  sin {f['sin']:.3f}->{l['sin']:.3f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "runs": out}))
    print(f"wrote {a.out}  ({len(out)} runs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
