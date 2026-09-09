"""Is there a low-rank bias at a fixed gate pattern? And for which patterns?

**Naming.** A *pattern* is a gate pattern `eps` -- one sign vector per layer, not necessarily
realized by any input. A *direction* is a singular index `k` of an operator. Earlier files
called both "mode"; they are unrelated and only the first is CReLU-specific.

**Why patterns are the right object.** `J_eps = W_L D(eps_{L-1}) ... D(eps_1) W_1` is a
polynomial in the weights with no discontinuity, so `t -> J_eps(t)` is smooth where
`t -> J(x)` jumps at region boundaries. And the weight gradient `Gamma_l = dL/dW_l` is *one
matrix shared by every pattern* -- the cross-input coupling that `theory/03` isolated is
already inside it. So the dynamics of every pattern is a fixed-gates linear network driven
by the same `Gamma`, and the patterns differ only in how they compose it:

    sdot_k(eps) = - sum_l u_k^T A_l^eps Gamma_l B_l^eps v_k          (exact)

**The measurement.** The bias rate is `b = d log|omega| / d log s` across directions, with
`omega_k = sdot_k / s_k` (Lemma 12: `b > 0` means every separation grows). Writing
`sdot_k = -c_k g_k` with `c_k` the pattern's own mode gain -- a definition, not a hypothesis --
splits it:

    b = d log c / d log s  -  1  +  d log|g| / d log s
        [geometry of the pattern]      [what the gradient asks of it]

The first term is pure geometry of `J_eps`'s factorization; the second is the drive. Both
are computed exactly.

**The axis.** Patterns are swept from realized (`frac = 0`) to Rademacher (`frac = 1`) by
flipping a fraction of the signs. `theory/03` measured that realized patterns are *more*
separated than typical ones (a factor 2.2 at `L = 64`) but said nothing about their
dynamics. This asks whether the bias itself is a property of the realized patterns or of
all of them.

Writes `runs/theory/patterns.json`.
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
from olo.theory.instability import amplification
from olo.theory.modes import hamming_mix, pattern_dynamics, random_modes, realized_modes


@torch.no_grad()
def nonlinearity(net) -> float:
    """`max_l ||Delta_l|| / ||S_l||`: how much the patterns can differ at all."""
    return max(float(split_layer(W)[1].norm() / split_layer(W)[0].norm().clamp_min(1e-300))
               for W in net.weights[1:])


def make_task(name: str, width: int, n: int, seed: int):
    t = (MNIST(n_train=n, n_val=128, n_test=128, seed=seed) if name == "mnist"
         else TeacherStudent(d=width, n=n, target="orthogonal", seed=seed))
    return t.to(torch.device("cpu"), torch.float64)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    v = float(xc @ xc)
    return float(xc @ (y - y.mean()) / v) if v > 1e-12 else float("nan")


def score(net, eps, grads, floor=1e-9) -> dict:
    """Bias rate at one pattern, plus its geometry/drive split."""
    d = pattern_dynamics(net, eps, grads)
    s, sdot, c, g = d["s"], d["sdot"], d["gain"], d["drive"]
    m = (s > floor) & (c > 0) & (np.abs(g) > 0)
    if int(m.sum()) < 4:
        return {"n": int(m.sum())}
    ls = np.log(s[m])
    amp = amplification(s[m], sdot[m])
    return {
        "n": int(m.sum()),
        "rate": amp.rate,                       # b: >0 means separation grows
        "r2": amp.r2,
        "uniform": amp.uniform,
        "gain_exponent": _slope(ls, np.log(c[m])),      # d log c / d log s
        "drive_exponent": _slope(ls, np.log(np.abs(g[m]))),
        "separation": float(ls.max() - ls.min()),
        "cond": float(s[m].max() / s[m].min()),
    }


def run(task_name: str, init: str, depth: int, width: int, steps: int, lr: float,
        batch: int, eval_every: int, seed: int, n_patterns: int, fracs) -> dict:
    torch.manual_seed(seed)
    task = make_task(task_name, width, max(512, batch), seed)
    net = CReLUMLP(d_in=task.d_in, d_out=task.d_out, width=width, depth=depth).double()
    net.initialize(init, seed=seed)
    opt = torch.optim.SGD(net.parameters(), lr=lr / depth)
    gen = torch.Generator().manual_seed(seed + 3)

    rec = []
    for step in range(steps + 1):
        X, Y = task.train_batch(step, batch)
        opt.zero_grad(set_to_none=True)
        loss = task.loss(task.forward(net, X), Y)
        loss.backward()
        grads = [W.grad.detach().clone() for W in net.weights]

        if step % eval_every == 0:
            real = realized_modes(net, X[:n_patterns])
            rand = random_modes(net, n_patterns, gen)
            # How far apart the patterns actually are. Without this, "realized behaves
            # like random" is vacuous whenever Delta is small: every pattern then gives
            # nearly the same operator and there is nothing for the comparison to see.
            spread = []
            for i in range(n_patterns):
                spread.append(np.log(np.clip(
                    pattern_dynamics(net, rand[i], grads)["s"], 1e-300, None)))
            spread = float(np.mean(np.std(np.array(spread), axis=0)))
            row = {"step": step, "loss": float(loss.detach()),
                   "pattern_spread": spread, "delta": nonlinearity(net), "by_frac": {}}
            for frac in fracs:
                out = []
                for i in range(n_patterns):
                    eps = (real[i] if frac == 0.0 else
                           rand[i] if frac == 1.0 else
                           hamming_mix(real[i], rand[i], frac, gen))
                    sc = score(net, eps, grads)
                    if sc.get("n", 0) >= 4:
                        out.append(sc)
                if out:
                    row["by_frac"][str(frac)] = {
                        k: float(np.median([o[k] for o in out]))
                        for k in ("rate", "r2", "gain_exponent", "drive_exponent",
                                  "separation", "uniform")
                    } | {"frac_positive": float(np.mean([o["rate"] > 0 for o in out])),
                         "n": len(out)}
            rec.append(row)
            if not np.isfinite(row["loss"]) or row["loss"] > 1e10:
                break
        if step == steps:
            break
        opt.step()

    return {"task": task_name, "init": init, "depth": depth, "width": width,
            "seed": seed, "lr": lr / depth, "trace": rec}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+", default=["teacher_student", "mnist"])
    ap.add_argument("--inits", nargs="+", default=["looks_linear", "xavier"])
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n-patterns", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="runs/theory/patterns.json")
    a = ap.parse_args()

    jobs = [(t, i, d, s) for t in a.tasks for i in a.inits for d in a.depths
            for s in a.seeds]
    out, t0 = [], time.time()
    for n, (t, i, d, s) in enumerate(jobs):
        rec = run(t, i, d, a.width, a.steps, a.lr, a.batch, a.eval_every, s,
                  a.n_patterns, a.fracs)
        out.append(rec)
        last = rec["trace"][-1]["by_frac"]
        real, rand = last.get("0.0", {}), last.get("1.0", {})
        print(f"[{n+1}/{len(jobs)}] {t:<16}{i:<13}L={d:<4}s={s}  "
              f"delta={rec['trace'][-1]['delta']:.2e} "
              f"spread={rec['trace'][-1]['pattern_spread']:.2e}  "
              f"realized b={real.get('rate', float('nan')):+.3f} "
              f"({real.get('frac_positive', float('nan')):.0%}>0)  "
              f"random b={rand.get('rate', float('nan')):+.3f} "
              f"({rand.get('frac_positive', float('nan')):.0%}>0)  "
              f"({time.time()-t0:.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "runs": out}))
    print(f"wrote {a.out}  ({len(out)} runs, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
