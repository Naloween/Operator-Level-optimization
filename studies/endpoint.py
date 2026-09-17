"""Do the two arms reach DIFFERENT endpoints, and when?

Check §4.3 of `theory/13-paper-plan.md`. Two arms, same architecture, same data, same starting
operator:

    arm "gd"  : coordinate gradient descent on the factors W_1..W_L  (the biased arm)
    arm "op"  : the counterfactual, J <- J - eta * G                 (the unbiased arm)

For a deep linear network of width >= min(d_in, d_out) the reachable set of operators is
unconstrained, so arm "op" is exactly what the certified min-norm solver would realise -- we can
integrate J directly and skip the solver entirely. That is the point of running this check first:
it needs no implementation of the method to answer whether the method would show anything.

**The endpoints can only differ where the minimiser is not unique.** With isotropic inputs and a
full-rank teacher the global minimum is unique (J = A), so both arms must agree and the only
difference is the path. The four setups below are ordered by how underdetermined they are:

    aniso     full-rank, ill-conditioned inputs  -> unique minimiser  -> CONTROL, expect same
    underdet  n < d samples                      -> affine solution set
    sensing   m < d^2 random measurements        -> affine solution set, low-rank truth
    deep      sensing at depth with Xavier init  -> separation by optimisation failure

Writes `runs/theory/endpoint.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- tasks


def make_task(name: str, d: int, rng: np.random.Generator) -> dict:
    """Each task returns loss(J) and grad(J), plus whatever the endpoint should be compared to."""
    if name == "aniso":
        # Full-rank ill-conditioned input covariance. Unique minimiser J = A. CONTROL.
        n = 8 * d
        cond = 30.0   # the slowest eigendirection converges ~cond^2 slower; keep it reachable
        scales = np.logspace(0, -np.log10(cond), d)
        X = (rng.standard_normal((d, n)) * scales[:, None])
        A = rng.standard_normal((d, d)) / np.sqrt(d)
        Y = A @ X
        S = X @ X.T / n
        C = Y @ X.T / n

        def loss(J):
            R = J @ X - Y
            return 0.5 * float(np.sum(R * R)) / n

        def grad(J):
            return (J @ S - C)

        return {"loss": loss, "grad": grad, "truth": A, "kind": "unique",
                "info": f"n={n} cond={cond:g}"}

    if name == "underdet":
        # Fewer samples than dimensions: solution set {J : JX = Y} is affine of dim d*(d-n).
        n = d // 2
        X = rng.standard_normal((d, n))
        A = rng.standard_normal((d, d)) / np.sqrt(d)
        Y = A @ X
        minnorm = Y @ np.linalg.pinv(X)  # the limit of the unbiased arm from ~0

        def loss(J):
            R = J @ X - Y
            return 0.5 * float(np.sum(R * R)) / n

        def grad(J):
            return (J @ X - Y) @ X.T / n

        return {"loss": loss, "grad": grad, "truth": A, "minnorm": minnorm,
                "kind": "affine", "info": f"n={n} < d={d}"}

    if name in ("sensing", "deep"):
        # Matrix sensing with a low-rank ground truth: the canonical implicit-bias test-bed.
        r = 2
        U = np.linalg.qr(rng.standard_normal((d, r)))[0]
        V = np.linalg.qr(rng.standard_normal((d, r)))[0]
        Jstar = U @ np.diag(np.linspace(1.0, 0.6, r)) @ V.T
        m = 3 * r * (2 * d - r)                      # enough for recovery, far below d^2
        Am = rng.standard_normal((m, d, d)) / np.sqrt(d)
        b = np.einsum("kij,ij->k", Am, Jstar)
        # least-norm solution of the affine system, = limit of the unbiased arm from 0
        Aflat = np.ascontiguousarray(Am.reshape(m, -1))
        minnorm = np.linalg.lstsq(Aflat, b, rcond=None)[0].reshape(d, d)

        def loss(J):
            res = Aflat @ J.ravel() - b
            return 0.5 * float(res @ res) / m

        def grad(J):
            res = Aflat @ J.ravel() - b
            return (Aflat.T @ res).reshape(d, d) / m

        return {"loss": loss, "grad": grad, "truth": Jstar, "minnorm": minnorm,
                "kind": "affine", "info": f"m={m} of d^2={d*d}, rank(J*)={r}"}

    raise ValueError(name)


# --------------------------------------------------------------------------- arms


def init_factors(d: int, L: int, scale: float, init: str,
                 rng: np.random.Generator) -> list[np.ndarray]:
    """Balanced small init (orthogonal, scaled) or Xavier. Returns W_1..W_L."""
    if init == "orth":
        eps = scale ** (1.0 / L)
        return [eps * np.linalg.qr(rng.standard_normal((d, d)))[0] for _ in range(L)]
    if init == "xavier":
        return [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(L)]
    raise ValueError(init)


def product(Ws: list[np.ndarray]) -> np.ndarray:
    J = Ws[0]
    for W in Ws[1:]:
        J = W @ J
    return J


def _flow(state, unpack, step_fn, loss_fn, gnorm_fn, max_steps, rel_move, gtol):
    """Integrate gradient FLOW with an adaptive step.

    Why the flow and not fixed-lr GD: the trajectory of the flow is step-size independent, so the
    two arms are comparable without per-arm lr tuning, and it is the object Arora's Thm 1 is
    stated for. Adaptive dt is pure discretisation control -- a rejected step is retried smaller,
    which does not move the trajectory, unlike a line search. Plateaus are crossed quickly
    because dt grows when the velocity is small, which is exactly the saddle-to-saddle
    pathology that made the fixed-lr version stall.
    """
    g0 = gnorm_fn(state)
    dt, t_tot, nrej = 1e-6, 0.0, 0
    for step in range(max_steps):
        v = step_fn(state)                       # the negative gradient, as a state-shaped object
        vn, sn = unpack(v), unpack(state)
        if not np.isfinite(vn) or not np.isfinite(sn):
            return state, {"stop": "diverged", "steps": step, "time": t_tot, "rejects": nrej}
        gn = gnorm_fn(state)
        if gn <= gtol * max(g0, 1e-300):
            return state, {"stop": "converged", "steps": step, "time": t_tot, "rejects": nrej}
        if vn > 0:
            dt = min(dt * 2.0, rel_move * max(sn, 1e-12) / vn)
        cur = loss_fn(state)
        for _ in range(60):                      # reject-and-halve until the step decreases loss
            trial = [s + dt * d for s, d in zip(state, v)]
            nxt = loss_fn(trial)
            if np.isfinite(nxt) and nxt <= cur:
                break
            dt *= 0.5
            nrej += 1
        else:
            return state, {"stop": "stuck", "steps": step, "time": t_tot, "rejects": nrej}
        state, t_tot = trial, t_tot + dt
    return state, {"stop": "max_steps", "steps": max_steps, "time": t_tot, "rejects": nrej}


def _nrm(xs) -> float:
    return float(np.sqrt(sum(float(np.sum(x * x)) for x in xs)))


def run_gd(Ws, task, steps, rel_move, gtol):
    """Coordinate gradient flow on the factors. dL/dW_l = A_l^T G B_l^T."""
    L = len(Ws)

    def grads(state):
        J = product(state)
        G = task["grad"](J)
        B = [np.eye(state[0].shape[1])]
        for l in range(L - 1):
            B.append(state[l] @ B[-1])
        A = [np.eye(state[-1].shape[0])]
        for l in range(L - 1, 0, -1):
            A.append(A[-1] @ state[l])
        A = A[::-1]                              # A[l] sits to the left of W_l
        return [A[l].T @ G @ B[l].T for l in range(L)]

    out, st = _flow([W.copy() for W in Ws], _nrm,
                    lambda s: [-g for g in grads(s)],
                    lambda s: task["loss"](product(s)),
                    lambda s: _nrm(grads(s)), steps, rel_move, gtol)
    return product(out), st


def run_op(J0, task, steps, rel_move, gtol):
    """The counterfactual: the operator itself follows the unbiased flow."""
    out, st = _flow([J0.copy()], _nrm,
                    lambda s: [-task["grad"](s[0])],
                    lambda s: task["loss"](s[0]),
                    lambda s: float(np.linalg.norm(task["grad"](s[0]))), steps, rel_move, gtol)
    return out[0], st


# --------------------------------------------------------------------------- metrics


def spectrum_stats(J: np.ndarray) -> dict:
    if not np.all(np.isfinite(J)):
        return {"nuclear": float("nan"), "fro": float("nan"), "stable_rank": float("nan"),
                "pr": float("nan"), "top": []}
    s = np.linalg.svd(J, compute_uv=False)
    tot, sq = float(s.sum()), float((s * s).sum())
    return {
        "nuclear": tot,
        "fro": float(np.sqrt(sq)),
        "stable_rank": (sq / float(s[0] ** 2)) if s[0] > 0 else 0.0,
        "pr": (tot * tot / sq) if sq > 0 else 0.0,     # participation ratio of singular values
        "top": [float(v) for v in s[:5]],
    }


def rel(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("nan")
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float("nan")


def stable(fn, lr, tries=12):
    """Largest lr in the halving sequence that does not diverge.

    Restarting from the initialisation with a smaller step keeps each reported trajectory a
    genuine fixed-lr GD trajectory, which matters: a line search would change the implicit bias
    we are trying to measure.
    """
    for _ in range(tries):
        J, st = fn(lr)
        if st["stop"] != "diverged" and np.all(np.isfinite(J)):
            return J, {**st, "lr": lr}
        lr *= 0.5
    return J, {**st, "lr": lr}


def cell(task_name, d, L, init, scale, steps, rel_move, gtol, seed) -> dict:
    rng = np.random.default_rng(seed)
    task = make_task(task_name, d, rng)
    Ws = init_factors(d, L, scale, init, rng)
    J0 = product(Ws)

    J_gd, s_gd = run_gd(Ws, task, steps, rel_move, gtol)
    J_op, s_op = run_op(J0, task, steps, rel_move, gtol)

    out = {
        "task": task_name, "info": task["info"], "kind": task["kind"],
        "d": d, "L": L, "init": init, "seed": seed,
        "J0_norm": float(np.linalg.norm(J0)),
        "gd": {"loss": task["loss"](J_gd), **s_gd, **spectrum_stats(J_gd)},
        "op": {"loss": task["loss"](J_op), **s_op, **spectrum_stats(J_op)},
        "endpoint_gap": rel(J_gd, J_op),
    }
    for key in ("truth", "minnorm"):
        if key in task:
            out["gd"][f"err_{key}"] = rel(J_gd, task[key])
            out["op"][f"err_{key}"] = rel(J_op, task[key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+",
                    default=["aniso", "underdet", "sensing", "deep"])
    ap.add_argument("--d", type=int, default=20)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--scale", type=float, default=1e-2, help="||J|| at initialisation")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--rel-move", type=float, default=2e-3,
                    help="max relative change of the state per integration step")
    ap.add_argument("--gtol", type=float, default=1e-11,
                    help="stop when ||grad|| falls to this fraction of its initial value")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="runs/theory/endpoint.json")
    a = ap.parse_args()

    jobs = []
    for t in a.tasks:
        # "deep" is the ill-conditioned separation case: Xavier init, larger depth
        depths = [8] if t == "deep" else a.depths
        init = "xavier" if t == "deep" else "orth"
        for L in depths:
            for s in a.seeds:
                jobs.append((t, L, init, s))

    rows, t0 = [], time.time()
    for i, (t, L, init, s) in enumerate(jobs):
        r = cell(t, a.d, L, init, a.scale, a.steps, a.rel_move, a.gtol, s)
        rows.append(r)
        print(f"[{i+1}/{len(jobs)}] {t:<9} L={L} {init:<7} s={s}  "
              f"gap={r['endpoint_gap']:.3e}  "
              f"gd(loss={r['gd']['loss']:.1e},sr={r['gd']['stable_rank']:.2f},{r['gd']['stop'][:4]})  "
              f"op(loss={r['op']['loss']:.1e},sr={r['op']['stable_rank']:.2f},{r['op']['stop'][:4]})  "
              f"({time.time()-t0:.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "runs": rows}, indent=1))
    print(f"wrote {a.out}  ({len(rows)} cells, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
