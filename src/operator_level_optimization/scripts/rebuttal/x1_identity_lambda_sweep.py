"""E1 — Identity-init, lambda>0 sweep with per-layer update mass.

Rebuttal experiment for NGj5 Q2 / AC point 4 / 4jZv Q1: in the identity-init
setting (no spectral collapse, no warm-start advantage), does the ALS-vs-baseline
gap survive lambda>0 (outside the Remark 4.1 trivial regime), and does the
regularizer distribute the update mass across layers?

Setup mirrors the paper's Fig. 4 (d=16, L=128, N=64, identity init, isotropic
orthogonal target, float64) with:
  - ALS-exact at lambda in {0, 1e-6, 1e-4, 1e-2}, recording ||dW_l||_F per layer
    at every step (weights diffed before/after the ALS step).
  - Baselines at per-method tuned learning rates (sweep, best final rel op error).

Outputs under --out_dir: results.json, config.json, fig_e1_convergence.png,
fig_e1_layer_mass.png.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x1_identity_lambda_sweep \
      --out_dir outputs/oplevel_rebuttal/e1_identity_lambda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from operator_level_optimization.scripts.train.deep_linear_compare import (
    DeepLinearModel,
    als_exact_shared_target_step,
    compose_operator,
    mse_grad_target,
    run_method,
)
from operator_level_optimization.scripts.train.variant_loss_curves import (
    _linobj_exact_dual_step,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify
from operator_level_optimization.scripts.utils.plotting import save_fig


def make_task(
    d: int, n: int, seed: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same generator protocol as run_method (orth target) so tasks match."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, d, device=device, dtype=dtype, generator=g)
    q1, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
    q2, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
    p_star = q1 @ q2.T
    y = x @ p_star.T
    return x, y, p_star


def run_solver_with_layer_mass(
    solver: str,  # "als_reverse" | "als_forward" | "linearized"
    depth: int,
    d: int,
    n: int,
    steps: int,
    lr_als: float,
    lam: float,
    n_sweeps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    floor_mse: float = 1e-24,
) -> dict:
    """Projection-solver run from identity init, recording per-layer ||dW||_F per step.

    "als_reverse"/"als_forward": sequential ALS (paper's solver), sweep order gauge.
    "linearized": exact coupled solve of the first-order projection objective
    (App. B.2 dual form) — the minimum-norm realization, distributes mass by
    construction.
    """
    x, y, p_star = make_task(d, n, seed, device, dtype)
    model = DeepLinearModel(depth=depth, d=d).to(device=device, dtype=dtype)
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.copy_(torch.eye(d, device=device, dtype=dtype))

    hist: list[tuple[int, float, float]] = []
    layer_mass: list[list[float]] = []  # layer_mass[t][l] = ||dW_l||_F at step t
    t0 = time.perf_counter()
    with torch.no_grad():
        for t in range(steps + 1):
            weights = [layer.weight.data for layer in model.layers]
            p = compose_operator(weights)
            rel = float((torch.norm(p - p_star) / torch.norm(p_star)).item())
            mse = float(torch.mean((x @ p.T - y) ** 2).item())
            hist.append((t, rel, mse))
            if t == steps:
                break
            if mse < floor_mse and t >= 50:
                break
            grad_p = mse_grad_target(p, x, y)
            w_old = [w.clone() for w in weights]
            if solver == "linearized":
                dW_list = _linobj_exact_dual_step(weights, grad_p, lr_als, lam)
                for layer, dW in zip(model.layers, dW_list):
                    layer.weight.data.sub_(dW)
            else:
                p_tgt = p - lr_als * grad_p
                w_new = als_exact_shared_target_step(
                    weights,
                    p_tgt=p_tgt,
                    lam=lam,
                    n_sweeps=n_sweeps,
                    reverse=(solver == "als_reverse"),
                    warmstart_identity=False,
                )
                for layer, w_k in zip(model.layers, w_new):
                    layer.weight.data.copy_(w_k)
            layer_mass.append(
                [
                    float(torch.norm(layer.weight.data - wo).item())
                    for layer, wo in zip(model.layers, w_old)
                ]
            )
    wall = time.perf_counter() - t0

    mass = np.array(layer_mass) if layer_mass else np.zeros((0, depth))
    # Participation ratio: (sum m^2)^2 / sum m^4 with m = per-layer dW norm.
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = mass**2
        pr = np.where(sq.sum(axis=1) > 0, sq.sum(axis=1) ** 2 / (sq**2).sum(axis=1), 0.0)
    return {
        "solver": solver,
        "lam": lam,
        "history": hist,
        "layer_mass": mass.tolist(),
        "participation_ratio": pr.tolist(),
        "final_rel_err": hist[-1][1],
        "steps_run": hist[-1][0],
        "wall_seconds": wall,
    }


def run_baseline_lr_sweep(
    methods: list[str],
    lrs: list[float],
    depth: int,
    d: int,
    n: int,
    steps: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Per-method lr sweep at identity init; keep run with best final rel op error."""
    out: dict = {}
    for m in methods:
        best = None
        sweep_final: dict[str, float] = {}
        for lr in lrs:
            print(f"[e1 sweep] method={m} lr={lr}", flush=True)
            try:
                res = run_method(
                    method=m,
                    depth=depth,
                    d=d,
                    n=n,
                    steps=steps,
                    lr=lr,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    target_mode="mse_grad",
                    dc_lam=0.0,
                    dc_alt=1,
                    dc_init_mode="plain",
                    dc_init_scale=1.0,
                    spec_every=10_000_000,
                    spec_steps=None,
                    target_kind="orth",
                    init_mode="identity",
                    als_lam=0.0,
                    als_sweeps=1,
                )
            except Exception as exc:  # diverged run (NaN weights break eigh/svd)
                print(f"[e1 sweep] method={m} lr={lr} failed: {exc}", flush=True)
                sweep_final[str(lr)] = float("inf")
                continue
            final_rel = float(res["history"][-1][1])
            sweep_final[str(lr)] = final_rel
            if np.isfinite(final_rel) and (best is None or final_rel < best["final_rel_err"]):
                best = {
                    "method": m,
                    "lr": lr,
                    "history": res["history"],
                    "final_rel_err": final_rel,
                }
        if best is None:
            raise RuntimeError(f"no finite result for method={m}")
        best["sweep_final_rel_err_by_lr"] = sweep_final
        out[m] = best
        print(
            f"[e1 sweep] best {m}: lr={best['lr']} final_rel={best['final_rel_err']:.3e}",
            flush=True,
        )
    return out


_BASELINE_STYLE = {
    "heavyball": ("Heavy Ball", "#1f77b4"),
    "adam": ("Adam", "#ff7f0e"),
    "muon": ("Muon", "#2ca02c"),
    "shampoo": ("Shampoo", "#d62728"),
    "soap": ("SOAP", "#8c564b"),
    "kfac": ("K-FAC", "#e377c2"),
}
_LAM_COLORS = ["#c6b3e0", "#9467bd", "#6a51a3", "#3f007d"]
_LIN_COLORS = ["#a6dcd5", "#4db6ac", "#00796b", "#004d40"]


def _solver_style(run: dict, lam_index: int) -> tuple[str, str, str]:
    """Return (label, color, linestyle) for a projection-solver run."""
    lam = run["lam"]
    if run["solver"] == "linearized":
        return (
            rf"Coupled linearized $\lambda$={lam:g}",
            _LIN_COLORS[lam_index % len(_LIN_COLORS)],
            "-.",
        )
    if run["solver"] == "als_forward":
        return (rf"ALS fwd sweep $\lambda$={lam:g}", "#e6550d", ":")
    return (
        rf"ALS-exact $\lambda$={lam:g}",
        _LAM_COLORS[lam_index % len(_LAM_COLORS)],
        "-",
    )


def plot_convergence(out_dir: Path, solver_runs: list[dict], baselines: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for m, res in baselines.items():
        arr = np.array(res["history"], dtype=float)
        label, color = _BASELINE_STYLE.get(m, (m, None))
        ax.plot(
            arr[:, 0],
            np.clip(arr[:, 1], 1e-18, None),
            label=f"{label} (lr={res['lr']:g})",
            color=color,
            linewidth=1.4,
            linestyle="--",
            alpha=0.85,
        )
    lam_seen: dict[str, int] = {}
    for run in solver_runs:
        i = lam_seen.get(run["solver"], 0)
        lam_seen[run["solver"]] = i + 1
        label, color, ls = _solver_style(run, i)
        arr = np.array(run["history"], dtype=float)
        ax.plot(
            arr[:, 0],
            np.clip(arr[:, 1], 1e-18, None),
            label=label,
            color=color,
            linewidth=2.2,
            linestyle=ls,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"Relative operator error $\|P - P^\star\|_F / \|P^\star\|_F$")
    ax.set_title("Identity init, d=16, L=128 — tuned baseline lrs vs projection solvers")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    save_fig(fig, out_dir / "fig_e1_convergence.png", dpi=180, bbox_inches="tight")


def plot_layer_mass(out_dir: Path, solver_runs: list[dict], depth: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    lam_seen: dict[str, int] = {}
    styles = []
    for run in solver_runs:
        i = lam_seen.get(run["solver"], 0)
        lam_seen[run["solver"]] = i + 1
        styles.append(_solver_style(run, i))
    for run, (label, color, ls) in zip(solver_runs, styles):
        mass = np.array(run["layer_mass"], dtype=float)
        if mass.shape[0] == 0:
            continue
        ax.plot(
            np.arange(1, depth + 1),
            np.clip(mass[0], 1e-20, None),
            label=label,
            color=color,
            linewidth=1.8,
            linestyle=ls,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"Layer index $\ell$")
    ax.set_ylabel(r"$\|\Delta W_\ell\|_F$ at step 1")
    ax.set_title("Per-layer update mass, first projection step")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    for run, (label, color, ls) in zip(solver_runs, styles):
        pr = np.array(run["participation_ratio"], dtype=float)
        if pr.shape[0] == 0:
            continue
        ax.plot(np.arange(pr.shape[0]), pr, label=label, color=color, linewidth=1.8, linestyle=ls)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(depth, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Participation ratio (effective # layers)")
    ax.set_ylim(0, depth * 1.08)
    ax.set_title("Update-mass participation ratio over training")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5)
    save_fig(fig, out_dir / "fig_e1_layer_mass.png", dpi=180, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr_als", type=float, default=1.0)
    ap.add_argument("--als_sweeps", type=int, default=4)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1e-6, 1e-4, 1e-2])
    ap.add_argument(
        "--gauge_lam",
        type=float,
        default=1e-4,
        help="Run one forward-sweep ALS at this lambda (sweep-order gauge demo); "
        "negative value disables.",
    )
    ap.add_argument(
        "--baseline_methods",
        nargs="+",
        default=["heavyball", "adam", "muon", "kfac", "shampoo", "soap"],
    )
    ap.add_argument(
        "--baseline_lrs",
        type=float,
        nargs="+",
        default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4],
    )
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    dtype = torch.float64

    solver_jobs: list[tuple[str, float]] = []
    for lam in args.lams:
        solver_jobs.append(("als_reverse", lam))
    for lam in args.lams:
        if lam > 0:  # linearized solve needs lam>0 for a well-posed dual system
            solver_jobs.append(("linearized", lam))
    # Sweep-order gauge demo at one lambda: forward sweep concentrates at the
    # opposite end of the chain, showing concentration is a solver choice.
    if args.gauge_lam is not None and args.gauge_lam >= 0:
        solver_jobs.append(("als_forward", args.gauge_lam))

    solver_runs = []
    for solver, lam in solver_jobs:
        print(f"[e1 solver] {solver} lam={lam}", flush=True)
        run = run_solver_with_layer_mass(
            solver=solver,
            depth=args.depth,
            d=args.d,
            n=args.n,
            steps=args.steps,
            lr_als=args.lr_als,
            lam=lam,
            n_sweeps=args.als_sweeps,
            seed=args.seed,
            device=device,
            dtype=dtype,
        )
        pr = run["participation_ratio"]
        print(
            f"[e1 solver] {solver} lam={lam} final_rel={run['final_rel_err']:.3e} "
            f"steps={run['steps_run']} PR_step1={pr[0] if pr else float('nan'):.1f} "
            f"wall={run['wall_seconds']:.1f}s",
            flush=True,
        )
        solver_runs.append(run)

    baselines = run_baseline_lr_sweep(
        methods=args.baseline_methods,
        lrs=args.baseline_lrs,
        depth=args.depth,
        d=args.d,
        n=args.n,
        steps=args.steps,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )

    payload = {
        "solver_runs": solver_runs,
        "baselines": baselines,
        "summary": {
            "solver_final_rel_err": {
                f"{r['solver']}_lam{r['lam']:g}": r["final_rel_err"] for r in solver_runs
            },
            "solver_pr_step1": {
                f"{r['solver']}_lam{r['lam']:g}": (
                    r["participation_ratio"][0] if r["participation_ratio"] else None
                )
                for r in solver_runs
            },
            "solver_pr_mean": {
                f"{r['solver']}_lam{r['lam']:g}": (
                    float(np.mean(r["participation_ratio"])) if r["participation_ratio"] else None
                )
                for r in solver_runs
            },
            "baseline_best_final_rel_err": {
                m: {"lr": res["lr"], "final_rel_err": res["final_rel_err"]}
                for m, res in baselines.items()
            },
        },
    }
    (out_dir / "results.json").write_text(json.dumps(jsonify(payload), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    plot_convergence(out_dir, solver_runs, baselines)
    plot_layer_mass(out_dir, solver_runs, args.depth)
    print(f"[done] wrote {out_dir}", flush=True)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
