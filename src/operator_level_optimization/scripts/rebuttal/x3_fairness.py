"""E3 — Fairness: per-step wall-clock, memory, tuned lrs, matched budget.

Rebuttal experiment for NGj5 Q1 / AC point 5.

Setup A (cost table): per-step wall-clock (median over timed steps after
warmup) and peak CUDA memory for every method at (d, L) in
{(16,32), (64,32), (256,32), (16,128)}, on cpu and (if available) cuda.

Setup B (matched wall-clock budget): paper's headline setting (Xavier init,
d=16, L=128, N=64, orthogonal target, float64). ALS-exact runs to its
numerical floor and defines the budget T_als. Every baseline is lr-tuned
(short runs with early stopping), then run at its best lr for up to
--budget_factor x T_als wall-clock (capped at --max_steps). The figure shows
relative operator error vs wall-clock seconds; stalled-vs-slow is read off
directly.

Run from repo root (alone — timing-sensitive, don't run concurrently with
other experiments):
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x3_fairness \
      --out_dir outputs/oplevel_rebuttal/e3_fairness
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from operator_level_optimization.core.optim.kfac import KFAC
from operator_level_optimization.core.optim.muon import Muon
from operator_level_optimization.core.optim.shampoo import Shampoo
from operator_level_optimization.core.optim.soap import SOAP
from operator_level_optimization.scripts.train.deep_linear_compare import (
    DeepLinearModel,
    als_exact_shared_target_step,
    compose_operator,
    ginibre_sn1,
    mse_grad_target,
    xavier_gaussian,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify
from operator_level_optimization.scripts.utils.plotting import save_fig

BASELINES = ["heavyball", "adam", "muon", "kfac", "shampoo", "soap"]


def build_optimizer(method: str, model: DeepLinearModel, lr: float):
    params = list(model.parameters())
    if method == "heavyball":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if method == "adam":
        return torch.optim.Adam(params, lr=lr)
    if method == "muon":
        return Muon(params, lr=lr)
    if method == "shampoo":
        return Shampoo(params, lr=lr, beta=0.9, momentum=0.0, weight_decay=0.0, eps=1e-4)
    if method == "soap":
        return SOAP(params, lr=lr, weight_decay=0.0, correct_bias=True)
    if method == "kfac":
        opt = KFAC(
            params, lr=lr, factor_decay=0.95, damping=0.1,
            momentum=0.0, weight_decay=0.0, inv_floor=1e-2,
        )
        opt.attach_hooks(model)
        return opt
    raise ValueError(method)


def make_task_xavier(
    depth: int, d: int, n: int, seed: int, device: torch.device, dtype: torch.dtype
) -> tuple[DeepLinearModel, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same generation protocol as run_method (orth target, xavier init)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, d, device=device, dtype=dtype, generator=g)
    q1, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
    q2, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
    p_star = q1 @ q2.T
    model = DeepLinearModel(depth=depth, d=d).to(device=device, dtype=dtype)
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.copy_(xavier_gaussian((d, d), device=device, dtype=dtype, g=g))
    return model, x, p_star, x @ p_star.T


# ---------------------------------------------------------------- Setup A ----


def time_per_step(
    method: str,
    depth: int,
    d: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    als_lam: float,
    als_sweeps: int,
    n_warmup: int = 5,
    n_timed: int = 30,
    seed: int = 0,
) -> dict:
    model, x, p_star, y = make_task_xavier(depth, d, n, seed, device, dtype)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        base_mem = torch.cuda.memory_allocated(device)

    if method == "als_exact":
        def one_step() -> None:
            with torch.no_grad():
                weights = [layer.weight.data for layer in model.layers]
                p = compose_operator(weights)
                p_tgt = p - 1.0 * mse_grad_target(p, x, y)
                w_new = als_exact_shared_target_step(
                    weights, p_tgt=p_tgt, lam=als_lam, n_sweeps=als_sweeps,
                    reverse=True, warmstart_identity=True,
                )
                for layer, w_k in zip(model.layers, w_new):
                    layer.weight.data.copy_(w_k)
    else:
        optim = build_optimizer(method, model, lr=1e-4)  # lr irrelevant for timing

        def one_step() -> None:
            optim.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            optim.step()

    for _ in range(n_warmup):
        one_step()
    times = []
    for _ in range(n_timed):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        one_step()
        if is_cuda:
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    out = {
        "method": method,
        "depth": depth,
        "d": d,
        "median_step_ms": float(np.median(times) * 1e3),
        "p90_step_ms": float(np.percentile(times, 90) * 1e3),
    }
    if is_cuda:
        out["peak_mem_mb"] = float(
            (torch.cuda.max_memory_allocated(device) - base_mem) / 2**20
        )
    return out


# ---------------------------------------------------------------- Setup B ----


def budget_run(
    method: str,
    lr: float,
    depth: int,
    d: int,
    n: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    max_steps: int,
    max_seconds: float,
    als_lam: float = 1e-4,
    als_sweeps: int = 4,
    record_every: int = 1,
    floor_rel: float = 1e-10,
    stall_patience: int | None = None,
    stall_rel_tol: float = 1e-4,
) -> dict:
    """Run one method with timestamps; stop on budget, floor, steps, or stall."""
    model, x, p_star, y = make_task_xavier(depth, d, n, seed, device, dtype)
    optim = None if method == "als_exact" else build_optimizer(method, model, lr)
    hist: list[tuple[int, float, float]] = []  # (step, rel_err, wall_seconds)
    best_rel = float("inf")
    stall = 0
    stop_reason = "max_steps"
    t0 = time.perf_counter()
    for t in range(max_steps + 1):
        with torch.no_grad():
            p = compose_operator([layer.weight.data for layer in model.layers])
            if not bool(torch.isfinite(p).all()):
                stop_reason = "diverged"
                break
            rel = float((torch.norm(p - p_star) / torch.norm(p_star)).item())
        el = time.perf_counter() - t0
        if t % record_every == 0 or t == max_steps:
            hist.append((t, rel, el))
        if rel < best_rel * (1.0 - stall_rel_tol):
            best_rel = rel
            stall = 0
        else:
            stall += 1
        if rel < floor_rel:
            stop_reason = "floor"
            break
        if el > max_seconds:
            stop_reason = "budget"
            break
        if stall_patience is not None and stall > stall_patience and t > 200:
            stop_reason = "stall"
            break
        if t == max_steps:
            break
        try:
            if method == "als_exact":
                with torch.no_grad():
                    weights = [layer.weight.data for layer in model.layers]
                    p_tgt = p - 1.0 * mse_grad_target(p, x, y)
                    # Warm-start ON: this is the paper's headline method
                    # ("ALS + warm-start") for the Xavier/collapse setting —
                    # without it ALS inherits the collapse and stalls at
                    # ~6e-1, exactly as Sec. 4 predicts.
                    w_new = als_exact_shared_target_step(
                        weights, p_tgt=p_tgt, lam=als_lam, n_sweeps=als_sweeps,
                        reverse=True, warmstart_identity=True,
                    )
                    for layer, w_k in zip(model.layers, w_new):
                        layer.weight.data.copy_(w_k)
            else:
                optim.zero_grad(set_to_none=True)
                loss = torch.mean((model(x) - y) ** 2)
                loss.backward()
                optim.step()
        except Exception as exc:
            print(f"[e3] {method} lr={lr} failed at t={t}: {exc}", flush=True)
            stop_reason = "error"
            break
    best_seen = min((h[1] for h in hist), default=float("inf"))
    return {
        "method": method,
        "lr": lr,
        "history": hist,
        "final_rel_err": hist[-1][1] if hist else float("inf"),
        "best_rel_err": best_seen,
        "wall_seconds": hist[-1][2] if hist else 0.0,
        "steps_run": hist[-1][0] if hist else 0,
        "stop_reason": stop_reason,
    }


_METHOD_COLORS = {
    "heavyball": "#1f77b4",
    "adam": "#ff7f0e",
    "muon": "#2ca02c",
    "shampoo": "#d62728",
    "soap": "#8c564b",
    "kfac": "#e377c2",
    "als_exact": "#9467bd",
}


def plot_budget(out_dir: Path, runs: dict, t_als: float, budget_factor: float) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for m, res in runs.items():
        arr = np.array(res["history"], dtype=float)
        if arr.shape[0] == 0:
            continue
        lbl = "ALS-exact" if m == "als_exact" else f"{m} (lr={res['lr']:g})"
        ax.plot(
            np.clip(arr[:, 2], 1e-3, None),
            np.clip(arr[:, 1], 1e-14, None),
            label=lbl,
            color=_METHOD_COLORS.get(m),
            linewidth=2.0 if m == "als_exact" else 1.4,
        )
    ax.axvline(t_als, color="gray", linestyle=":", linewidth=1.0)
    ax.axvline(t_als * budget_factor, color="gray", linestyle="--", linewidth=1.0)
    # Blended transform (x = data, y = axes fraction): placing the labels at a
    # data y-coordinate on a log axis can explode the tight bounding box.
    import matplotlib.transforms as mtransforms

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(t_als, 0.02, r" $T_{ALS}$", fontsize=8, va="bottom", transform=trans)
    ax.text(
        t_als * budget_factor, 0.02,
        rf" ${budget_factor:g}\times T_{{ALS}}$", fontsize=8, va="bottom",
        transform=trans,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wall-clock seconds")
    ax.set_ylabel(r"Relative operator error $\|P - P^\star\|_F/\|P^\star\|_F$")
    ax.set_title("Xavier init, d=16, L=128 — tuned lrs, matched wall-clock")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, out_dir / "fig_e3_budget.png", dpi=180, bbox_inches="tight")


def write_cost_table(out_dir: Path, cost_rows: list[dict]) -> None:
    lines = [
        "# E3 cost table (median per-step wall-clock; peak CUDA memory)",
        "",
        "| method | device | d | L | ms/step | peak MB |",
        "|---|---|---|---|---|---|",
    ]
    for r in cost_rows:
        mem = f"{r['peak_mem_mb']:.1f}" if "peak_mem_mb" in r else "-"
        lines.append(
            f"| {r['method']} | {r['device']} | {r['d']} | {r['depth']} "
            f"| {r['median_step_ms']:.2f} | {mem} |"
        )
    (out_dir / "table_e3_cost.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--als_lam", type=float, default=1e-4)
    ap.add_argument("--als_sweeps", type=int, default=4)
    ap.add_argument(
        "--cost_configs",
        type=str,
        default="16x32,64x32,256x32,16x128",
        help="Comma list of dxL for the cost table.",
    )
    ap.add_argument("--cost_devices", nargs="+", default=["cpu", "cuda"])
    ap.add_argument("--budget_depth", type=int, default=128)
    ap.add_argument("--budget_d", type=int, default=16)
    ap.add_argument("--budget_factor", type=float, default=10.0)
    ap.add_argument("--tune_lrs", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    ap.add_argument("--tune_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--budget_device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--skip_cost", action="store_true")
    ap.add_argument("--skip_budget", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64

    # ---- Setup A: cost table
    cost_rows: list[dict] = []
    if not args.skip_cost:
        configs = [
            (int(s.split("x")[0]), int(s.split("x")[1]))
            for s in args.cost_configs.split(",")
        ]
        for dev_name in args.cost_devices:
            if dev_name == "cuda" and not torch.cuda.is_available():
                continue
            device = torch.device(dev_name)
            for d, depth in configs:
                for method in BASELINES + ["als_exact"]:
                    row = time_per_step(
                        method, depth, d, args.n, device, dtype,
                        args.als_lam, args.als_sweeps, seed=args.seed,
                    )
                    row["device"] = dev_name
                    cost_rows.append(row)
                    mem = row.get("peak_mem_mb")
                    print(
                        f"[e3 cost] {dev_name} d={d} L={depth} {method}: "
                        f"{row['median_step_ms']:.2f} ms/step"
                        + (f", {mem:.1f} MB" if mem is not None else ""),
                        flush=True,
                    )
        write_cost_table(out_dir, cost_rows)

    # ---- Setup B: matched wall-clock budget
    budget_payload: dict = {}
    if not args.skip_budget:
        device = torch.device(args.budget_device)
        print("[e3 budget] ALS reference run", flush=True)
        als_run = budget_run(
            "als_exact", 1.0, args.budget_depth, args.budget_d, args.n,
            args.seed, device, dtype,
            max_steps=args.max_steps, max_seconds=float("inf"),
            als_lam=args.als_lam, als_sweeps=args.als_sweeps,
            # ALS plateaus near its numerical floor (~2e-10 in the paper's
            # Fig. 2) which can sit above floor_rel — stop on stall instead of
            # running out the full step budget at ~1 s/step.
            stall_patience=300,
        )
        t_als = als_run["wall_seconds"]
        print(
            f"[e3 budget] ALS: rel={als_run['final_rel_err']:.3e} "
            f"steps={als_run['steps_run']} T_als={t_als:.1f}s "
            f"({als_run['stop_reason']})",
            flush=True,
        )

        runs = {"als_exact": als_run}
        tuning: dict = {}
        for m in BASELINES:
            best_lr, best_val = None, float("inf")
            tuning[m] = {}
            for lr in args.tune_lrs:
                res = budget_run(
                    m, lr, args.budget_depth, args.budget_d, args.n,
                    args.seed, device, dtype,
                    max_steps=args.tune_steps, max_seconds=float("inf"),
                )
                tuning[m][str(lr)] = res["best_rel_err"]
                print(
                    f"[e3 tune] {m} lr={lr}: best_rel={res['best_rel_err']:.3e} "
                    f"({res['stop_reason']})",
                    flush=True,
                )
                if np.isfinite(res["best_rel_err"]) and res["best_rel_err"] < best_val:
                    best_val, best_lr = res["best_rel_err"], lr
            print(f"[e3 budget] {m} best lr={best_lr}", flush=True)
            runs[m] = budget_run(
                m, best_lr, args.budget_depth, args.budget_d, args.n,
                args.seed, device, dtype,
                max_steps=args.max_steps,
                max_seconds=t_als * args.budget_factor,
                stall_patience=2000,
            )
            r = runs[m]
            print(
                f"[e3 budget] {m}: best_rel={r['best_rel_err']:.3e} "
                f"steps={r['steps_run']} wall={r['wall_seconds']:.1f}s "
                f"({r['stop_reason']})",
                flush=True,
            )
        plot_budget(out_dir, runs, t_als, args.budget_factor)
        budget_payload = {
            "t_als_seconds": t_als,
            "runs": runs,
            "lr_tuning_best_rel_err": tuning,
            "summary": {
                m: {
                    "lr": r["lr"],
                    "best_rel_err": r["best_rel_err"],
                    "wall_seconds": r["wall_seconds"],
                    "steps_run": r["steps_run"],
                    "stop_reason": r["stop_reason"],
                }
                for m, r in runs.items()
            },
        }

    payload = {"cost_table": cost_rows, "budget": budget_payload}
    (out_dir / "results.json").write_text(json.dumps(jsonify(payload), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
