"""X12 — depth-aware learning-rate scaling and warmup as additional baselines.

AC point 5 / NGj5 W5: do depth-aware lr schemes or warmup change the picture?
Schedules, applied to Adam and heavy-ball at d=16, L=128 under both inits:

  none        — constant base lr (control; matches E1/E5/E8 protocol)
  warmup200   — linear warmup 0 -> base lr over 200 steps, then constant
  depth_sqrt  — constant base lr / sqrt(L)
  depth_lin   — constant base lr / L
  warmup_ds   — warmup200 combined with depth_sqrt

Note the depth-scaled variants are constant-lr points already inside the E8
grid; they are run here under their standard names so the revision can cite
them as explicit baselines. Warmup is genuinely time-varying.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x12_lr_schedules \
      --out_dir outputs/oplevel_rebuttal/e12_lr_schedules
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from operator_level_optimization.scripts.rebuttal.x3_fairness import (
    build_optimizer,
    make_task_xavier,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify


def make_task(
    init: str, depth: int, d: int, n: int, seed: int,
    device: torch.device, dtype: torch.dtype,
):
    model, x, p_star, y = make_task_xavier(depth, d, n, seed, device, dtype)
    if init == "identity":
        with torch.no_grad():
            for layer in model.layers:
                layer.weight.copy_(torch.eye(d, device=device, dtype=dtype))
    return model, x, p_star, y


def compose(model) -> torch.Tensor:
    with torch.no_grad():
        p = model.layers[0].weight
        for layer in model.layers[1:]:
            p = layer.weight @ p
        return p.clone()


def run_one(
    method: str, schedule: str, base_lr: float, init: str,
    depth: int, d: int, n: int, steps: int, warmup: int, seed: int,
    device: torch.device,
) -> dict:
    dtype = torch.float64
    model, x, p_star, y = make_task(init, depth, d, n, seed, device, dtype)

    lr0 = base_lr
    if schedule in ("depth_sqrt", "warmup_ds"):
        lr0 = base_lr / math.sqrt(depth)
    elif schedule == "depth_lin":
        lr0 = base_lr / depth

    opt = build_optimizer(method, model, lr0)
    use_warmup = schedule in ("warmup200", "warmup_ds")
    sched = (
        torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: min(1.0, (t + 1) / warmup)
        )
        if use_warmup
        else None
    )

    crit = torch.nn.MSELoss()
    p_star_norm = float(p_star.norm().item())
    hist = []
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        if t % 20 == 0 or t == steps - 1:
            rel = float((compose(model) - p_star).norm().item()) / p_star_norm
            hist.append((t, rel))
            if not (rel == rel) or rel > 1e6:  # NaN or divergence
                return {"final_rel_err": float("inf"), "history": hist,
                        "diverged": True}
    return {"final_rel_err": hist[-1][1], "history": hist, "diverged": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    base_lrs = {"adam": [1e-3, 1e-4], "heavyball": [1e-2, 1e-3]}
    schedules = ["none", "warmup200", "depth_sqrt", "depth_lin", "warmup_ds"]
    inits = ["xavier", "identity"]

    results: dict = {}
    for init in inits:
        for method, lrs in base_lrs.items():
            for schedule in schedules:
                best = None
                for lr in lrs:
                    res = run_one(method, schedule, lr, init, args.depth,
                                  args.d, 4 * args.d, args.steps, args.warmup,
                                  args.seed, device)
                    res["base_lr"] = lr
                    print(f"[e12] {init} {method} {schedule} lr={lr:g}: "
                          f"final={res['final_rel_err']:.3e}", flush=True)
                    if best is None or res["final_rel_err"] < best["final_rel_err"]:
                        best = res
                results[f"{init}_{method}_{schedule}"] = best

    summary = {k: {"base_lr": v["base_lr"], "final_rel_err": v["final_rel_err"]}
               for k, v in results.items()}
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"results": results, "summary": summary}), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
