"""X5 — full learning-rate grid {1e-1 ... 1e-6} for all six baselines.

The rebuttal promises a sweep down to 1e-6 at L=128 (AC point 5). E1's grid
stopped at 1e-4 and used 500 steps; at lr <= 1e-5 a 500-step budget would
confound "stall" with "not enough steps", so everything here runs 2000 steps
(matching E5) at identity init, d=16, L=128. One table, one budget.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x8_lr_grid_full \
      --out_dir outputs/oplevel_rebuttal/e8_lr_grid_full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from operator_level_optimization.scripts.train.deep_linear_compare import run_method
from operator_level_optimization.scripts.utils.io import get_device, jsonify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--methods", nargs="+",
        default=["heavyball", "adam", "muon", "kfac", "shampoo", "soap"],
    )
    ap.add_argument(
        "--lrs", type=float, nargs="+",
        default=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6],
    )
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    grid: dict[str, dict[str, float]] = {}
    for method in args.methods:
        grid[method] = {}
        for lr in args.lrs:
            try:
                res = run_method(
                    method=method, depth=args.depth, d=args.d, n=4 * args.d,
                    steps=args.steps, lr=lr, seed=args.seed, device=device,
                    dtype=torch.float64, target_mode="mse_grad", dc_lam=0.0,
                    dc_alt=1, dc_init_mode="plain", dc_init_scale=1.0,
                    spec_every=args.steps, spec_steps=None,
                    target_kind="orth", init_mode="identity",
                    als_lam=1e-4, als_sweeps=4,
                )
                final = float(res["history"][-1][1])
            except Exception as exc:
                final = float("inf")
                print(f"[e8] {method} lr={lr:g}: FAILED ({exc})", flush=True)
            grid[method][f"{lr:g}"] = final
            print(f"[e8] {method} lr={lr:g}: final={final:.3e}", flush=True)

    best = {
        m: min(v.items(), key=lambda kv: kv[1]) for m, v in grid.items() if v
    }
    lines = ["| method | " + " | ".join(f"{lr:g}" for lr in args.lrs) + " | best |",
             "|---" * (len(args.lrs) + 2) + "|"]
    for m, v in grid.items():
        row = " | ".join(
            ("div" if v[f"{lr:g}"] == float("inf") else f"{v[f'{lr:g}']:.1e}")
            for lr in args.lrs
        )
        lines.append(f"| {m} | {row} | {best[m][1]:.1e} @ {best[m][0]} |")
    (out_dir / "table_e8_lr_grid.md").write_text("\n".join(lines) + "\n")
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"grid": grid, "best": {m: {"lr": b[0], "final": b[1]}
                                                   for m, b in best.items()}}),
                   indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps({m: {"lr": b[0], "final": b[1]} for m, b in best.items()},
                     indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
