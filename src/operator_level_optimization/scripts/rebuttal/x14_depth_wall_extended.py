"""E13b — resolve the grid-edge caveat in E13: does the escaping lr for K-FAC
and Shampoo keep shrinking with depth, and does escape eventually fail?

E13 (d=16, identity init, L up to 1024) found K-FAC's and Shampoo's winning lr
sat at the edge of the 4-point probe grid at L in {512, 1024} — the true
optimum might be smaller, and the trend of "lr shrinks ~9x per depth
doubling-ish" might eventually make escape impossible within budget, which
would BE the wall the anchoring hypothesis predicts. Here we extend the grid
two more decades below the E13 anchor for K-FAC and Shampoo only, and add
L=2048, with a longer budget to give slow-but-shrinking-lr runs room to finish.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x14_depth_wall_extended \
      --out_dir outputs/oplevel_rebuttal/e14_depth_wall_ext
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from operator_level_optimization.scripts.rebuttal.x13_depth_wall import (
    ANCHOR_LR,
    ESCAPE_THRESHOLD,
    escape_step,
    flatness_series,
    run_cfg,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--methods", nargs="+", default=["kfac", "shampoo"])
    ap.add_argument("--lr_multipliers", type=float, nargs="+",
                     default=[1.0 / 9.0, 1.0 / 27.0, 1.0 / 81.0, 1.0 / 243.0])
    ap.add_argument("--probe_steps", type=int, default=800)
    ap.add_argument("--long_steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec_every", type=int, default=200)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    results: dict = {}
    for depth in args.depths:
        for method in args.methods:
            anchor = ANCHOR_LR[method]
            probe_sweep = {}
            best = None
            for mult in args.lr_multipliers:
                lr = anchor * mult
                res = run_cfg(method, lr, args.d, depth, args.probe_steps,
                              args.seed, device, args.spec_every)
                probe_sweep[f"{lr:g}"] = res["final_rel_err"]
                print(f"[e14-probe] L={depth} {method} lr={lr:g}: "
                      f"final={res['final_rel_err']:.3e}", flush=True)
                if best is None or res["final_rel_err"] < best[1]:
                    best = (lr, res["final_rel_err"])
            best_lr = best[0]
            at_edge = best_lr == anchor * args.lr_multipliers[-1]
            long_res = run_cfg(method, best_lr, args.d, depth, args.long_steps,
                               args.seed, device, args.spec_every)
            long_res["lr"] = best_lr
            long_res["probe_sweep"] = probe_sweep
            long_res["at_grid_edge"] = at_edge
            steps_s, flat = flatness_series(long_res)
            esc = escape_step(long_res)
            long_res["escape_step"] = esc
            long_res["escaped"] = esc is not None
            long_res["min_flatness"] = min(flat) if flat else None
            print(f"[e14-long]  L={depth} {method} lr={best_lr:g} "
                  f"(edge={at_edge}) steps={args.long_steps}: "
                  f"final={long_res['final_rel_err']:.3e} "
                  f"escaped={long_res['escaped']} escape_step={esc}", flush=True)
            results[f"{method}_L{depth}"] = long_res

    summary = {
        key: {
            "lr": res.get("lr"), "at_grid_edge": res.get("at_grid_edge"),
            "final_rel_err": res.get("final_rel_err"),
            "escaped": res.get("escaped"), "escape_step": res.get("escape_step"),
        }
        for key, res in results.items()
    }
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"results": results, "summary": summary}), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
