"""E18 -- the natural collapse-at-depth scenario: fix lr at its L=128-tuned
value and DO NOT retune as depth grows.

E13/E16 always retuned lr per depth (giving baselines the benefit of a full
per-depth search), and found the mismatch-only effect (identity init,
isometric target) scales only linearly in depth -- controllable, not walled.
The matrix-power argument (M = I - eta*G applied identically to every layer
at step 1, composed operator = M^L) predicts that if eta is instead held
FIXED at its L=128-optimal value while L grows, the effective exponent
L*eta*c grows linearly in L, so flatness should degrade increasingly badly
with depth even though the network starts perfectly conditioned. This is the
realistic failure mode: nobody re-derives the optimal lr from scratch at
every depth encountered in practice.

lr values are each method's own L=128 optimum from E13's probe grid.
ALS-exact is the reference: it should be indifferent to L by construction
(fresh modewise solve every step, no dependence on a fixed step size).

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x18_fixed_lr_depth_collapse \
      --out_dir outputs/oplevel_rebuttal/e18_fixed_lr_collapse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from operator_level_optimization.scripts.rebuttal.x13_depth_wall import (
    escape_step,
    flatness_series,
    run_cfg,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify

# Each method's own L=128-tuned lr from E13's probe (held fixed across depth).
FIXED_LR_L128 = {
    "adam": 3e-4,
    "heavyball": 1e-3,
    "kfac": 1e-4,
    "shampoo": 3e-4,
    "muon": 3e-4,
    "soap": 3e-5,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--als_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec_every", type=int, default=200)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    results: dict = {}
    for depth in args.depths:
        for method, lr in FIXED_LR_L128.items():
            res = run_cfg(method, lr, args.d, depth, args.steps, args.seed,
                          device, args.spec_every)
            res["lr"] = lr
            steps_s, flat = flatness_series(res)
            esc = escape_step(res)
            res["escape_step"] = esc
            res["escaped"] = esc is not None
            res["min_flatness"] = min(flat) if flat else None
            res["final_flatness"] = flat[-1] if flat else None
            print(f"[e18] L={depth:5d} {method:10s} lr={lr:.3e} (fixed): "
                  f"final={res['final_rel_err']:.3e} escaped={res['escaped']} "
                  f"min_flat={res['min_flatness']} final_flat={res['final_flatness']}",
                  flush=True)
            results[f"{method}_L{depth}"] = res

        als_res = run_cfg("als_exact", 1.0, args.d, depth, args.als_steps,
                          args.seed, device, args.spec_every)
        als_res["lr"] = 1.0
        print(f"[e18] L={depth:5d} als_exact: final={als_res['final_rel_err']:.3e}",
              flush=True)
        results[f"als_exact_L{depth}"] = als_res

    summary = {
        k: {"lr": r.get("lr"), "final_rel_err": r.get("final_rel_err"),
            "escaped": r.get("escaped"), "escape_step": r.get("escape_step"),
            "min_flatness": r.get("min_flatness"),
            "final_flatness": r.get("final_flatness")}
        for k, r in results.items()
    }
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"results": results, "summary": summary}), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
