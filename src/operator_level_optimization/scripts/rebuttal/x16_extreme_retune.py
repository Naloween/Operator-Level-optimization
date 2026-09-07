"""E16 — proper retune of the 5 methods that failed in E15's single-shot guess
at the extreme corner d=64, L=1024, identity init.

E15 used one multiplicatively-extrapolated lr per method, no re-probing:
K-FAC escaped cleanly (1.3e-10, step 548); Adam, heavy-ball, Muon, Shampoo,
SOAP did not. Shampoo's failure is the most surprising (it was a fast, clean
recoverer in every previous experiment at d=64/L=128 and d=16/L=1024
separately), which suggests the log-additive extrapolation may simply have
missed, not that these methods are walled. Here each failing method gets a
5-point log-spaced grid centered on its E15 lr (wider on the downside, since
depth previously required shrinking lr and the composition may under-shrink),
probed at a short budget, then a longer run at the probe winner.

Per-step cost at this size (measured): adam ~172ms, heavyball ~78ms,
muon ~238ms, shampoo ~440ms, soap ~446ms. Budgets below are set per-method
from these measurements to keep the whole sweep on the order of ~2 hours.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x16_extreme_retune \
      --out_dir outputs/oplevel_rebuttal/e16_extreme_retune
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

# center = E15 single-shot lr; budgets sized to measured per-step cost.
PLAN = {
    "adam":      {"center": 9.99e-06, "probe_steps": 600,  "long_steps": 6000},
    "heavyball": {"center": 3.33e-03, "probe_steps": 1500, "long_steps": 10000},
    "muon":      {"center": 3.33e-05, "probe_steps": 500,  "long_steps": 5000},
    "shampoo":   {"center": 1.11e-04, "probe_steps": 250,  "long_steps": 3000},
    "soap":      {"center": 3.00e-04, "probe_steps": 250,  "long_steps": 3000},
}
LR_MULTIPLIERS = [3.0, 1.0, 1.0 / 3.0, 1.0 / 9.0, 1.0 / 27.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--depth", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec_every", type=int, default=100)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    results: dict = {}
    for method, plan in PLAN.items():
        center = plan["center"]
        probe_sweep = {}
        best = None
        for mult in LR_MULTIPLIERS:
            lr = center * mult
            res = run_cfg(method, lr, args.d, args.depth, plan["probe_steps"],
                          args.seed, device, args.spec_every)
            probe_sweep[f"{lr:g}"] = res["final_rel_err"]
            print(f"[e16-probe] {method} lr={lr:.3e}: "
                  f"final={res['final_rel_err']:.3e}", flush=True)
            if best is None or res["final_rel_err"] < best[1]:
                best = (lr, res["final_rel_err"])
        best_lr = best[0]
        at_edge = best_lr == center * LR_MULTIPLIERS[-1]
        long_res = run_cfg(method, best_lr, args.d, args.depth, plan["long_steps"],
                           args.seed, device, args.spec_every)
        long_res["lr"] = best_lr
        long_res["probe_sweep"] = probe_sweep
        long_res["at_grid_edge"] = at_edge
        steps_s, flat = flatness_series(long_res)
        esc = escape_step(long_res)
        long_res["escape_step"] = esc
        long_res["escaped"] = esc is not None
        long_res["min_flatness"] = min(flat) if flat else None
        print(f"[e16-long]  {method} lr={best_lr:.3e} (edge={at_edge}) "
              f"steps={plan['long_steps']}: final={long_res['final_rel_err']:.3e} "
              f"escaped={long_res['escaped']} escape_step={esc}", flush=True)
        results[method] = long_res

    summary = {
        m: {"lr": r.get("lr"), "at_grid_edge": r.get("at_grid_edge"),
            "final_rel_err": r.get("final_rel_err"),
            "escaped": r.get("escaped"), "escape_step": r.get("escape_step"),
            "min_flatness": r.get("min_flatness")}
        for m, r in results.items()
    }
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"results": results, "summary": summary}), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
