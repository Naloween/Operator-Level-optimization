"""E15 — single extreme point: d=64, L=1024, identity init, one informed lr per
method (no re-probing), to see whether the corner where BOTH known stressors
(large operator dimension from E5/E7, large depth from E13) combine still
escapes or finally shows a hard wall.

Learning rates are not re-swept; each is a multiplicative combination of the
two known shrink/growth factors measured independently:
  lr(d=64, L=1024) ~= anchor(d=16,L=128) * [lr(d=64,L=128)/anchor] * [lr(d=16,L=1024)/anchor]
using the E1/E7/E13 tuned values (see ANCHOR/RATIO tables below). This is a
first-order log-additive extrapolation, not a fit; a miss by a factor of a
few is expected and acceptable for a single diagnostic point.

Per-step cost at this corner is high (adam alone: ~170 ms/step measured), and
K-FAC/Shampoo/SOAP carry extra per-layer eigendecomposition cost on top, so
step budgets are set per-method to keep total wall-clock bounded, favoring
enough steps to see the qualitative trend (escaping / stuck / diverging) over
full convergence.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x15_extreme_d64_L1024 \
      --out_dir outputs/oplevel_rebuttal/e15_extreme
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

# lr(d=64,L=128) from E5(long-run)/E7; lr(d=16,L=1024) from E13; anchor = lr(d=16,L=128).
# combined = anchor * [lr_d64L128/anchor] * [lr_d16L1024/anchor]
LR_PLAN = {
    "adam":     {"anchor": 1e-4,  "lr_d64_L128": 3e-5,  "lr_d16_L1024": 3.33e-5, "steps": 3000},
    "heavyball":{"anchor": 1e-3,  "lr_d64_L128": 1e-2,  "lr_d16_L1024": 3.33e-4, "steps": 3000},
    "muon":     {"anchor": 1e-4,  "lr_d64_L128": 1e-4,  "lr_d16_L1024": 3.33e-5, "steps": 1500},
    "kfac":     {"anchor": 1e-4,  "lr_d64_L128": 1e-4,  "lr_d16_L1024": 1.11e-5, "steps": 800},
    "shampoo":  {"anchor": 3e-4,  "lr_d64_L128": 1e-3,  "lr_d16_L1024": 3.33e-5, "steps": 800},
    "soap":     {"anchor": 1e-5,  "lr_d64_L128": 3e-4,  "lr_d16_L1024": 1e-5,    "steps": 800},
}


def combined_lr(plan: dict) -> float:
    ratio_d = plan["lr_d64_L128"] / plan["anchor"]
    ratio_L = plan["lr_d16_L1024"] / plan["anchor"]
    return plan["anchor"] * ratio_d * ratio_L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--depth", type=int, default=1024)
    ap.add_argument("--als_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec_every", type=int, default=100)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    results: dict = {}
    for method, plan in LR_PLAN.items():
        lr = combined_lr(plan)
        steps = plan["steps"]
        print(f"[e15] {method}: lr={lr:.3e} steps={steps} ...", flush=True)
        res = run_cfg(method, lr, args.d, args.depth, steps, args.seed, device,
                      args.spec_every)
        res["lr"] = lr
        steps_s, flat = flatness_series(res)
        esc = escape_step(res)
        res["escape_step"] = esc
        res["escaped"] = esc is not None
        res["min_flatness"] = min(flat) if flat else None
        res["final_flatness"] = flat[-1] if flat else None
        print(f"[e15] {method}: final={res['final_rel_err']:.3e} "
              f"escaped={res['escaped']} escape_step={esc} "
              f"min_flatness={res['min_flatness']}", flush=True)
        results[method] = res

    als_res = run_cfg("als_exact", 1.0, args.d, args.depth, args.als_steps,
                      args.seed, device, args.spec_every)
    als_res["lr"] = 1.0
    print(f"[e15] als_exact: final={als_res['final_rel_err']:.3e}", flush=True)
    results["als_exact"] = als_res

    summary = {
        m: {"lr": r.get("lr"), "final_rel_err": r.get("final_rel_err"),
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
