"""E20 -- companion to E19 at more aggressive (still plausible) learning rates.

E19 uses each method's small, previously-tuned L=128 lr, held fixed. The toy
matrix-power model predicts only mild growth from that at expansive targets
within L<=2048 (e.g. heavyball scale=2 at lr=1e-3 reaches only 1.29x by
L=2048). At more aggressive but still ordinary lr choices (1e-2 to 3e-2), the
same toy model predicts genuine, dramatic blowup (heavyball scale=2, lr=3e-2:
factor 2130 by L=2048). This run tests whether that dramatic prediction
survives real (non-toy) iterative training.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x20_expansive_wall_aggressive_lr \
      --out_dir outputs/oplevel_rebuttal/e20_expansive_wall_aggressive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from operator_level_optimization.scripts.train.deep_linear_compare import run_method
from operator_level_optimization.scripts.utils.io import get_device, jsonify

FIXED_LR = {
    "adam": 1e-2,
    "heavyball": 3e-2,
    "kfac": 1e-2,
    "shampoo": 1e-2,
}


def run_cfg(method, lr, d, depth, steps, seed, device, spec_every, target_scale):
    try:
        res = run_method(
            method=method, depth=depth, d=d, n=4 * d, steps=steps, lr=lr,
            seed=seed, device=device, dtype=torch.float64,
            target_mode="mse_grad", dc_lam=0.0, dc_alt=1,
            dc_init_mode="plain", dc_init_scale=1.0,
            spec_every=spec_every, spec_steps=None,
            target_kind="orth", init_mode="identity",
            als_lam=1e-4, als_sweeps=4, target_scale=target_scale,
        )
    except Exception as exc:
        return {"failed": str(exc), "final_rel_err": float("inf")}
    res["final_rel_err"] = float(res["history"][-1][1])
    hist = res.get("history", [])
    res["peak_rel_err"] = max((h[1] for h in hist), default=float("nan"))
    res["diverged"] = (not torch.isfinite(torch.tensor(res["final_rel_err"]))) or res["final_rel_err"] > 1e6
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--target_scales", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--als_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec_every", type=int, default=100)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    results: dict = {}
    for scale in args.target_scales:
        for depth in args.depths:
            for method, lr in FIXED_LR.items():
                res = run_cfg(method, lr, args.d, depth, args.steps, args.seed,
                              device, args.spec_every, scale)
                res["lr"] = lr
                print(f"[e20] scale={scale:g} L={depth:5d} {method:10s} "
                      f"lr={lr:.1e} (fixed): final={res['final_rel_err']:.3e} "
                      f"peak={res['peak_rel_err']:.3e} diverged={res['diverged']}",
                      flush=True)
                results[f"{method}_s{scale:g}_L{depth}"] = res

            als_res = run_cfg("als_exact", 1.0, args.d, depth, args.als_steps,
                              args.seed, device, args.spec_every, scale)
            als_res["lr"] = 1.0
            print(f"[e20] scale={scale:g} L={depth:5d} als_exact: "
                  f"final={als_res['final_rel_err']:.3e}", flush=True)
            results[f"als_exact_s{scale:g}_L{depth}"] = als_res

    summary = {
        k: {"lr": r.get("lr"), "final_rel_err": r.get("final_rel_err"),
            "peak_rel_err": r.get("peak_rel_err"), "diverged": r.get("diverged")}
        for k, r in results.items()
    }
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"results": results, "summary": summary}), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
