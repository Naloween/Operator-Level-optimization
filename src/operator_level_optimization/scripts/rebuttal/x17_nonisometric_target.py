"""E17 — does a non-isometric target (singular value 0.9, not 1) break the
depth-linear (not exponential) behavior at identity init, L=1024?

Motivation: the matrix-power argument for why depth enters ONLY linearly (not
exponentially) at identity init relies on the target being an isometry: the
theta=0 (identity-aligned) eigenmode of M = I - eta*G is then EXACTLY
protected (zero decay), only rotation-mismatched modes decay. Break isometry
(target singular values = 0.9) and even the aligned mode picks up a nonzero
decay rate ~ (1-s)*eta*c per step, compounding as (1-(1-s)*eta*c)^L over L
steps -- a genuinely different mechanism that the tuned-at-s=1 learning rates
were never selected to control.

No retuning here: each baseline uses the exact lr already found optimal for
the isometric target at L=1024 in E13 (i.e. this tests robustness of the
tuned operating point to a mismatch specified as a magnitude, not the
rotation-only mismatch studied so far), plus ALS-exact as the reference that
should be unaffected by construction (it solves the modewise projection
against whatever target is given, it doesn't rely on any theta=0-protection
argument at all).

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x17_nonisometric_target \
      --out_dir outputs/oplevel_rebuttal/e17_nonisometric
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from operator_level_optimization.scripts.train.deep_linear_compare import run_method
from operator_level_optimization.scripts.utils.io import get_device, jsonify

# lr already tuned at L=1024, d=16, isometric target (E13 summary).
TUNED_LR_L1024 = {
    "adam": 3.3333e-05,
    "heavyball": 3.3333e-04,
    "kfac": 1.1111e-05,
    "shampoo": 3.3333e-05,
    "muon": 3.3333e-05,
    "soap": 1.0e-05,
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
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depth", type=int, default=1024)
    ap.add_argument("--target_scales", type=float, nargs="+", default=[1.0, 0.9])
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
    for scale in args.target_scales:
        for method, lr in TUNED_LR_L1024.items():
            res = run_cfg(method, lr, args.d, args.depth, args.steps, args.seed,
                          device, args.spec_every, scale)
            res["lr"] = lr
            spectra = res.get("spectra", {})
            if spectra:
                final_t = max(spectra.keys())
                sv = np.array(spectra[final_t])
                res["final_flatness"] = float(sv.min() / sv.max())
                # For a target with singular value s, the "recovered" reference
                # is sv close to s, not 1.
                res["max_dev_from_target_sv"] = float(np.abs(sv - scale).max())
            print(f"[e17] scale={scale:g} {method:10s} lr={lr:.3e}: "
                  f"final={res['final_rel_err']:.3e} "
                  f"final_flatness={res.get('final_flatness')}", flush=True)
            results[f"{method}_s{scale:g}"] = res

        als_res = run_cfg("als_exact", 1.0, args.d, args.depth, args.als_steps,
                          args.seed, device, args.spec_every, scale)
        als_res["lr"] = 1.0
        print(f"[e17] scale={scale:g} als_exact: final={als_res['final_rel_err']:.3e}",
              flush=True)
        results[f"als_exact_s{scale:g}"] = als_res

    summary = {
        k: {"lr": r.get("lr"), "final_rel_err": r.get("final_rel_err"),
            "final_flatness": r.get("final_flatness"),
            "max_dev_from_target_sv": r.get("max_dev_from_target_sv")}
        for k, r in results.items()
    }
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"results": results, "summary": summary}), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
