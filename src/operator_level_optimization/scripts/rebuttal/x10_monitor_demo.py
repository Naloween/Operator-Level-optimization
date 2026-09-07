"""X10 demo — the ALS collapse monitor fires under Xavier, is silent at identity.

Monitor (implemented in deep_linear_compare.als_exact_shared_target_step and in
core/optim/operator.py OperatorLevelMLP._step_als): per layer, per sweep,
min-eig(M_k) * min-eig(N_k) compared against lam; degenerate when < 10 lam,
i.e. the regime where filter (6) reduces to G/lam.

Three cases, d=16, L=128, lam=1e-4, one sweep (128 layer solves):
  A. Xavier init, no warm-start  -> collapsed contexts, fires everywhere;
  B. identity init               -> product = 1 >> lam, silent;
  C. Xavier init + warm-start    -> the warm-start resets the working stack to
     identity-like maps, so the *solved* contexts are healthy: this is exactly
     how the warm-start escapes the collapse the monitor detects.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x10_monitor_demo \
      --out_dir outputs/oplevel_rebuttal/e10_monitor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from operator_level_optimization.scripts.rebuttal.x4_monotone import make_instance
from operator_level_optimization.scripts.train.deep_linear_compare import (
    als_exact_shared_target_step,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    dtype = torch.float64

    cases = {
        "xavier_no_warmstart": ("xavier", False),
        "identity": ("identity", False),
        "xavier_warmstart": ("xavier", True),
    }
    report: dict = {}
    for name, (init, warm) in cases.items():
        weights, p_tgt = make_instance(
            args.depth, args.d, init, args.seed, device, dtype
        )
        mon: list = []
        als_exact_shared_target_step(
            weights, p_tgt, lam=args.lam, n_sweeps=1,
            warmstart_identity=warm, collapse_monitor=mon,
        )
        vals = np.array([m["min_gram_product"] for m in mon])
        n_deg = int(sum(m["degenerate"] for m in mon))
        report[name] = {
            "n_layer_solves": len(mon),
            "n_degenerate": n_deg,
            "min_gram_product_min": float(vals.min()),
            "min_gram_product_median": float(np.median(vals)),
            "min_gram_product_max": float(vals.max()),
            "threshold_10lam": 10.0 * args.lam,
        }
        print(
            f"[e10] {name:22s}: {n_deg}/{len(mon)} degenerate solves; "
            f"min-gram-product range [{vals.min():.3e}, {vals.max():.3e}] "
            f"vs 10*lam = {10.0 * args.lam:.1e}",
            flush=True,
        )

    (out_dir / "results.json").write_text(json.dumps(jsonify(
        {"report": report, "config": vars(args)}), indent=2))
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
