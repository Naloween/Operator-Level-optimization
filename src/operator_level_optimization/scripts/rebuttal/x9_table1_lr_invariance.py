"""X11 — is the Table 1 one-step cosine invariant to the baseline learning rate?

Table 1 (paper): 2D toy setting, L=2, h=2, N=100, lam=1e-4, Xavier init;
cosine between the one-step operator increment dP and the ideal direction
P* - P0 (OLS solution), normalized.

Claim to verify (AC point 5 / NGj5 Q1): for the one-step measurement the
baseline lr is a positive scalar on the update whose direction is (to first
order in the increment) lr-independent, so the cosine cannot be a tuning
artifact. Exact invariance is only first-order: dP contains the cross term
dW_2 dW_1 which is O(lr^2), so mild deviations at lr ~ 1e-1 are expected and
reported honestly.

Reuses the Table-1 producer machinery from scripts/toy2d.py verbatim.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x9_table1_lr_invariance \
      --out_dir outputs/oplevel_rebuttal/e9_table1_lr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from operator_level_optimization.core.optim import KFAC, SOAP, Muon, Shampoo
from operator_level_optimization.scripts.toy2d import (
    DeepLinear,
    ToyOperatorALS,
    _one_step,
    compute_P,
    init_gaussian,
    make_dataset,
    ols_solution,
)
from operator_level_optimization.scripts.utils.io import jsonify


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        ((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-300)).item()
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--init_seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument(
        "--lrs", type=float, nargs="+",
        default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)

    X, y = make_dataset(args.n_samples, args.noise, args.data_seed)
    X, y = X.double(), y.double()
    net = DeepLinear(depth=args.depth, hidden=args.hidden).double()
    init_gaussian(net, args.init_seed)
    dp_star = ols_solution(X, y) - compute_P(net)

    factories = {
        "heavyball": lambda lr: (lambda m: torch.optim.SGD(
            m.parameters(), lr=lr, momentum=0.9)),
        "adam": lambda lr: (lambda m: torch.optim.Adam(
            m.parameters(), lr=lr, betas=(0.9, 0.999))),
        "muon": lambda lr: (lambda m: Muon(m.parameters(), lr=lr, momentum=0.0)),
        "kfac": lambda lr: (lambda m: KFAC(m.parameters(), lr=lr)),
        "shampoo": lambda lr: (lambda m: Shampoo(m.parameters(), lr=lr)),
        "soap": lambda lr: (lambda m: SOAP(m.parameters(), lr=lr)),
    }

    table: dict[str, dict[str, float]] = {}
    for method, mk in factories.items():
        table[method] = {}
        for lr in args.lrs:
            dp = _one_step(net, X, y, mk(lr))
            table[method][f"{lr:g}"] = cos(dp, dp_star)

    dp_als = _one_step(
        net, X, y,
        lambda m: ToyOperatorALS(m, X, y, lr=1.0, lam=args.lam, n_sweeps=10),
    )
    als_cos = cos(dp_als, dp_star)

    lines = [
        "| method | " + " | ".join(f"{lr:g}" for lr in args.lrs)
        + " | max dev |",
        "|---" * (len(args.lrs) + 2) + "|",
    ]
    spread: dict[str, float] = {}
    for method, row in table.items():
        vals = [row[f"{lr:g}"] for lr in args.lrs]
        spread[method] = max(vals) - min(vals)
        lines.append(
            f"| {method} | " + " | ".join(f"{v:+.4f}" for v in vals)
            + f" | {spread[method]:.2e} |"
        )
    lines.append(f"| als_exact (lr=1, ref) | {als_cos:+.4f} |" + " |" * (len(args.lrs) + 1))
    (out_dir / "table_e9_cosine_vs_lr.md").write_text("\n".join(lines) + "\n")

    payload = {
        "cos_by_method_lr": table,
        "als_cos": als_cos,
        "max_deviation_by_method": spread,
        "config": vars(args),
    }
    (out_dir / "results.json").write_text(json.dumps(jsonify(payload), indent=2))
    print("\n".join(lines), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
