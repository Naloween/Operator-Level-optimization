"""E21 -- the initialization-exactness wall, and its dissociation from warm-start.

Three things this script establishes together, at identity init, isotropic
inputs, orthogonal target (deep linear, d=16):

1. Depth-sweep escape (Adam): with a properly per-depth-tuned learning rate,
   training from EXACT identity (init_diag=1.0) converges cleanly at every
   depth. Perturbing every layer to init_diag=0.99 (a 1% per-layer deviation,
   target unchanged) is harmless at L<=256 but produces a permanent stall at
   L>=512 at the SAME learning rate, because the composed initial operator
   has already collapsed geometrically before training starts (0.99^512 ~
   6e-3, 0.99^1024 ~ 3e-5): a conditioning failure, not a step-size failure.

2. Full lr sweep confirms it is not a retuning problem: at L=1024,
   init_diag=0.99, no learning rate across 13 points spanning 1e-7 to 1e-1
   escapes below ~79% relative error (too small = insufficient signal to move
   at all; too large = numerical divergence).

3. ALS-exact is unaffected by the same defect, and -- critically -- identically
   so whether als_exact_shared_target_step's warm_start_identity is enabled
   or disabled. This dissociates the robustness from the warm-start heuristic:
   it comes from solving for the operator target directly at every step
   (decoupling the outer P-space trajectory from the internal L-layer weight
   configuration), not from resetting to identity before each solve. Warm
   start is needed only in the strictly more severe Xavier regime (context
   degenerate from step 0), tested separately in x3_fairness.py.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x21_init_exactness_wall \
      --out_dir outputs/oplevel_rebuttal/e21_init_exactness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from operator_level_optimization.scripts.train.deep_linear_compare import (
    als_exact_shared_target_step,
    compose_operator,
    mse_grad_target,
    run_method,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify

# Adam's own per-depth-tuned lr (identity init, isometric target; from the
# depth-scaling sweep in x13_depth_wall.py).
ADAM_TUNED_LR = {128: 3e-4, 256: 1e-4, 512: 1e-4, 1024: 3.33e-5}
LR_SWEEP = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def run_adam_depth_sweep(d: int, steps: int, seed: int, device: torch.device) -> dict:
    out = {}
    for init_diag in (1.0, 0.99):
        for depth, lr in ADAM_TUNED_LR.items():
            res = run_method(
                method="adam", depth=depth, d=d, n=4 * d, steps=steps, lr=lr,
                seed=seed, device=device, dtype=torch.float64,
                target_mode="mse_grad", dc_lam=0.0, dc_alt=1,
                dc_init_mode="plain", dc_init_scale=1.0,
                spec_every=steps, spec_steps=None,
                target_kind="orth", init_mode="identity",
                als_lam=1e-4, als_sweeps=4, target_scale=1.0,
                init_diag_scale=init_diag,
            )
            hist = res["history"]
            key = f"adam_init{init_diag:g}_L{depth}"
            out[key] = {
                "lr": lr, "initial_err": hist[0][1], "final_err": hist[-1][1],
            }
            print(f"[e21-sweep] init_diag={init_diag:g} L={depth:5d} lr={lr:.2e}: "
                  f"initial={hist[0][1]:.4e} final={hist[-1][1]:.4e}", flush=True)
    return out


def run_lr_sweep(d: int, depth: int, steps: int, seed: int, device: torch.device,
                  init_diag: float) -> dict:
    out = {}
    for lr in LR_SWEEP:
        res = run_method(
            method="adam", depth=depth, d=d, n=4 * d, steps=steps, lr=lr,
            seed=seed, device=device, dtype=torch.float64,
            target_mode="mse_grad", dc_lam=0.0, dc_alt=1,
            dc_init_mode="plain", dc_init_scale=1.0,
            spec_every=steps, spec_steps=None,
            target_kind="orth", init_mode="identity",
            als_lam=1e-4, als_sweeps=4, target_scale=1.0,
            init_diag_scale=init_diag,
        )
        hist = res["history"]
        peak = max(h[1] for h in hist)
        out[f"{lr:g}"] = {"final_err": hist[-1][1], "peak_err": peak}
        print(f"[e21-lrsweep] lr={lr:.0e}: final={hist[-1][1]:.4e} peak={peak:.4e}",
              flush=True)
    return out


def run_als_manual(depth: int, d: int, n: int, steps: int, lr: float, seed: int,
                    device: torch.device, init_diag_scale: float,
                    warmstart_identity: bool, als_lam: float = 1e-4,
                    als_sweeps: int = 4) -> float:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, d, device=device, dtype=torch.float64, generator=g)
    q1, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=torch.float64, generator=g))
    q2, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=torch.float64, generator=g))
    p_star = q1 @ q2.T
    y = x @ p_star.T
    weights = [init_diag_scale * torch.eye(d, device=device, dtype=torch.float64)
               for _ in range(depth)]
    rel = None
    for t in range(steps + 1):
        p = compose_operator(weights)
        rel = (torch.norm(p - p_star) / torch.norm(p_star)).item()
        if t == steps:
            break
        grad_p = mse_grad_target(p, x, y)
        p_tgt = p - lr * grad_p
        weights = als_exact_shared_target_step(
            weights, p_tgt, lam=als_lam, n_sweeps=als_sweeps,
            warmstart_identity=warmstart_identity,
        )
    return rel


def run_als_comparison(d: int, steps: int, seed: int, device: torch.device) -> dict:
    out = {}
    for depth in (512, 1024):
        for init_diag in (1.0, 0.99):
            for warmstart in (False, True):
                final = run_als_manual(depth, d, 4 * d, steps, 1.0, seed, device,
                                       init_diag, warmstart)
                key = f"als_L{depth}_init{init_diag:g}_warmstart{warmstart}"
                out[key] = final
                print(f"[e21-als] L={depth} init_diag={init_diag:g} "
                      f"warmstart={warmstart}: final={final:.4e}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depth_sweep_steps", type=int, default=3000)
    ap.add_argument("--lr_sweep_depth", type=int, default=1024)
    ap.add_argument("--lr_sweep_steps", type=int, default=4000)
    ap.add_argument("--lr_sweep_init_diag", type=float, default=0.99)
    ap.add_argument("--als_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    depth_sweep = run_adam_depth_sweep(args.d, args.depth_sweep_steps, args.seed, device)
    lr_sweep = run_lr_sweep(args.d, args.lr_sweep_depth, args.lr_sweep_steps,
                            args.seed, device, args.lr_sweep_init_diag)
    als_comparison = run_als_comparison(args.d, args.als_steps, args.seed, device)

    payload = {
        "adam_depth_sweep_init_vs_defect": depth_sweep,
        "adam_lr_sweep_at_defect": lr_sweep,
        "als_init_vs_defect_and_warmstart": als_comparison,
    }
    (out_dir / "results.json").write_text(json.dumps(jsonify(payload), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
