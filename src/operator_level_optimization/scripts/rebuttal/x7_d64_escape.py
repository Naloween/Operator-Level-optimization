"""Q1 verification — is the d=64 identity-init stall a wall or a slow rate?

E5 showed Adam/HB stuck at ~1.8e-1 at d=64, L=128 after 2000 steps (lr-swept),
while at d<=32 they escape after a long collapsed plateau. Two gaps remain
before the rebuttal can claim "cannot escape even with tuned lr at large d":

Part A (coverage): Muon, Shampoo, SOAP were only swept at d=16 (E1). Run them
    at d=64, L=128, identity init, 2000 steps, per-method lr grids centered on
    their E1 optima.
Part B (wall test): run Adam (lr 1e-4, 3e-5) and HB (lr 1e-2) for 20000 steps
    (10x the E5 budget), tracking sigma_min(P) along the way. If rel err stays
    flat at ~1.8e-1 with a collapsed tail for 20k steps, "no escape at any
    practical budget" is supported; if a late escape begins, the claim must
    stay a rate statement.

NOTE: this is not plan-item X7 (MNIST accuracy, dropped); file numbering just
continues the x1..x6 sequence.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x7_d64_escape \
      --out_dir outputs/oplevel_rebuttal/e7_d64_escape
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from operator_level_optimization.scripts.train.deep_linear_compare import run_method
from operator_level_optimization.scripts.utils.io import get_device, jsonify
from operator_level_optimization.scripts.utils.plotting import save_fig


def run_cfg(
    method: str, lr: float, d: int, depth: int, steps: int, seed: int,
    device: torch.device, spec_every: int,
) -> dict:
    try:
        res = run_method(
            method=method, depth=depth, d=d, n=4 * d, steps=steps, lr=lr,
            seed=seed, device=device, dtype=torch.float64,
            target_mode="mse_grad", dc_lam=0.0, dc_alt=1,
            dc_init_mode="plain", dc_init_scale=1.0,
            spec_every=spec_every, spec_steps=None,
            target_kind="orth", init_mode="identity",
            als_lam=1e-4, als_sweeps=4,
        )
    except Exception as exc:
        return {"failed": str(exc), "final_rel_err": float("inf")}
    res["final_rel_err"] = float(res["history"][-1][1])
    return res


def flatness_series(res: dict) -> tuple[list[int], list[float]]:
    """(steps, sigma_min/sigma_max) from recorded spectra."""
    if "spectra" not in res:
        return [], []
    steps_s = sorted(int(s) for s in res["spectra"].keys())
    vals = []
    for s in steps_s:
        sv = np.array(res["spectra"][str(s) if str(s) in res["spectra"] else s])
        vals.append(float(sv.min() / sv.max()))
    return steps_s, vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--steps_sweep", type=int, default=2000)
    ap.add_argument("--steps_long", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    grids = {
        "muon": [3e-4, 1e-4, 3e-5],
        "shampoo": [3e-3, 1e-3, 3e-4],
        "soap": [3e-3, 1e-3, 3e-4],
    }
    long_cfgs = [("adam", 1e-4), ("adam", 3e-5), ("heavyball", 1e-2)]

    results: dict = {}

    # Part A — six-method coverage at d=64.
    for method, lrs in grids.items():
        best = None
        sweep = {}
        for lr in lrs:
            res = run_cfg(method, lr, args.d, args.depth, args.steps_sweep,
                          args.seed, device, spec_every=100)
            sweep[str(lr)] = res["final_rel_err"]
            print(f"[e7-A] {method} lr={lr:g}: final={res['final_rel_err']:.3e}",
                  flush=True)
            if best is None or res["final_rel_err"] < best["final_rel_err"]:
                best = res
                best["lr"] = lr
        best["sweep_final_by_lr"] = sweep
        results[f"{method}_sweep"] = best

    # Part B — long-budget wall test.
    for method, lr in long_cfgs:
        res = run_cfg(method, lr, args.d, args.depth, args.steps_long,
                      args.seed, device, spec_every=500)
        res["lr"] = lr
        arr = np.array(res.get("history", [[0, np.inf]]), dtype=float)
        # Escape diagnostic: rel err drop over the last half of the run.
        half = arr[len(arr) // 2:, 1]
        res["relerr_half_to_final_ratio"] = float(half[0] / max(half[-1], 1e-300))
        print(f"[e7-B] {method} lr={lr:g} steps={args.steps_long}: "
              f"final={res['final_rel_err']:.3e} "
              f"half/final={res['relerr_half_to_final_ratio']:.3f}", flush=True)
        results[f"{method}_lr{lr:g}_long"] = res

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = {"adam": "#ff7f0e", "heavyball": "#1f77b4", "muon": "#2ca02c",
              "shampoo": "#8c564b", "soap": "#e377c2"}
    ax = axes[0]
    for key, res in results.items():
        if "history" not in res:
            continue
        method = key.split("_")[0]
        arr = np.array(res["history"], dtype=float)
        ax.plot(arr[:, 0], np.clip(arr[:, 1], 1e-16, None),
                color=colors.get(method), linewidth=1.5,
                linestyle="-" if key.endswith("_long") else "--",
                label=f"{method} lr={res['lr']:g}"
                      f"{' (20k)' if key.endswith('_long') else ''}")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative operator error")
    ax.set_title(f"Identity init, d={args.d}, L={args.depth}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for key, res in results.items():
        steps_s, vals = flatness_series(res)
        if not steps_s:
            continue
        method = key.split("_")[0]
        ax.plot(steps_s, np.clip(vals, 1e-18, None), color=colors.get(method),
                linewidth=1.5, linestyle="-" if key.endswith("_long") else "--",
                label=f"{method} lr={res['lr']:g}")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\sigma_{min}(P)/\sigma_{max}(P)$")
    ax.set_title("Tail collapse over training (target flatness = 1)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, out_dir / "fig_e7_d64_escape.png", dpi=180, bbox_inches="tight")

    summary = {}
    for key, res in results.items():
        steps_s, vals = flatness_series(res)
        summary[key] = {
            "lr": res.get("lr"),
            "final_rel_err": res.get("final_rel_err"),
            "min_flatness": min(vals) if vals else None,
            "final_flatness": vals[-1] if vals else None,
            "relerr_half_to_final_ratio": res.get("relerr_half_to_final_ratio"),
            "sweep_final_by_lr": res.get("sweep_final_by_lr"),
        }
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"results": results, "summary": summary}), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
