"""E13 — is the collapse-and-recovery detour (E5/E7) amplified by DEPTH, not just
operator dimension d, to the point of an unrecoverable wall?

E5/E7 fixed L=128 and swept d in {16,32,64}: escape time grows with d, but every
method still escapes at d=16/L=128 given enough steps (E8), and even at d=64
only Adam's short-horizon-tuned lr got permanently stuck (E7). None of that
tested depth itself. Here we fix d=16 (the paper's own d) and sweep
L in {128, 256, 512, 1024}, identity init, tuned lr per (method, L), tracking:
  - whether the method escapes at all within budget (final rel err < 1e-6),
  - time-to-escape if it does,
  - the deepest flatness collapse reached (sigma_min/sigma_max of P).
ALS-exact is run at every L as the reference: by construction (per-layer exact
modewise solve against the same target every step) it should show no
depth dependence at all.

Two-phase per (method, L): short probe over a small lr grid centered on the
d=16/L=128 tuned optimum from E1/E8, then a long run at the probe's best lr.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x13_depth_wall \
      --out_dir outputs/oplevel_rebuttal/e13_depth_wall
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

# Tuned optima at d=16, L=128 from E1/E8 (identity init) — sweep centers.
ANCHOR_LR = {
    "adam": 1e-4,
    "heavyball": 1e-3,
    "kfac": 1e-4,
    "shampoo": 3e-4,
    "muon": 1e-4,
    "soap": 1e-5,
}
ESCAPE_THRESHOLD = 1e-6


def run_cfg(method: str, lr: float, d: int, depth: int, steps: int, seed: int,
            device: torch.device, spec_every: int) -> dict:
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


def escape_step(res: dict) -> int | None:
    for row in res.get("history", []):
        t, err = row[0], row[1]
        if err < ESCAPE_THRESHOLD:
            return int(t)
    return None


def flatness_series(res: dict) -> tuple[list[int], list[float]]:
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
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--methods", nargs="+",
                     default=["adam", "heavyball", "kfac", "shampoo", "muon", "soap"])
    ap.add_argument("--lr_multipliers", type=float, nargs="+",
                     default=[3.0, 1.0, 1.0 / 3.0, 1.0 / 9.0])
    ap.add_argument("--probe_steps", type=int, default=800)
    ap.add_argument("--long_steps", type=int, default=6000)
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
        for method in args.methods:
            anchor = ANCHOR_LR[method]
            probe_sweep = {}
            best = None
            for mult in args.lr_multipliers:
                lr = anchor * mult
                res = run_cfg(method, lr, args.d, depth, args.probe_steps,
                              args.seed, device, args.spec_every)
                probe_sweep[f"{lr:g}"] = res["final_rel_err"]
                print(f"[e13-probe] L={depth} {method} lr={lr:g}: "
                      f"final={res['final_rel_err']:.3e}", flush=True)
                if best is None or res["final_rel_err"] < best[1]:
                    best = (lr, res["final_rel_err"])
            best_lr = best[0]
            long_res = run_cfg(method, best_lr, args.d, depth, args.long_steps,
                               args.seed, device, args.spec_every)
            long_res["lr"] = best_lr
            long_res["probe_sweep"] = probe_sweep
            steps_s, flat = flatness_series(long_res)
            esc = escape_step(long_res)
            long_res["escape_step"] = esc
            long_res["escaped"] = esc is not None
            long_res["min_flatness"] = min(flat) if flat else None
            print(f"[e13-long]  L={depth} {method} lr={best_lr:g} "
                  f"steps={args.long_steps}: final={long_res['final_rel_err']:.3e} "
                  f"escaped={long_res['escaped']} escape_step={esc} "
                  f"min_flatness={long_res['min_flatness']}", flush=True)
            results[f"{method}_L{depth}"] = long_res

        als_res = run_cfg("als_exact", 1.0, args.d, depth, args.als_steps,
                          args.seed, device, args.spec_every)
        als_res["lr"] = 1.0
        results[f"als_exact_L{depth}"] = als_res
        print(f"[e13-als]   L={depth}: final={als_res['final_rel_err']:.3e}",
              flush=True)

    # ---- figure ------------------------------------------------------------
    colors = {"adam": "#ff7f0e", "heavyball": "#1f77b4", "kfac": "#2ca02c",
              "shampoo": "#8c564b", "muon": "#9467bd", "soap": "#e377c2",
              "als_exact": "#7f7f7f"}
    ls_by_depth = {d: ls for d, ls in zip(args.depths, ["-", "--", ":", "-."])}

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8))

    ax = axes[0]
    for key, res in results.items():
        method = key.rsplit("_L", 1)[0]
        depth = int(key.rsplit("_L", 1)[1])
        if "history" not in res:
            continue
        arr = np.array(res["history"], dtype=float)
        ax.plot(arr[:, 0], np.clip(arr[:, 1], 1e-16, None),
                color=colors.get(method), linestyle=ls_by_depth.get(depth, "-"),
                linewidth=1.3, alpha=0.9,
                label=f"{method} L={depth}" if depth == args.depths[0] else None)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative operator error")
    ax.set_title(f"Identity init, d={args.d} — depth sweep (tuned lr)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    ax = axes[1]
    for method in args.methods + ["als_exact"]:
        depths_esc = []
        escape_steps = []
        for depth in args.depths:
            res = results.get(f"{method}_L{depth}")
            if res is None:
                continue
            es = res.get("escape_step")
            if es is not None:
                depths_esc.append(depth)
                escape_steps.append(es)
        if depths_esc:
            ax.plot(depths_esc, escape_steps, color=colors.get(method),
                    marker="o", markersize=5, linewidth=1.6, label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Depth L")
    ax.set_ylabel(f"Steps to reach rel err < {ESCAPE_THRESHOLD:g}")
    ax.set_title("Escape time vs depth (missing point = did not escape)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    for method in args.methods + ["als_exact"]:
        depths_all = []
        min_flat = []
        for depth in args.depths:
            res = results.get(f"{method}_L{depth}")
            if res is None or res.get("min_flatness") is None:
                continue
            depths_all.append(depth)
            min_flat.append(res["min_flatness"])
        if depths_all:
            ax.plot(depths_all, np.clip(min_flat, 1e-18, None),
                    color=colors.get(method), marker="s", markersize=5,
                    linewidth=1.6, label=method)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Depth L")
    ax.set_ylabel(r"deepest $\sigma_{min}(P)/\sigma_{max}(P)$ reached")
    ax.set_title("Collapse depth vs network depth")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    save_fig(fig, out_dir / "fig_e13_depth_wall.png", dpi=180, bbox_inches="tight")

    summary = {
        key: {
            "lr": res.get("lr"), "final_rel_err": res.get("final_rel_err"),
            "escaped": res.get("escaped"), "escape_step": res.get("escape_step"),
            "min_flatness": res.get("min_flatness"),
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
