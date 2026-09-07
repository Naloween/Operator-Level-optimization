"""E5 — Does tuned Adam's identity-init escape survive spectral scrutiny and width?

Follow-up to E1's surprise (tuned Adam reaches 4.5e-6 at d=16, L=128, identity
init). Two questions:
  (a) Transient bias: even when Adam converges, does its operator spectrum
      collapse in the tail along the way (dominant modes first, tail
      exponentially slower)? Track sigma_min(P) and full spectra over training.
  (b) Dimension scaling: does the escape survive d in {16, 32, 64} at L=128,
      with a per-d Adam lr sweep? (The Gram-anisotropy the mismatch builds up
      grows with operator dimension — paper App. D.1 shows all methods pass at
      d=2; E1 shows most fail at d=16.)

Controls: heavyball (tuned) as a stalling reference at each d; ALS-exact
(lam=1e-4) as the projection reference.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x5_adam_spectra_dims \
      --out_dir outputs/oplevel_rebuttal/e5_adam_spectra
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--dims", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--adam_lrs", type=float, nargs="+", default=[3e-4, 1e-4, 3e-5])
    ap.add_argument("--hb_lrs", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec_every", type=int, default=100)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    results: dict = {}
    for d in args.dims:
        for method, lrs in (("adam", args.adam_lrs), ("heavyball", args.hb_lrs)):
            best = None
            sweep = {}
            for lr in lrs:
                res = run_cfg(method, lr, d, args.depth, args.steps, args.seed,
                              device, args.spec_every)
                sweep[str(lr)] = res["final_rel_err"]
                print(f"[e5] d={d} {method} lr={lr}: final={res['final_rel_err']:.3e}",
                      flush=True)
                if best is None or res["final_rel_err"] < best["final_rel_err"]:
                    best = res
                    best["lr"] = lr
            best["sweep_final_by_lr"] = sweep
            results[f"{method}_d{d}"] = best
        res = run_cfg("als_exact", 1.0, d, args.depth, min(args.steps, 500),
                      args.seed, device, args.spec_every)
        res["lr"] = 1.0
        print(f"[e5] d={d} als_exact: final={res['final_rel_err']:.3e}", flush=True)
        results[f"als_exact_d{d}"] = res

    # ---- figures -----------------------------------------------------------
    colors = {"adam": "#ff7f0e", "heavyball": "#1f77b4", "als_exact": "#9467bd"}
    ls_by_d = {args.dims[i]: ["-", "--", ":"][i % 3] for i in range(len(args.dims))}

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.6))
    ax = axes[0]
    for key, res in results.items():
        if "history" not in res:
            continue
        method, d = key.rsplit("_d", 1)
        arr = np.array(res["history"], dtype=float)
        ax.plot(arr[:, 0], np.clip(arr[:, 1], 1e-16, None),
                color=colors[method], linestyle=ls_by_d[int(d)],
                label=f"{method} d={d} (lr={res['lr']:g})", linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Relative operator error")
    ax.set_title(f"Identity init, L={args.depth} — tuned lr per (method, d)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5)

    # sigma_min(P)/sigma_max(P) over training (target has flat spectrum = 1).
    ax = axes[1]
    for key, res in results.items():
        if "spectra" not in res:
            continue
        method, d = key.rsplit("_d", 1)
        steps_s = sorted(int(s) for s in res["spectra"].keys())
        cond = [np.array(res["spectra"][str(s) if str(s) in res["spectra"] else s])
                for s in steps_s]
        vals = [c.min() / c.max() for c in cond]
        ax.plot(steps_s, np.clip(vals, 1e-16, None), color=colors[method],
                linestyle=ls_by_d[int(d)], label=f"{method} d={d}", linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\sigma_{min}(P)/\sigma_{max}(P)$")
    ax.set_title("Spectral flatness over training (target = 1)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5)

    # final spectra at largest d
    ax = axes[2]
    d_show = max(args.dims)
    for method in ("adam", "heavyball", "als_exact"):
        res = results.get(f"{method}_d{d_show}", {})
        if "spectra" not in res:
            continue
        final_t = max(int(s) for s in res["spectra"].keys())
        sv = np.array(res["spectra"][str(final_t) if str(final_t) in res["spectra"] else final_t])
        ax.plot(np.arange(1, len(sv) + 1), np.clip(sv, 1e-16, None),
                color=colors[method], marker="o", markersize=2.5,
                label=f"{method} (t={final_t})", linewidth=1.4)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel(r"$\sigma_i(P)$")
    ax.set_title(f"Final spectrum, d={d_show}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    save_fig(fig, out_dir / "fig_e5_adam_dims.png", dpi=180, bbox_inches="tight")

    summary = {
        key: {"lr": res.get("lr"), "final_rel_err": res.get("final_rel_err")}
        for key, res in results.items()
    }
    (out_dir / "results.json").write_text(json.dumps(jsonify(
        {"results": results, "summary": summary}), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
