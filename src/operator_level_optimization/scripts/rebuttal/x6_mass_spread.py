"""Q2 verification — does lambda > 0 spread the update mass across layers?

The global minimizer of objective (3) with lam > 0 is the minimum-norm
realization of the operator displacement, which distributes the increment
across layers (the user's intuition). Yet E1 measured participation ratio ~ 1
for sequential ALS at every lambda. Hypothesis (from E4): the implemented
solver penalizes only the per-visit increment dW (proximal variant) rather
than the accumulated increment W - W0, so the restoring force -lam (W - W0)
that would migrate mass out of the first-visited layer is absent.

Here we run both variants on one projection instance (identity init, d=16,
L=128, orthogonal target — the E4 setup) for many reverse sweeps and track,
after every sweep:
  - per-layer accumulated mass m_l = ||W_l - W0_l||_F^2,
  - participation ratio PR = (sum m)^2 / sum m^2  (1 = concentrated, L = flat),
  - fidelity ||prod W - P_tgt||_F^2.

Expected: proximal PR stays ~1 at any sweep count; recentred PR grows at a
rate set by lambda (each revisit shrinks the concentrated layer's excess by
~1/(1+lam)), so lam = 1e-1 spreads visibly within ~100 sweeps while
lam = 1e-4 is quasi-static.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x6_mass_spread \
      --out_dir outputs/oplevel_rebuttal/e6_mass_spread
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from operator_level_optimization.scripts.rebuttal.x4_monotone import (
    make_instance,
    solve_sub_exact,
)
from operator_level_optimization.scripts.train.deep_linear_compare import (
    _product_excluding,
    compose_operator,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify
from operator_level_optimization.scripts.utils.plotting import save_fig


def run_variant(
    variant: str,  # "recentred" | "proximal"
    weights0: list[torch.Tensor],
    p_tgt: torch.Tensor,
    lam: float,
    n_sweeps: int,
) -> dict:
    w = [x.clone() for x in weights0]
    depth = len(w)
    pr_hist: list[float] = []
    fid_hist: list[float] = []
    mass_final: list[float] = []
    for _ in range(n_sweeps):
        for k in range(depth - 1, -1, -1):  # reverse sweep, as in the paper
            A_k, B_k = _product_excluding(w, k)
            R_k = p_tgt - A_k @ w[k] @ B_k
            G = A_k.T @ R_k @ B_k.T
            if variant == "recentred":
                G = G - lam * (w[k] - weights0[k])
            dW = solve_sub_exact(A_k.T @ A_k, B_k @ B_k.T, G, lam)
            w[k] = w[k] + dW
        masses = np.array(
            [float(((wi - w0i).norm() ** 2).item()) for wi, w0i in zip(w, weights0)]
        )
        tot = masses.sum()
        pr = float(tot**2 / max((masses**2).sum(), 1e-300))
        pr_hist.append(pr)
        fid_hist.append(float(((compose_operator(w) - p_tgt).norm() ** 2).item()))
        mass_final = masses.tolist()
    return {
        "variant": variant,
        "lam": lam,
        "pr_per_sweep": pr_hist,
        "fid_per_sweep": fid_hist,
        "final_layer_mass": mass_final,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-4, 1e-2, 1e-1])
    ap.add_argument("--n_sweeps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    dtype = torch.float64

    weights0, p_tgt = make_instance(
        args.depth, args.d, "identity", args.seed, device, dtype
    )

    runs: list[dict] = []
    for lam in args.lams:
        for variant in ("proximal", "recentred"):
            res = run_variant(variant, weights0, p_tgt, lam, args.n_sweeps)
            runs.append(res)
            print(
                f"[e6] {variant:9s} lam={lam:g}: PR sweep1={res['pr_per_sweep'][0]:.2f} "
                f"-> sweep{args.n_sweeps}={res['pr_per_sweep'][-1]:.2f}; "
                f"fid final={res['fid_per_sweep'][-1]:.3e}",
                flush=True,
            )

    colors = {1e-4: "#9467bd", 1e-2: "#2ca02c", 1e-1: "#d62728"}
    ls = {"proximal": ":", "recentred": "-"}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))

    ax = axes[0]
    for res in runs:
        ax.plot(
            np.arange(1, len(res["pr_per_sweep"]) + 1),
            res["pr_per_sweep"],
            color=colors.get(res["lam"]),
            linestyle=ls[res["variant"]],
            label=f"{res['variant']}, $\\lambda$={res['lam']:g}",
            linewidth=1.6,
        )
    ax.axhline(args.depth, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Participation ratio of layer mass")
    ax.set_title(f"Mass spreading, identity init d={args.d}, L={args.depth}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    for res in runs:
        m = np.array(res["final_layer_mass"])
        ax.plot(
            np.arange(1, len(m) + 1),
            np.clip(np.sqrt(m), 1e-18, None),
            color=colors.get(res["lam"]),
            linestyle=ls[res["variant"]],
            label=f"{res['variant']}, $\\lambda$={res['lam']:g}",
            linewidth=1.4,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Layer index $\\ell$")
    ax.set_ylabel(r"$\|W_\ell - W^0_\ell\|_F$ after last sweep")
    ax.set_title("Final per-layer mass profile")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    for res in runs:
        ax.plot(
            np.arange(1, len(res["fid_per_sweep"]) + 1),
            np.clip(res["fid_per_sweep"], 1e-32, None),
            color=colors.get(res["lam"]),
            linestyle=ls[res["variant"]],
            label=f"{res['variant']}, $\\lambda$={res['lam']:g}",
            linewidth=1.4,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Sweep")
    ax.set_ylabel(r"$\|\prod W - P_{tgt}\|_F^2$")
    ax.set_title("Fidelity while spreading")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    save_fig(fig, out_dir / "fig_e6_mass_spread.png", dpi=180, bbox_inches="tight")

    summary = {
        f"{r['variant']}_lam{r['lam']:g}": {
            "pr_first": r["pr_per_sweep"][0],
            "pr_last": r["pr_per_sweep"][-1],
            "fid_last": r["fid_per_sweep"][-1],
        }
        for r in runs
    }
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"runs": runs, "summary": summary}), indent=2)
    )
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
