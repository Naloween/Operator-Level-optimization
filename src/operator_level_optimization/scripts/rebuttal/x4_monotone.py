"""E4 — Monotone-decrease verification for ALS + corrected uniqueness claim.

Rebuttal experiment for 4jZv Q2 / AC point 3. The corrected theory:

- The joint objective (3) is non-convex for L >= 2; global uniqueness is FALSE.
- Each layer sub-problem (4) is strongly convex for lam > 0: unique minimizer.
- Two provable monotonicity statements, verified here numerically:

  (a) Recentred ALS (exact block-coordinate descent on Eq. 3, penalty anchored
      at the original weights W^0): the FULL objective
        F = ||P - P_tgt||_F^2 + lam * sum_l ||W_l - W_l^0||_F^2
      is non-increasing after every layer solve.

  (b) Proximal ALS (the paper's per-visit-increment solve, Eq. 4 with residual
      recomputed at the current iterate): the FIDELITY term
        F_fid = ||P - P_tgt||_F^2
      is non-increasing after every layer solve (delta = 0 is feasible, so
      fid_new + lam ||delta||^2 <= fid_old).

Layer solves use the exact modewise eigh solve with no extra eps shifts
(lam > 0 guarantees well-posedness), so measured violations reflect only
float64 roundoff.

Instances: L in {4, 16, 64, 128}, lam in {1e-6, 1e-4, 1e-2}, init in
{xavier, ginibre_sn1, identity}, orthogonal targets, 2 seeds; d=16; 8 sweeps.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x4_monotone \
      --out_dir outputs/oplevel_rebuttal/e4_monotone
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from operator_level_optimization.scripts.train.deep_linear_compare import (
    _product_excluding,
    compose_operator,
    ginibre_sn1,
    xavier_gaussian,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify
from operator_level_optimization.scripts.utils.plotting import save_fig


def solve_sub_exact(
    M: torch.Tensor, N: torch.Tensor, G: torch.Tensor, lam: float
) -> torch.Tensor:
    """Unique minimizer of the strongly convex sub-problem: M d N + lam d = G.

    M, N are rescaled to O(1) before eigh (exact reparametrization: the
    denominators are rescaled back), which avoids eigh failures on deeply
    collapsed contexts whose entries underflow toward subnormals.
    """
    if not bool(torch.isfinite(M).all() and torch.isfinite(N).all() and torch.isfinite(G).all()):
        raise FloatingPointError("non-finite context/gradient (solver breakdown)")
    s_m = float(M.abs().max().clamp(min=1e-300).item())
    s_n = float(N.abs().max().clamp(min=1e-300).item())

    def _eigh_robust(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return torch.linalg.eigh(S)
        except torch.linalg.LinAlgError:
            # SVD route for symmetric PSD matrices where eigh fails to converge.
            U, sv, _ = torch.linalg.svd(S)
            return sv.flip(0), U.flip(1)

    evals_m, U_m = _eigh_robust(M / s_m)
    evals_n, U_n = _eigh_robust(N / s_n)
    G_t = U_m.T @ G @ U_n
    denom = (
        (s_m * evals_m.clamp(min=0.0)).unsqueeze(1)
        * (s_n * evals_n.clamp(min=0.0)).unsqueeze(0)
        + lam
    )
    return U_m @ (G_t / denom) @ U_n.T


def make_instance(
    depth: int, d: int, init: str, seed: int, device: torch.device, dtype: torch.dtype
) -> tuple[list[torch.Tensor], torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    weights = []
    for _ in range(depth):
        if init == "identity":
            weights.append(torch.eye(d, device=device, dtype=dtype))
        elif init == "xavier":
            weights.append(xavier_gaussian((d, d), device=device, dtype=dtype, g=g))
        else:
            weights.append(ginibre_sn1((d, d), device=device, dtype=dtype, g=g))
    q1, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
    q2, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
    p_tgt = q1 @ q2.T
    return weights, p_tgt


def run_instance(
    variant: str,  # "recentred" | "proximal"
    weights0: list[torch.Tensor],
    p_tgt: torch.Tensor,
    lam: float,
    n_sweeps: int,
) -> dict:
    """Track the relevant objective after every layer solve."""
    w = [x.clone() for x in weights0]
    depth = len(w)

    def fid() -> float:
        return float((compose_operator(w) - p_tgt).norm().item() ** 2)

    def penalty() -> float:
        return float(sum(((wi - w0i).norm().item() ** 2) for wi, w0i in zip(w, weights0)))

    def objective() -> float:
        return fid() + lam * penalty() if variant == "recentred" else fid()

    vals = [objective()]
    failed: str | None = None
    for _ in range(n_sweeps):
        for k in range(depth - 1, -1, -1):
            A_k, B_k = _product_excluding(w, k)
            R_k = p_tgt - A_k @ w[k] @ B_k
            M = A_k.T @ A_k
            N = B_k @ B_k.T
            G = A_k.T @ R_k @ B_k.T
            if variant == "recentred":
                G = G - lam * (w[k] - weights0[k])
            try:
                dW = solve_sub_exact(M, N, G, lam)
            except (FloatingPointError, torch.linalg.LinAlgError) as exc:
                failed = str(exc)
                break
            w[k] = w[k] + dW
            new_val = objective()
            if not np.isfinite(new_val):
                failed = "non-finite objective (solver breakdown)"
                break
            vals.append(new_val)
        if failed is not None:
            break

    vals_a = np.array(vals)
    # Per-solve increase relative to the INITIAL objective scale; positive =
    # violation. (Relative-to-previous blows up meaninglessly once the
    # objective reaches the float64 noise floor ~1e-30 x initial.)
    rel_inc = (vals_a[1:] - vals_a[:-1]) / max(vals_a[0], 1e-300)
    return {
        "objective_per_solve": vals_a.tolist(),
        "max_rel_increase": float(rel_inc.max()) if rel_inc.size else 0.0,
        "n_solves": int(len(vals) - 1),
        "final_over_initial": float(vals_a[-1] / max(vals_a[0], 1e-300)),
        "failed": failed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 16, 64, 128])
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2])
    ap.add_argument("--inits", nargs="+", default=["xavier", "ginibre", "identity"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n_sweeps", type=int, default=8)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    dtype = torch.float64

    records = []
    worst = {"recentred": -np.inf, "proximal": -np.inf}
    n_instances = 0
    for depth, lam, init, seed in itertools.product(
        args.depths, args.lams, args.inits, args.seeds
    ):
        weights0, p_tgt = make_instance(depth, args.d, init, seed, device, dtype)
        n_instances += 1
        for variant in ("recentred", "proximal"):
            res = run_instance(variant, weights0, p_tgt, lam, args.n_sweeps)
            rec = {
                "variant": variant,
                "depth": depth,
                "lam": lam,
                "init": init,
                "seed": seed,
                "max_rel_increase": res["max_rel_increase"],
                "final_over_initial": res["final_over_initial"],
                "n_solves": res["n_solves"],
                "failed": res["failed"],
            }
            # Keep full traces only for representative instances (plot size).
            if seed == args.seeds[0] and lam == 1e-4:
                rec["objective_per_solve"] = res["objective_per_solve"]
            records.append(rec)
            worst[variant] = max(worst[variant], res["max_rel_increase"])
        print(
            f"[e4] L={depth} lam={lam} init={init} seed={seed}: "
            f"max_rel_inc recentred={records[-2]['max_rel_increase']:.2e} "
            f"proximal={records[-1]['max_rel_increase']:.2e}",
            flush=True,
        )

    n_variant_runs = len(records) // 2

    def classify(r: dict) -> str:
        """Three observed regimes of the raw (unstabilized) ALS solve.

        monotone            — descent to float64 roundoff;
        stalled_zero        — contexts underflowed to 0, updates exactly 0
                              (fully collapsed regime: solve returns G/lam = 0);
        numerical_breakdown — collapsed contexts with small lam: the modewise
                              system's conditioning (~sigma^4/lam) exceeds
                              float64 and the computed 'solve' explodes.
        """
        if r["failed"] or r["max_rel_increase"] > 1e-6:
            return "numerical_breakdown"
        if r["max_rel_increase"] == 0.0 and r["final_over_initial"] == 1.0:
            return "stalled_zero"
        return "monotone"

    by_class: dict[str, dict[str, list]] = {}
    for r in records:
        cls = classify(r)
        r["regime"] = cls
        by_class.setdefault(r["variant"], {}).setdefault(cls, []).append(
            {k: r[k] for k in ("depth", "lam", "init", "seed", "max_rel_increase")}
        )
    summary = {
        "n_instances": n_instances,
        "n_runs_per_variant": n_variant_runs,
        "counts_by_regime": {
            v: {cls: len(items) for cls, items in classes.items()}
            for v, classes in by_class.items()
        },
        "max_rel_increase_within_monotone": {
            v: max(
                (x["max_rel_increase"] for x in classes.get("monotone", [])),
                default=None,
            )
            for v, classes in by_class.items()
        },
        "breakdown_instances": {
            v: [
                (x["depth"], x["lam"], x["init"], x["seed"])
                for x in classes.get("numerical_breakdown", [])
            ]
            for v, classes in by_class.items()
        },
        "stalled_instances": {
            v: [
                (x["depth"], x["lam"], x["init"], x["seed"])
                for x in classes.get("stalled_zero", [])
            ]
            for v, classes in by_class.items()
        },
    }

    # Figure: objective vs cumulative layer solves for representative instances.
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    for ax, variant, title in (
        (axes[0], "recentred", "Recentred ALS — objective (3)"),
        (axes[1], "proximal", "Proximal ALS (paper solver) — fidelity term"),
    ):
        for rec in records:
            if rec["variant"] != variant or "objective_per_solve" not in rec:
                continue
            vals = np.array(rec["objective_per_solve"])
            ax.plot(
                np.arange(len(vals)),
                np.clip(vals / vals[0], 1e-30, None),
                linewidth=1.2,
                alpha=0.8,
                label=f"L={rec['depth']}, {rec['init']}",
            )
        ax.set_yscale("log")
        # Clamp: numerical-breakdown traces (collapsed regime) exit the top;
        # without the clamp they stretch the axis to ~1e180 and hide the
        # monotone traces.
        ax.set_ylim(1e-32, 1e4)
        ax.set_xlabel("Cumulative layer solves")
        ax.set_ylabel("Objective / initial value")
        ax.set_title(title + rf"  ($\lambda=10^{{-4}}$)")
        ax.text(
            0.98, 0.96,
            "traces exiting top =\nnumerical breakdown (collapsed contexts)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="gray",
        )
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="center right")
    save_fig(fig, out_dir / "fig_e4_monotone.png", dpi=180, bbox_inches="tight")

    payload = {"records": records, "summary": summary}
    (out_dir / "results.json").write_text(json.dumps(jsonify(payload), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
