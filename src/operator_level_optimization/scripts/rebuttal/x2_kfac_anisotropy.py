"""E2 — K-FAC anisotropy: testing the Eq. (8) gap prediction.

Rebuttal experiment for NGj5 Q3 / AC point 1. The paper identifies K-FAC as the
separable approximation to the exact operator-projection modewise solve, exact
in basis (V_A, U_B) under isotropic data, with per-mode denominator gap
(sigma_A^2+lam)(sigma_B^2+lam) - (sigma_A^2 sigma_B^2 + lam)
= lam (sigma_A^2 + sigma_B^2 + lam - 1)          [Eq. 8].

Two measurements, all analytic (population covariances, no sampling noise):

1. Isotropy check (Sigma_x = I): the measured per-mode inverse-filter
   discrepancy between the K-FAC update and the exact modewise solve must equal
   the predicted Eq.-8 gap to machine precision (bases coincide).
2. Anisotropy sweep: Sigma_x = Q D Q^T with cond(D) from 1 to 1e4 (log-spaced
   eigenvalues, trace normalized to d), Sigma_g = I. Per-layer K-FAC update
   dW_k = (M_k+lam)^{-1} G_{W_k} (Ahat_k+lam)^{-1}, Ahat_k = B_k Sigma_x B_k^T,
   vs exact solve of M_k dW N_k + lam dW = G_{W_k}. The per-layer error is
   decomposed into
     err_total(cond)  — full K-FAC vs exact solve;
     err_denom(cond)  — "aligned-basis" K-FAC (Ahat eigenvalues taken as
                        Rayleigh quotients in the ALS basis U_B) vs exact:
                        the part explained by the separable-denominator gap
                        alone, with no basis rotation.
   At cond=1 the two coincide (bases identical, Eq. 8 regime); the wedge
   between them as cond grows is the anisotropy-induced basis-rotation error —
   the quantity the framework predicts should appear once isotropy breaks.
   The eigenbasis rotation of Ahat_k away from U_B is reported directly as well.

Setup: deep linear d=16, L in {8, 32}, weight states: Haar-orthogonal (clean
regime: all context sigma = 1, isotropic baseline error ~ lam) and
Ginibre(sn=1) (generic spread spectra), orthogonal target defining
G = (2/d) (P - P*) Sigma_x.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x2_kfac_anisotropy \
      --out_dir outputs/oplevel_rebuttal/e2_kfac_anisotropy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from operator_level_optimization.scripts.train.deep_linear_compare import (
    _product_excluding,
    compose_operator,
    ginibre_sn1,
)
from operator_level_optimization.scripts.utils.io import get_device, jsonify
from operator_level_optimization.scripts.utils.plotting import save_fig


def make_state(
    depth: int, d: int, init: str, seed: int, device: torch.device, dtype: torch.dtype
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Weight stack (haar or ginibre) + orthogonal target P*."""
    g = torch.Generator(device=device).manual_seed(seed)
    weights = []
    for _ in range(depth):
        if init == "haar":
            q, _ = torch.linalg.qr(
                torch.randn((d, d), device=device, dtype=dtype, generator=g)
            )
            weights.append(q)
        else:
            weights.append(ginibre_sn1((d, d), device=device, dtype=dtype, g=g))
    q1, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
    q2, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
    return weights, q1 @ q2.T


def make_sigma_x(
    d: int, cond: float, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """SPD covariance with log-spaced spectrum, cond(Sigma)=cond, trace = d."""
    g = torch.Generator(device=device).manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
    evals = torch.logspace(
        0, float(np.log10(cond)), d, device=device, dtype=dtype
    )
    evals = evals * (d / evals.sum())
    return q @ torch.diag(evals) @ q.T


def exact_modewise_solve(
    M: torch.Tensor, N: torch.Tensor, G_w: torch.Tensor, lam: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve M dW N + lam dW = G_w. Returns (dW, evals_M, U_M, evals_N, U_N)."""
    evals_m, U_m = torch.linalg.eigh(M)
    evals_n, U_n = torch.linalg.eigh(N)
    G_t = U_m.T @ G_w @ U_n
    denom = evals_m.unsqueeze(1) * evals_n.unsqueeze(0) + lam
    dW = U_m @ (G_t / denom) @ U_n.T
    return dW, evals_m, U_m, evals_n, U_n


def kfac_update(
    Ghat: torch.Tensor, Ahat: torch.Tensor, G_w: torch.Tensor, lam: float
) -> torch.Tensor:
    """K-FAC layer update (Ghat+lam)^{-1} G_w (Ahat+lam)^{-1} (damping lam)."""
    d = Ahat.shape[0]
    eye = torch.eye(d, device=Ahat.device, dtype=Ahat.dtype)
    return torch.linalg.solve(Ghat + lam * eye, torch.linalg.solve((Ahat + lam * eye).T, G_w.T).T)


def cos_frob(a: torch.Tensor, b: torch.Tensor) -> float:
    num = float((a * b).sum().item())
    den = float((a.norm() * b.norm()).clamp(min=1e-300).item())
    return num / den


def basis_misalignment(U_ref: torch.Tensor, S: torch.Tensor) -> float:
    """1 - mean_i max_j <u_i, v_j>^2 where v = eigenvectors of SPD matrix S.

    0 when the eigenbasis of S matches U_ref up to permutation/sign; grows as
    the bases rotate apart.
    """
    _, V = torch.linalg.eigh(S)
    O = (U_ref.T @ V) ** 2  # overlap matrix, rows sum to 1
    return float((1.0 - O.max(dim=1).values.mean()).item())


def gap_check_isotropic(
    weights: list[torch.Tensor], lam: float
) -> dict:
    """Per-mode predicted vs measured K-FAC/exact inverse-filter gap at Sigma_x=I."""
    predicted: list[float] = []
    measured: list[float] = []
    for k in range(len(weights)):
        A_k, B_k = _product_excluding(weights, k)
        M = A_k.T @ A_k
        N = B_k @ B_k.T
        evals_m, U_m = torch.linalg.eigh(M)
        evals_n, U_n = torch.linalg.eigh(N)
        # Random probe gradient; measurement is per-mode so probe-independent.
        G_w = torch.randn_like(weights[k])
        # Exact modewise: filter denominator sigma_A^2 sigma_B^2 + lam per mode.
        dW_exact, *_ = exact_modewise_solve(M, N, G_w, lam)
        # K-FAC with isotropic data: Ahat = N, Ghat = M.
        dW_kfac = kfac_update(M, N, G_w, lam)
        G_t = U_m.T @ G_w @ U_n
        F_exact = (U_m.T @ dW_exact @ U_n) / G_t  # 1/(s_A^2 s_B^2 + lam)
        F_kfac = (U_m.T @ dW_kfac @ U_n) / G_t  # 1/((s_A^2+lam)(s_B^2+lam))
        meas_gap = 1.0 / F_kfac - 1.0 / F_exact
        pred_gap = lam * (
            evals_m.unsqueeze(1) + evals_n.unsqueeze(0) + lam - 1.0
        )
        predicted.extend(pred_gap.flatten().tolist())
        measured.extend(meas_gap.flatten().tolist())
    predicted_a = np.array(predicted)
    measured_a = np.array(measured)
    scale = np.abs(predicted_a).max() if np.abs(predicted_a).max() > 0 else 1.0
    max_rel_dev = float(np.max(np.abs(measured_a - predicted_a)) / scale)
    return {
        "predicted": predicted_a.tolist(),
        "measured": measured_a.tolist(),
        "max_rel_deviation": max_rel_dev,
    }


def anisotropy_sweep(
    weights: list[torch.Tensor],
    p_star: torch.Tensor,
    lam: float,
    conds: list[float],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Per-layer K-FAC-vs-exact error decomposition as cond(Sigma_x) grows."""
    depth = len(weights)
    d = weights[0].shape[0]
    P = compose_operator(weights)
    ctx = []
    for k in range(depth):
        A_k, B_k = _product_excluding(weights, k)
        ctx.append((A_k, B_k, A_k.T @ A_k, B_k @ B_k.T))

    rows = []
    for cond in conds:
        sigma_x = make_sigma_x(d, cond, seed + 1, device, dtype)
        # Population MSE operator gradient under anisotropic inputs.
        G = (2.0 / d) * (P - p_star) @ sigma_x
        err_total = []
        err_denom = []
        basis_mis = []
        for k in range(depth):
            A_k, B_k, M, N = ctx[k]
            G_w = A_k.T @ G @ B_k.T
            Ahat = B_k @ sigma_x @ B_k.T
            # Exact modewise projection solve (data-independent contexts).
            dW_e, evals_m, U_m, _, U_n = exact_modewise_solve(M, N, G_w, lam)
            # Full K-FAC with the data-dependent input-covariance factor.
            dW_kfac = kfac_update(M, Ahat, G_w, lam)
            # Aligned-basis K-FAC: separable denominator with Ahat's Rayleigh
            # quotients in the ALS basis U_B — the deviation Eq. 8 accounts
            # for, with the basis rotation removed.
            alpha = torch.diagonal(U_n.T @ Ahat @ U_n)
            G_t = U_m.T @ G_w @ U_n
            denom_sep = (evals_m + lam).unsqueeze(1) * (alpha + lam).unsqueeze(0)
            dW_aligned = U_m @ (G_t / denom_sep) @ U_n.T
            dnorm = dW_e.norm().clamp(min=1e-300)
            err_total.append(float(((dW_kfac - dW_e).norm() / dnorm).item()))
            err_denom.append(float(((dW_aligned - dW_e).norm() / dnorm).item()))
            basis_mis.append(basis_misalignment(U_n, Ahat))
        rows.append(
            {
                "cond": cond,
                "err_total_mean": float(np.mean(err_total)),
                "err_denom_mean": float(np.mean(err_denom)),
                "err_basis_gap_mean": float(np.mean(np.array(err_total) - np.array(err_denom))),
                "basis_misalignment_mean": float(np.mean(basis_mis)),
                "basis_misalignment_max": float(np.max(basis_mis)),
            }
        )
    return {"lam": lam, "rows": rows}


def plot_gap_check(out_dir: Path, checks: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    markers = {8: "o", 32: "s"}
    colors = {1e-4: "#9467bd", 1e-2: "#2ca02c", 1e-1: "#d62728"}
    for (depth, lam), chk in checks.items():
        pred = np.abs(np.array(chk["predicted"]))
        meas = np.abs(np.array(chk["measured"]))
        ax.scatter(
            pred,
            meas,
            s=8,
            alpha=0.35,
            marker=markers.get(depth, "o"),
            color=colors.get(lam, None),
            label=rf"L={depth}, $\lambda$={lam:g}",
        )
    lims = ax.get_xlim()
    grid = np.logspace(-8, 3, 50)
    ax.plot(grid, grid, color="k", linewidth=0.8, linestyle="--", label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Predicted per-mode gap  $\lambda(\sigma_A^2+\sigma_B^2+\lambda-1)$")
    ax.set_ylabel("Measured inverse-filter gap (K-FAC vs exact)")
    ax.set_title("Eq. (8) gap prediction at isotropy — all modes, all layers")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_fig(fig, out_dir / "fig_e2_gap_check.png", dpi=180, bbox_inches="tight")


def plot_anisotropy(out_dir: Path, sweeps: dict, inits: list[str]) -> None:
    colors = {1e-4: "#9467bd", 1e-2: "#2ca02c", 1e-1: "#d62728"}
    ls_depth = {8: "-", 32: "--"}
    fig, axes = plt.subplots(1, len(inits), figsize=(6.3 * len(inits), 4.6), squeeze=False)
    for ax, init in zip(axes[0], inits):
        for (init_k, depth, lam), sweep in sweeps.items():
            if init_k != init:
                continue
            conds = [r["cond"] for r in sweep["rows"]]
            ax.plot(
                conds,
                [max(r["err_total_mean"], 1e-18) for r in sweep["rows"]],
                color=colors.get(lam),
                linestyle=ls_depth.get(depth, "-"),
                marker="o",
                markersize=3.5,
                label=rf"total, L={depth}, $\lambda$={lam:g}",
            )
            ax.plot(
                conds,
                [max(r["err_denom_mean"], 1e-18) for r in sweep["rows"]],
                color=colors.get(lam),
                linestyle=ls_depth.get(depth, "-"),
                marker="x",
                markersize=4.0,
                alpha=0.45,
                label=rf"denom.-only, L={depth}, $\lambda$={lam:g}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"cond($\Sigma_x$)")
        ax.set_ylabel(r"per-layer $\|\Delta W^{KFAC} - \Delta W^{proj}\| / \|\Delta W^{proj}\|$")
        ax.set_title(f"{init} weight state — K-FAC deviation vs anisotropy")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
    save_fig(fig, out_dir / "fig_e2_anisotropy.png", dpi=180, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--inits", nargs="+", default=["haar", "ginibre"])
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-4, 1e-2, 1e-1])
    ap.add_argument(
        "--conds", type=float, nargs="+", default=list(np.logspace(0, 4, 9))
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    dtype = torch.float64

    checks: dict = {}
    sweeps: dict = {}
    summary: dict = {"gap_check_max_rel_deviation": {}, "anisotropy": {}}
    for init in args.inits:
        for depth in args.depths:
            weights, p_star = make_state(depth, args.d, init, args.seed, device, dtype)
            for lam in args.lams:
                key = (init, depth, lam)
                # Gap check on the spread-spectrum (ginibre) state: haar has all
                # sigma^2 = 1 so its per-mode gaps collapse to a single value.
                gap_init = "ginibre" if "ginibre" in args.inits else args.inits[0]
                if init == gap_init:
                    print(f"[e2] gap check {init} L={depth} lam={lam}", flush=True)
                    chk = gap_check_isotropic(weights, lam)
                    checks[(depth, lam)] = chk
                    summary["gap_check_max_rel_deviation"][f"L{depth}_lam{lam:g}"] = chk[
                        "max_rel_deviation"
                    ]
                    print(
                        f"[e2]   max rel deviation predicted vs measured: "
                        f"{chk['max_rel_deviation']:.3e}",
                        flush=True,
                    )
                print(f"[e2] anisotropy sweep {init} L={depth} lam={lam}", flush=True)
                sw = anisotropy_sweep(
                    weights, p_star, lam, list(args.conds), args.seed, device, dtype
                )
                sweeps[key] = sw
                r0, r_last = sw["rows"][0], sw["rows"][-1]
                summary["anisotropy"][f"{init}_L{depth}_lam{lam:g}"] = {
                    "err_total_iso": r0["err_total_mean"],
                    "err_total_maxcond": r_last["err_total_mean"],
                    "err_denom_iso": r0["err_denom_mean"],
                    "err_denom_maxcond": r_last["err_denom_mean"],
                    "err_basis_gap_maxcond": r_last["err_basis_gap_mean"],
                    "basis_misalignment_maxcond": r_last["basis_misalignment_mean"],
                }
                print(
                    f"[e2]   err_total: iso={r0['err_total_mean']:.3e} -> "
                    f"cond1e4={r_last['err_total_mean']:.3e}; "
                    f"err_denom: {r0['err_denom_mean']:.3e} -> {r_last['err_denom_mean']:.3e}; "
                    f"basis_mis(maxcond)={r_last['basis_misalignment_mean']:.3f}",
                    flush=True,
                )

    payload = {
        "gap_checks": {f"L{d}_lam{l:g}": c for (d, l), c in checks.items()},
        "anisotropy_sweeps": {f"{i}_L{d}_lam{l:g}": s for (i, d, l), s in sweeps.items()},
        "summary": summary,
    }
    (out_dir / "results.json").write_text(json.dumps(jsonify(payload), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    plot_gap_check(out_dir, checks)
    plot_anisotropy(out_dir, sweeps, list(args.inits))
    print(f"[done] wrote {out_dir}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
