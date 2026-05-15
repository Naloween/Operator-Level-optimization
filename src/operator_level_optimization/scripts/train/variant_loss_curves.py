"""Train appendix-style variants and save loss / operator-error curves.

Produces Matplotlib figures under ``--out_dir`` (default: gitignored ``outputs/``)
plus a JSON bundle for replotting or downstream tables, and ``config.json`` with
full reproduction metadata (CLI, environment, data, architectures, and every
optimizer keyword dict).

Examples:

    python -m operator_level_optimization.scripts.train.variant_loss_curves \\
        --out_dir outputs/variant_curves/run01

    python -m operator_level_optimization.scripts.train.variant_loss_curves \\
        --out_dir outputs/variant_curves/quick --steps_deeplinear 80 --steps_mlp 60 --steps_fgln 40
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from operator_level_optimization.core.optim.kfac import KFAC
from operator_level_optimization.core.optim.operator import OperatorLevelMLP
from operator_level_optimization.models.fgln import FGLN, MaskedOperatorALS, compute_P_fgln, init_weights
from operator_level_optimization.scripts.train.deep_linear_compare import run_method as deep_linear_run
from operator_level_optimization.scripts.utils.io import get_device, jsonify
from operator_level_optimization.scripts.utils.plotting import annotate_clip, clip_for_plot, save_fig
from operator_level_optimization.scripts.utils.tasks import make_mlp, make_tiny_mlp
from operator_level_optimization.scripts.utils.training import run_mlp_loop


def _mlp_architecture() -> dict[str, Any]:
    """Fixed synthetic MLP used for variant comparison (must match ``make_tiny_mlp``)."""
    return {
        "description": "bias-free ReLU MLP for synthetic cross-entropy task",
        "layers": [
            {"type": "Linear", "in_features": 24, "out_features": 8, "bias": False},
            {"type": "ReLU", "inplace": False},
            {"type": "Linear", "in_features": 8, "out_features": 5, "bias": False},
        ],
        "num_classes": 5,
        "total_trainable_params": sum(p.numel() for p in make_tiny_mlp().parameters()),
    }


def _operator_level_mlp_kwargs(**overrides: Any) -> dict[str, Any]:
    """Full keyword dict for ``OperatorLevelMLP`` (defaults + overrides) for reproducibility."""
    sig = inspect.signature(OperatorLevelMLP.__init__)
    out: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name in ("self", "params"):
            continue
        if p.default is not inspect.Parameter.empty:
            out[name] = p.default
    out.update(overrides)
    return out


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    flat = torch.cat([v.detach().reshape(-1).cpu().to(torch.float64) for v in state_dict.values()])
    return hashlib.sha256(flat.numpy().tobytes()).hexdigest()


# --- Deep linear -----------------------------------------------------------------

def curves_deep_linear(
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    depth: int,
    d: int,
    n: int,
    seed: int,
    plot_y_max: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dtype = torch.float64
    common = dict(
        depth=depth,
        d=d,
        n=n,
        steps=steps,
        seed=seed,
        device=device,
        dtype=dtype,
        target_mode="mse_grad",
        target_kind="orth",
        init_mode="xavier",
        als_lam=1e-4,
        als_sweeps=50,
        dc_lam=1e-4,
        dc_alt=3,
        dc_init_mode="plain",
        dc_init_scale=1.0,
        spec_every=1_000_000,
        spec_steps=None,
        early_stop_patience=None,
        early_stop_rel_tol=1e-4,
        early_stop_abs_tol=1e-10,
        early_stop_min_steps=10_000,
    )
    configs: list[tuple[str, float, dict[str, Any]]] = [
        ("als_exact", 1.0, {}),
        ("dc", 1.0, {}),
    ]
    labels = [
        "ALS-exact",
        "D&C",
    ]
    series: list[dict[str, Any]] = []
    fig_mse, ax_mse = plt.subplots(figsize=(8.0, 4.8))
    fig_rel, ax_rel = plt.subplots(figsize=(8.0, 4.8))
    colors = ["#9467bd", "#17becf", "#bcbd22"]
    slugs = ["als_exact", "dc"]
    clipped_mse_any = False
    clipped_rel_any = False
    for slug, label, (method, lr, extra), c in zip(slugs, labels, configs, colors):
        kw = {**common, **extra, "lr": lr}
        out = deep_linear_run(method=method, **kw)
        hist = out["history"]
        ts = [h[0] for h in hist]
        rels = [h[1] for h in hist]
        mses = [h[2] for h in hist]
        series.append({"label": label, "method": method, "lr": lr, "extra": extra, "history": hist})
        y_mse, cm = clip_for_plot(mses, plot_y_max)
        y_rel, cr = clip_for_plot(rels, plot_y_max)
        clipped_mse_any = clipped_mse_any or cm
        clipped_rel_any = clipped_rel_any or cr
        ax_mse.plot(ts, y_mse, label=label, color=c, linewidth=2.0)
        ax_rel.plot(ts, y_rel, label=label, color=c, linewidth=2.0)
        fig_i, ax_i = plt.subplots(figsize=(6.5, 4.0))
        ax_i.plot(ts, y_mse, color=c, lw=2.0)
        ax_i.set_xlabel("Step")
        ax_i.set_ylabel("Train MSE")
        ax_i.set_yscale("log")
        ax_i.set_title(label)
        ax_i.grid(True, alpha=0.25)
        annotate_clip(ax_i, cm, plot_y_max)
        save_fig(fig_i, out_dir / f"deep_linear_mse_{slug}.png")

    for ax, ylabel, fname, clipped_any in (
        (ax_mse, "Train MSE", "deep_linear_train_mse.png", clipped_mse_any),
        (ax_rel, r"$\|P-P^\star\|_F / \|P^\star\|_F$", "deep_linear_rel_operator_error.png", clipped_rel_any),
    ):
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, loc="best")
        annotate_clip(ax, clipped_any, plot_y_max)
        save_fig(ax.figure, out_dir / fname)

    per_method_cfg: list[dict[str, Any]] = []
    for slug, label, (method, lr, extra) in zip(slugs, labels, configs):
        kw = {**common, **extra, "lr": lr}
        per_method_cfg.append(
            {
                "slug": slug,
                "plot_label": label,
                "method": method,
                "operator_step_lr": lr,
                "run_kwargs_overrides_vs_shared": jsonify(extra),
                "merged_run_kwargs_passed_to_deep_linear_run": jsonify(kw),
            }
        )

    cfg_section = {
        "scenario": "deep_linear_teacher_student",
        "description": (
            "Synthetic Gaussian inputs, orthogonal P*, same protocol as "
            "deep_linear_compare.run_method (mse_grad target)."
        ),
        "model": {
            "class": "DeepLinearModel (from deep_linear_compare)",
            "depth": depth,
            "hidden_width_d": d,
            "layer_type": "nn.Linear(d,d,bias=False) chain",
        },
        "data": {
            "n_samples": n,
            "batch_design": "Full-batch reuses same X each step (see deep_linear_compare).",
            "seed": seed,
            "target_P_star": {"kind": "orth", "shape": [d, d]},
        },
        "dtype": str(dtype),
        "device": str(device),
        "plot_y_max": plot_y_max,
        "shared_run_kwargs": jsonify(common),
        "per_method": per_method_cfg,
    }

    return {"deep_linear": series}, {"deep_linear": cfg_section}


# --- MLP (shared init, synthetic CE) ----------------------------------------------

def _mlp_variants() -> list[dict[str, Any]]:
    """Each entry: ``key`` + full ``operator_level_mlp_kwargs`` for ``OperatorLevelMLP``."""
    return [
        {"key": "mlp_linearized_exact", "kwargs": _operator_level_mlp_kwargs(
            lr=0.08, lam=1e-3, approximation="exact", momentum=0.0,
        )},
        {"key": "mlp_block_diagonal", "kwargs": _operator_level_mlp_kwargs(
            lr=0.08, lam=1e-3, approximation="block_diagonal", momentum=0.0,
        )},
        {"key": "mlp_block_diagonal_adaptive_lam", "kwargs": _operator_level_mlp_kwargs(
            lr=0.08, lam=None, lam_alpha=0.5, approximation="block_diagonal", momentum=0.0,
        )},
        {"key": "mlp_operator_kfac", "kwargs": _operator_level_mlp_kwargs(
            lr=0.08, lam=1e-3, approximation="operator_kfac", momentum=0.0,
        )},
        {"key": "mlp_secant", "kwargs": _operator_level_mlp_kwargs(
            lr=0.01, lam=1e-3, approximation="secant", momentum=0.0,
        )},
        # secant_r (rank-r) omitted: storing A_l, B_l per layer for rank>1 is equivalent
        # to using the exact context, removing the point of the secant approximation.
        {"key": "mlp_cg_fallback_bd", "kwargs": _operator_level_mlp_kwargs(
            lr=0.08, lam=1e-3, approximation="cg", momentum=0.0,
        )},
        {"key": "mlp_als_mn", "kwargs": _operator_level_mlp_kwargs(
            lr=0.35,
            lam=1e-3,
            approximation="als",
            als_layer_solve="mn",
            n_sweeps=3,
            als_reverse_sweep=True,
            momentum=0.0,
            als_gateperm_warmstart=True,
            als_gateperm_warmstart_once=True,
            als_lam_anchor_post_warmstart=False,
        )},
        {"key": "mlp_als_exact_layers", "kwargs": _operator_level_mlp_kwargs(
            lr=0.35,
            lam=1e-3,
            approximation="als",
            als_layer_solve="exact",
            n_sweeps=3,
            als_reverse_sweep=True,
            momentum=0.0,
            als_gateperm_warmstart=True,
            als_gateperm_warmstart_once=True,
            als_lam_anchor_post_warmstart=False,
        )},
        {"key": "mlp_als_per_sample", "kwargs": _operator_level_mlp_kwargs(
            lr=0.35,
            lam=1e-3,
            approximation="als",
            als_layer_solve="per_sample",
            n_sweeps=3,
            als_reverse_sweep=True,
            momentum=0.0,
            als_gateperm_warmstart=True,
            als_gateperm_warmstart_once=True,
            als_lam_anchor_post_warmstart=False,
        )},
        {"key": "mlp_als_adaptive_lam", "kwargs": _operator_level_mlp_kwargs(
            lr=0.35,
            lam=None,
            lam_alpha=0.5,
            approximation="als",
            als_layer_solve="mn",
            n_sweeps=3,
            als_reverse_sweep=True,
            momentum=0.0,
            als_gateperm_warmstart=True,
            als_gateperm_warmstart_once=True,
            als_lam_anchor_post_warmstart=False,
        )},
        # App. B.5: rank-1 secant curvature + exact backprop gradient direction
        {"key": "mlp_secant_grad_exact", "kwargs": _operator_level_mlp_kwargs(
            lr=0.01, lam=1e-3, approximation="secant_grad_exact", momentum=0.0,
        )},
        # App. B.4: D&C failure on MLP — per-sample ALS averaged across samples
        {"key": "mlp_dc_mlp", "kwargs": _operator_level_mlp_kwargs(
            lr=0.35,
            lam=1e-3,
            approximation="dc_mlp",
            n_sweeps=3,
            momentum=0.0,
        )},
    ]


def curves_mlp(
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    batch: int,
    seed: int,
    plot_y_max: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dtype = torch.float64
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, 24, device=device, dtype=dtype, generator=g)
    y = torch.randint(0, 5, (batch,), device=device, dtype=torch.long)

    torch.manual_seed(seed)
    template = make_tiny_mlp().to(device=device, dtype=dtype)
    for m in template.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.05)
    state0 = {k: v.clone() for k, v in template.state_dict().items()}

    crit = nn.CrossEntropyLoss()
    variants = _mlp_variants()
    cmap = plt.cm.tab20(np.linspace(0, 1, len(variants)))

    bundle: list[dict[str, Any]] = []
    fig_lin, ax_lin = plt.subplots(figsize=(10.0, 5.5))
    fig_log, ax_log = plt.subplots(figsize=(10.0, 5.5))
    clipped_lin_any = False
    clipped_log_any = False

    for i, entry in enumerate(variants):
        key = entry["key"]
        kwargs = entry["kwargs"]
        model = make_tiny_mlp().to(device=device, dtype=dtype)
        model.load_state_dict(state0)
        opt = OperatorLevelMLP(list(model.parameters()), **kwargs)
        opt.attach_hooks(model)

        hist = run_mlp_loop(model, opt, lambda: crit(model(x), y), steps)
        losses = [v for _, v in hist]
        bundle.append({"key": key, "losses": losses})
        ts = [h[0] for h in hist]
        label = key.replace("mlp_", "").replace("_", " ")
        color = cmap[i % len(cmap)]
        y_plot, c_any = clip_for_plot(losses, plot_y_max)
        clipped_lin_any = clipped_lin_any or c_any
        clipped_log_any = clipped_log_any or c_any
        ax_lin.plot(ts, y_plot, label=label, color=color, linewidth=1.8)
        y_log = np.maximum(y_plot, 1e-12)
        ax_log.plot(ts, y_log, label=label, color=color, linewidth=1.8)

        fig_i, ax_i = plt.subplots(figsize=(6.5, 4.0))
        ax_i.plot(ts, y_plot, color="#1f77b4", linewidth=2.0)
        ax_i.set_xlabel("Step")
        ax_i.set_ylabel("Cross-entropy loss")
        ax_i.set_title(label)
        ax_i.grid(True, alpha=0.25)
        annotate_clip(ax_i, c_any, plot_y_max)
        save_fig(fig_i, out_dir / f"loss_{key}.png")

    for ax, fname, clipped_any in (
        (ax_lin, "mlp_cross_entropy_linear.png", clipped_lin_any),
        (ax_log, "mlp_cross_entropy_logy.png", clipped_log_any),
    ):
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-entropy loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best", fontsize=8, ncol=2)
        if ax is ax_log:
            ax.set_yscale("log")
        annotate_clip(ax, clipped_any, plot_y_max)
        save_fig(ax.figure, out_dir / fname)

    cfg_section = {
        "scenario": "mlp_synthetic_cross_entropy",
        "description": (
            "Single fixed mini-batch, identical initial weights per variant "
            "(cloned state_dict). Cross-entropy; OperatorLevelMLP with hooks."
        ),
        "architecture": _mlp_architecture(),
        "dtype": str(dtype),
        "device": str(device),
        "data": {
            "batch_size": batch,
            "input_shape": [batch, 24],
            "labels_shape": [batch],
            "num_classes": 5,
            "x_generation": f"torch.randn(..., generator=torch.Generator(device).manual_seed({seed}))",
            "y_generation": f"torch.randint(0, 5, ..., device, dtype=long) using same generator after x",
            "seed": seed,
        },
        "weight_init": {
            "protocol": "torch.manual_seed(seed); nn.init.normal_(each Linear.weight, std=0.05)",
            "seed": seed,
            "std": 0.05,
            "initial_trainable_state_sha256": _state_dict_sha256(state0),
        },
        "loss": {"class": "torch.nn.CrossEntropyLoss", "constructor_defaults": True},
        "training": {
            "optimizer_steps_recorded": steps,
            "loss_index_0": "cross-entropy before first optimizer.step()",
            "plot_y_max": plot_y_max,
        },
        "variants": [
            {"key": v["key"], "operator_level_mlp_kwargs": jsonify(v["kwargs"])}
            for v in variants
        ],
    }

    return {"mlp": bundle}, {"mlp_synthetic_ce": cfg_section}


# --- FGLN -------------------------------------------------------------------------

def curves_fgln(
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    d: int,
    depth: int,
    n: int,
    p_gate: float,
    seed: int,
    plot_y_max: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dtype = torch.float64
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, d, generator=g, device=device, dtype=dtype)
    u, _, vh = torch.linalg.svd(torch.randn(d, d, generator=g, device=device, dtype=dtype))
    p_star = u @ vh
    y = x @ p_star.T

    net0 = FGLN(d, depth=depth, p_gate=p_gate, seed=seed).to(device=device, dtype=dtype)
    init_weights(net0, init="xavier", seed=seed + 1, target_norm=float(p_star.norm().item()), rescale_mode="all")

    outer_lr = 0.6
    masked_als_kwargs_base = {
        "lr": outer_lr,
        "lam": 1e-4,
        "n_sweeps": 3,
        "als_reverse_sweep": True,
        "adaptive_lambda_mode": "max",
        "als_gateperm_warmstart": True,
        "als_gateperm_warmstart_once": True,
        "als_lam_anchor_post_warmstart": True,
        "normalize_contexts": True,
        "damping": True,
        "damping_beta": 0.5,
        "damping_min": 1e-6,
        "damping_max_trials": 20,
        "gauge_balance": True,
        "gauge_balance_passes": 1,
        "debug_nan": False,
    }

    def run_fgln(*, adaptive_lambda: bool) -> list[tuple[int, float]]:
        model = copy.deepcopy(net0)
        opt = MaskedOperatorALS(
            model,
            P_star=p_star,
            adaptive_lambda_step=adaptive_lambda,
            **masked_als_kwargs_base,
        )
        hist: list[tuple[int, float]] = []
        for t in range(steps + 1):
            with torch.no_grad():
                p = compute_P_fgln(model)
                mse = float(torch.mean((x @ p.T - y) ** 2).item())
                hist.append((t, mse))
            if t == steps:
                break
            try:
                with torch.no_grad():
                    pred = x @ p.T
                    grad_p = (2.0 / (n * d)) * (pred - y).T @ x
                    p_tgt = p - outer_lr * grad_p
                opt.step(P_tgt_override=p_tgt)
            except (FloatingPointError, RuntimeError):
                hist.append((t + 1, float("nan")))
                break
        return hist

    h_fixed = run_fgln(adaptive_lambda=False)
    h_adapt = run_fgln(adaptive_lambda=True)

    def _fgln_single_plot(hist: list[tuple[int, float]], title: str, fname: str, color: str) -> bool:
        xs = [a[0] for a in hist]
        raw = [a[1] for a in hist]
        y_plot, clipped = clip_for_plot(raw, plot_y_max)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(xs, y_plot, color=color, lw=2.0)
        ax.set_xlabel("Step")
        ax.set_ylabel("Train MSE")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        annotate_clip(ax, clipped, plot_y_max)
        save_fig(fig, out_dir / fname)
        return clipped

    _ = _fgln_single_plot(h_fixed, "FGLN ALS-exact (fixed λ)", "fgln_mse_als_fixed_lam.png", "#9467bd")
    _ = _fgln_single_plot(h_adapt, "FGLN ALS-exact + adaptive λ step", "fgln_mse_als_adaptive_lam.png", "#ff7f0e")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    yf, cf = clip_for_plot([a[1] for a in h_fixed], plot_y_max)
    ya, ca = clip_for_plot([a[1] for a in h_adapt], plot_y_max)
    ax.plot([a[0] for a in h_fixed], yf, label="ALS-exact (fixed λ)", color="#9467bd", lw=2.0)
    ax.plot([a[0] for a in h_adapt], ya, label="ALS-exact + adaptive λ step", color="#ff7f0e", lw=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    annotate_clip(ax, cf or ca, plot_y_max)
    save_fig(fig, out_dir / "fgln_train_mse.png")

    return (
        {
            "fgln": {
                "config": {"d": d, "depth": depth, "n": n, "p_gate": p_gate},
                "als_fixed_lam": h_fixed,
                "als_adaptive_lambda_step": h_adapt,
            }
        },
        {
            "fgln_masked_als": {
                "scenario": "fgln_teacher_student",
                "model": {
                    "class": "FGLN",
                    "d": d,
                    "depth": depth,
                    "p_gate": p_gate,
                    "constructor_seed": seed,
                },
                "init_weights": {
                    "function": "init_weights",
                    "init": "xavier",
                    "seed": seed + 1,
                    "rescale_mode": "all",
                    "target_norm": float(p_star.norm().item()),
                },
                "data": {
                    "n_samples": n,
                    "seed": seed,
                    "x": "randn(n,d) with Generator.manual_seed(seed)",
                    "P_star": "SVD of randn(d,d) with same generator after x",
                    "y": "x @ P_star.T",
                },
                "dtype": str(dtype),
                "device": str(device),
                "outer_operator_step": {
                    "lr": outer_lr,
                    "P_tgt_override_formula": "P - lr * grad_P MSE; grad_P = (2/(n*d)) * (pred-y).T @ x",
                },
                "masked_operator_als_shared_kwargs": jsonify(masked_als_kwargs_base),
                "runs": [
                    {
                        "key": "als_fixed_lam",
                        "adaptive_lambda_step": False,
                        "MaskedOperatorALS_kwargs": jsonify(
                            {**masked_als_kwargs_base, "adaptive_lambda_step": False}
                        ),
                    },
                    {
                        "key": "als_adaptive_lambda_step",
                        "adaptive_lambda_step": True,
                        "MaskedOperatorALS_kwargs": jsonify(
                            {**masked_als_kwargs_base, "adaptive_lambda_step": True}
                        ),
                    },
                ],
                "plot_y_max": plot_y_max,
            }
        },
    )


# --- Operator-KFAC depth sweep ---------------------------------------------------

def curves_operator_kfac_depth_sweep(
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    depths: list[int],
    hidden: int,
    d_in: int,
    d_out: int,
    batch: int,
    seed: int,
    plot_y_max: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compare operator_kfac vs block_diagonal vs als_mn across network depths.

    Trains a depth-d bias-free linear MLP (no ReLU) with each optimizer and
    records the cross-entropy loss curve.  The depth sweep surfaces spectral
    collapse: block_diagonal degrades at depth while operator_kfac remains
    stable thanks to its coupled factorization.  App. B.3.
    """
    dtype = torch.float64
    # Each entry: (key, opt_type, kwargs)
    # opt_type="kfac" → KFAC; opt_type="operator_level_mlp" → OperatorLevelMLP
    methods = [
        ("kfac", "kfac", {
            "lr": 0.001,  # fine-tuned: smaller lr + larger damping for stability at depth 8-16
            "factor_decay": 0.95, "damping": 0.1,
            "momentum": 0.0, "weight_decay": 0.0, "inv_floor": 1e-2,
        }),
        ("operator_kfac", "operator_level_mlp", {
            "lr": 0.08, "lam": 1e-3, "approximation": "operator_kfac", "momentum": 0.0,
        }),
        ("als_exact", "operator_level_mlp", {
            "lr": 50.0, "lam": 1e-3, "approximation": "als",
            "als_layer_solve": "exact", "n_sweeps": 10,
            "als_reverse_sweep": True, "momentum": 0.0,
            "als_gateperm_warmstart": False, "als_gateperm_warmstart_once": False,
            # als_linear_target: for a linear MLP (no ReLU) all gate masks are
            # identity, so the single shared target P_target = P - lr*∇_P L is exact.
            # High lr (50) lets ALS make aggressive operator-space steps; the exact
            # per-layer factorisation solve keeps the update stable.
            "als_linear_target": True,
        }),
    ]
    colors = {"kfac": "#d62728", "operator_kfac": "#2ca02c", "als_exact": "#9467bd"}
    labels = {"kfac": "K-FAC", "operator_kfac": "Operator-KFAC", "als_exact": "ALS-Exact"}

    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, d_in, device=device, dtype=dtype, generator=g)
    y_cls = torch.randint(0, d_out, (batch,), device=device)
    crit = nn.CrossEntropyLoss()

    all_results: dict[str, Any] = {}

    for depth in depths:
        depth_results: dict[str, list[tuple[int, float]]] = {}

        for method_key, opt_type, opt_kwargs in methods:
            torch.manual_seed(seed)
            model = make_mlp(depth, d_in, hidden, d_out).to(device=device, dtype=dtype)
            if opt_type == "kfac":
                opt = KFAC(list(model.parameters()), **opt_kwargs)
            else:
                kw = _operator_level_mlp_kwargs(**opt_kwargs)
                opt = OperatorLevelMLP(list(model.parameters()), **kw)
            opt.attach_hooks(model)
            depth_results[method_key] = run_mlp_loop(
                model, opt, lambda: crit(model(x), y_cls), steps  # noqa: B023
            )

        all_results[f"depth_{depth}"] = depth_results

        # Per-depth curve plot
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for method_key, _, __ in methods:
            hist = depth_results[method_key]
            xs = [h[0] for h in hist]
            raw = [h[1] for h in hist]
            ys, clipped = clip_for_plot(raw, plot_y_max)
            ax.plot(xs, ys, label=labels[method_key], color=colors[method_key], lw=2.0)
            annotate_clip(ax, clipped, plot_y_max)
        ax.set_xlabel("Step")
        ax.set_ylabel("Train CE loss")
        ax.set_yscale("log")
        ax.set_title(f"Operator-KFAC depth sweep — depth={depth}")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        save_fig(fig, out_dir / f"operator_kfac_depth_{depth}.png")

    # Summary: final loss vs depth for each method
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for method_key, _, __ in methods:
        final_losses = [all_results[f"depth_{d}"][method_key][-1][1] for d in depths]
        ys, clipped = clip_for_plot(final_losses, plot_y_max)
        ax.plot(depths, ys, marker="o", label=labels[method_key], color=colors[method_key], lw=2.0)
        annotate_clip(ax, clipped, plot_y_max)
    ax.set_xlabel("Depth")
    ax.set_ylabel(f"Final CE loss (step {steps})")
    ax.set_yscale("log")
    ax.set_title("Final loss vs depth — K-FAC vs Operator-KFAC vs ALS-Exact")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    save_fig(fig, out_dir / "operator_kfac_depth_summary.png")

    # Featured: per-step loss curve at the maximum depth (most informative)
    featured_depth = max(depths)
    fig_f, ax_f = plt.subplots(figsize=(7.0, 4.5))
    for method_key, _, __ in methods:
        hist = all_results[f"depth_{featured_depth}"][method_key]
        xs = [h[0] for h in hist]
        raw = [h[1] for h in hist]
        ys, clipped = clip_for_plot(raw, plot_y_max)
        ax_f.plot(xs, ys, label=labels[method_key], color=colors[method_key], lw=2.0)
        annotate_clip(ax_f, clipped, plot_y_max)
    ax_f.set_xlabel("Step")
    ax_f.set_ylabel("Train CE loss")
    ax_f.set_yscale("log")
    ax_f.set_title(f"K-FAC vs Operator-KFAC vs ALS-Exact — depth={featured_depth} (App. B.3)")
    ax_f.legend(frameon=False)
    ax_f.grid(True, alpha=0.25)
    save_fig(fig_f, out_dir / "operator_kfac_featured.png")

    cfg_section: dict[str, Any] = {
        "operator_kfac_depth_sweep": {
            "scenario": "synthetic_ce_linear_mlp",
            "architecture": {
                "type": "bias-free linear MLP (no ReLU)",
                "d_in": d_in,
                "hidden": hidden,
                "d_out": d_out,
                "depths_swept": depths,
            },
            "training": {
                "batch": batch,
                "steps": steps,
                "seed": seed,
                "dtype": str(dtype),
                "device": str(device),
                "loss": "CrossEntropyLoss",
            },
            "methods": [
                {"key": k, "opt_type": ot, "kwargs": jsonify(v)}
                for k, ot, v in methods
            ],
        }
    }

    return {"operator_kfac_depth_sweep": all_results}, cfg_section


# --- Mean-field comparison --------------------------------------------------------

def curves_mean_field_comparison(
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    depth: int,
    hidden: int,
    d_in: int,
    d_out: int,
    batch: int,
    lr: float,
    lam: float,
    n_sweeps: int,
    seed: int,
    plot_y_max: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mean-field bias: per-sample ALS vs batch ALS on linear vs ReLU networks.

    Two-panel comparison (App. B.1):
    - Deep linear (no ReLU): D_l = 1 for every sample → per-sample solve and
      batch solve are mathematically identical.  Both curves overlap.
    - ReLU MLP (same depth/width): each sample has its own gate pattern.
      Per-sample solves push shared weights in incompatible directions; their
      average is a poor solution for the batch objective.  The curve diverges
      from batch ALS-MN.
    """
    dtype  = torch.float64
    crit   = nn.CrossEntropyLoss()
    g      = torch.Generator(device=device).manual_seed(seed)
    x      = torch.randn(batch, d_in,   device=device, dtype=dtype, generator=g)
    y_cls  = torch.randint(0, d_out, (batch,), device=device)

    als_shared_kw = _operator_level_mlp_kwargs(
        lr=lr,
        lam=lam,
        approximation="als",
        n_sweeps=n_sweeps,
        als_reverse_sweep=True,
        momentum=0.0,
        als_gateperm_warmstart=False,
        als_gateperm_warmstart_once=False,
    )

    variants = [
        ("als_mn",         {**als_shared_kw, "als_layer_solve": "mn"}),
        ("als_per_sample", {**als_shared_kw, "als_layer_solve": "per_sample"}),
    ]
    labels  = {"als_mn": "Batch ALS-MN", "als_per_sample": "Mean-field (per-sample avg)"}
    colors  = {"als_mn": "#1f77b4",       "als_per_sample": "#d62728"}
    results: dict[str, Any] = {}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)
    panel_clipped = False

    for col, (net_label, relu) in enumerate([("linear", False), ("relu_mlp", True)]):
        ax = axes[col]
        net_results: dict[str, list[tuple[int, float]]] = {}

        torch.manual_seed(seed)
        init_model = make_mlp(depth, d_in, hidden, d_out, relu=relu).to(device=device, dtype=dtype)
        for m in init_model.modules():
            if isinstance(m, nn.Linear):
                if relu:
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                else:
                    # LeCun init (gain=1): preserves signal variance through linear chain
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        state0 = {k: v.clone() for k, v in init_model.state_dict().items()}

        for key, kw in variants:
            model = make_mlp(depth, d_in, hidden, d_out, relu=relu).to(device=device, dtype=dtype)
            model.load_state_dict(state0)
            opt = OperatorLevelMLP(list(model.parameters()), **kw)
            opt.attach_hooks(model)

            hist = run_mlp_loop(model, opt, lambda: crit(model(x), y_cls), steps)
            net_results[key] = hist

            xs  = [h[0] for h in hist]
            raw = [h[1] for h in hist]
            ys, clipped = clip_for_plot(raw, plot_y_max)
            panel_clipped = panel_clipped or clipped
            ax.plot(xs, ys, label=labels[key], color=colors[key], lw=2.0)
            annotate_clip(ax, clipped, plot_y_max)

        ax.set_xlabel("Step")
        ax.set_ylabel("Train CE loss")
        ax.set_yscale("log")
        title = "Deep linear (no ReLU)\n[mean-field ≡ batch ALS]" if net_label == "linear" \
                else "ReLU MLP\n[mean-field biased — gate heterogeneity]"
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(True, alpha=0.25)
        results[net_label] = net_results

    fig.suptitle(
        f"Mean-field vs batch ALS-MN  (depth={depth}, d={hidden}, batch={batch})",
        fontsize=11,
    )
    save_fig(fig, out_dir / "mean_field_comparison.png")

    cfg_section: dict[str, Any] = {
        "mean_field_comparison": {
            "scenario": "mean_field_vs_batch_als_linear_and_relu",
            "architecture": {
                "depth": depth, "hidden": hidden, "d_in": d_in, "d_out": d_out,
                "networks": ["bias-free linear (no ReLU)", "bias-free ReLU MLP"],
            },
            "training": {
                "batch": batch, "steps": steps, "seed": seed,
                "dtype": str(dtype), "device": str(device),
                "loss": "CrossEntropyLoss",
            },
            "optimizer_shared": jsonify(als_shared_kw),
            "variants": [{"key": k, "als_layer_solve": kw["als_layer_solve"]} for k, kw in variants],
        }
    }

    return {"mean_field_comparison": results}, cfg_section


# --- Linearized objective comparison ---------------------------------------------

def _linobj_product_P(params: list[torch.Tensor]) -> torch.Tensor:
    """Compute W_L @ … @ W_1 for a deep-linear stack (2-D weight tensors)."""
    P = params[0]
    for W in params[1:]:
        P = W @ P
    return P


def _linobj_contexts(params: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return left (A_k) and right (B_k) contexts for each layer.

    A_k = W_{k+1} … W_{L-1}  (identity for last layer)  shape (d_out, d_k_out)
    B_k = W_{k-1} … W_0      (identity for first layer) shape (d_k_in, d_in)
    """
    L = len(params)
    dtype, device = params[0].dtype, params[0].device
    d_out = params[-1].shape[0]
    d_in  = params[0].shape[1]

    A: list[torch.Tensor] = [None] * L  # type: ignore[list-item]
    A[L - 1] = torch.eye(d_out, dtype=dtype, device=device)
    for k in range(L - 2, -1, -1):
        A[k] = A[k + 1] @ params[k + 1]   # (d_out, d_{k+1}_in) = (d_out, d_k_out)

    B: list[torch.Tensor] = [None] * L  # type: ignore[list-item]
    B[0] = torch.eye(d_in, dtype=dtype, device=device)
    for k in range(1, L):
        B[k] = params[k - 1] @ B[k - 1]   # (d_k_in, d_in)

    return A, B


def _linobj_exact_dual_step(
    params: list[torch.Tensor],
    G_P: torch.Tensor,
    lr: float,
    lam: float,
) -> list[torch.Tensor]:
    """Exact coupled linearised solve via the dual form — O(d_out² · d_in²) per step.

    Minimises  ||Σ_k A_k ΔW_k B_k − lr·G_P||²_F + λ Σ_k ||ΔW_k||²_F  exactly.

    The Kronecker normal-equations system has total size N = Σ_k d_k_out·d_k_in
    (scales as depth × hidden²).  By duality the solve reduces to a single
    (d_out·d_in) × (d_out·d_in) system — feasible at depth=16, hidden=32.

    Solution:
        q   = (MM^T + λI)^{-1} (lr · vec G_P),   dim = d_out·d_in
        ΔW_k = A_k^T Q B_k^T,                      Q = reshape(q, d_out, d_in)
    where MM^T = Σ_k kron(A_k A_k^T,  B_k^T B_k).
    """
    A_list, B_list = _linobj_contexts(params)
    d_out, d_in = G_P.shape
    n_op = d_out * d_in
    dtype, device = G_P.dtype, G_P.device

    # Build MM^T (d_out·d_in × d_out·d_in) = Σ_k kron(A_k A_k^T, B_k^T B_k)
    MMT = torch.zeros(n_op, n_op, dtype=dtype, device=device)
    for k, p in enumerate(params):
        Ak = A_list[k]   # (d_out, d_k_out)
        Bk = B_list[k]   # (d_k_in, d_in)
        MMT.add_(torch.kron(Ak @ Ak.T, Bk.T @ Bk))

    MMT.add_(torch.eye(n_op, dtype=dtype, device=device), alpha=lam)
    q = torch.linalg.solve(MMT, (lr * G_P).reshape(-1))   # (n_op,)
    Q = q.reshape(d_out, d_in)

    # ΔW_k = A_k^T Q B_k^T  (gradient of the dual objective w.r.t. each ΔW_k)
    dW_list = [A_list[k].T @ Q @ B_list[k].T for k, _ in enumerate(params)]
    return dW_list


def _linobj_run(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    lam: float,
    steps: int,
    *,
    opt: "OperatorLevelMLP | None" = None,
    use_dual_linearized: bool = False,
) -> dict[str, list]:
    """Custom training loop recording loss, target residual, and per-layer ΔW norms.

    Supports two modes:
      opt != None            — use OperatorLevelMLP (ALS or any hook-based method)
      use_dual_linearized    — exact coupled linearised solve via dual form (no hooks)

    Returns dict with keys:
      'loss'     : list of (step, mse)
      'residual' : list of (step, ||P_new − P_target|| / ||P_target||)
      'dw_norms' : list of (step, [||ΔW_l|| for l in 0..L-1])
    """
    crit = nn.MSELoss()
    params = [p for p in model.parameters() if p.ndim == 2]
    B_sz, d_out = y.shape

    losses: list[tuple[int, float]] = []
    residuals: list[tuple[int, float]] = []
    dw_norms: list[tuple[int, list[float]]] = []

    for t in range(steps):
        model.train()
        with torch.no_grad():
            P_old = _linobj_product_P(params)
            G_P = (2.0 / (B_sz * d_out)) * (P_old @ x.T - y.T) @ x
            P_target = P_old - lr * G_P
            W_old = [p.data.clone() for p in params]

        if use_dual_linearized:
            loss_val = crit(model(x), y).item()
            losses.append((t, loss_val))
            with torch.no_grad():
                dW_list = _linobj_exact_dual_step(params, G_P, lr, lam)
                for p, dW in zip(params, dW_list):
                    p.data.sub_(dW)
        else:
            assert opt is not None
            opt.zero_grad()
            loss = crit(model(x), y)
            losses.append((t, loss.item()))
            loss.backward()
            opt.step()

        with torch.no_grad():
            P_new = _linobj_product_P(params)
            denom = P_target.norm().clamp(min=1e-30)
            res = (P_new - P_target).norm() / denom
            residuals.append((t, res.item()))
            dw = [(p.data - W_old[l]).norm().item() for l, p in enumerate(params)]
            dw_norms.append((t, dw))

        if not torch.isfinite(torch.tensor(losses[-1][1])):
            break

    return {"loss": losses, "residual": residuals, "dw_norms": dw_norms}


def curves_linearized_obj_comparison(
    *,
    out_dir: Path,
    device: torch.device,
    steps_small: int,
    steps_large: int,
    depth: int,
    hidden: int,
    d_in: int,
    d_out: int,
    batch: int,
    lr_small: float,
    lr_large: float,
    lam: float,
    n_sweeps: int,
    seed: int,
    plot_y_max: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """ALS-exact vs linearized-exact: objective quality at small and large lr (App. B.2).

    Deep linear network (no activations), depth ``depth``.  All samples share the same
    operator P(W) = W_L…W_1 so ALS uses a single batch-averaged target
        P_target = P_curr − lr · ∇_P L   (``als_linear_target=True``)
    and matches it to machine precision via ALS.  Linearized-exact instead solves the
    first-order Taylor expansion of the product (Sylvester equation), which only
    approximately achieves P_target.

    Three-row figure (small lr | large lr):
      Row 0 — Train MSE: both methods converge at small lr; at large lr the linearised
               method spikes before recovering because it overshoots the target.
      Row 1 — Target residual ||P_new − P_target|| / ||P_target||: near-zero for
               ALS-exact at all lr; grows as O(lr²) for linearised-exact (cross-layer
               coupling terms ignored by the first-order Taylor expansion).
      Row 2 — Mean per-layer update norm (1/L) Σ ||ΔW_l||: similar for both methods at
               the same lr, confirming that the target-residual gap is not caused by
               different step magnitudes but by the quadratic linearisation error.
    """
    dtype = torch.float64

    g      = torch.Generator(device=device).manual_seed(seed)
    x      = torch.randn(batch, d_in, device=device, dtype=dtype, generator=g)
    P_star = torch.randn(d_out, d_in, device=device, dtype=dtype, generator=g) * 0.3
    y_reg  = x @ P_star.T

    labels = {
        "als_exact":        "ALS-exact (nonlinear obj.)",
        "linearized_exact": "Linearized exact (coupled, 1-shot)",
    }
    colors = {
        "als_exact":        "#1f77b4",
        "linearized_exact": "#ff7f0e",
    }

    torch.manual_seed(seed)
    init_model = make_mlp(depth, d_in, hidden, d_out, relu=False).to(device=device, dtype=dtype)
    for m in init_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.1)
    state0 = {k: v.clone() for k, v in init_model.state_dict().items()}

    all_results: dict[str, Any] = {}

    panel_configs = [
        (lr_small, steps_small, f"Small lr = {lr_small}"),
        (lr_large, steps_large, f"Large lr = {lr_large}"),
    ]
    row_labels = ["Train MSE", "Target residual\n||P_new − P_target|| / ||P_target||", "Mean ‖ΔW_l‖ per layer"]
    row_scales = ["log", "log", "log"]

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.0))

    for col, (lr, steps, col_title) in enumerate(panel_configs):
        als_kw = _operator_level_mlp_kwargs(
            lr=lr, lam=lam, approximation="als",
            als_layer_solve="exact", n_sweeps=n_sweeps,
            als_reverse_sweep=True, momentum=0.0,
            als_gateperm_warmstart=False,
            als_linear_target=True,
        )

        col_data: dict[str, dict] = {}
        for key in ("als_exact", "linearized_exact"):
            model = make_mlp(depth, d_in, hidden, d_out, relu=False).to(device=device, dtype=dtype)
            model.load_state_dict(state0)
            if key == "als_exact":
                opt = OperatorLevelMLP(list(model.parameters()), **als_kw)
                opt.attach_hooks(model)
                col_data[key] = _linobj_run(model, x, y_reg, lr, lam, steps, opt=opt)
            else:  # linearized_exact: fast dual-form coupled solve, no OperatorLevelMLP
                col_data[key] = _linobj_run(model, x, y_reg, lr, lam, steps,
                                            use_dual_linearized=True)

        all_results[f"lr_{lr}"] = col_data

        # Row 0: train MSE
        ax0 = axes[0, col]
        for key in labels:
            rec = col_data[key]
            xs  = [h[0] for h in rec["loss"]]
            raw = [h[1] for h in rec["loss"]]
            ys, clipped = clip_for_plot(raw, plot_y_max)
            ax0.plot(xs, ys, label=labels[key], color=colors[key], lw=2.0)
            annotate_clip(ax0, clipped, plot_y_max)
        ax0.set_yscale("log")
        ax0.set_title(col_title, fontsize=10)
        ax0.set_ylabel(row_labels[0])
        ax0.legend(frameon=False, fontsize=8)
        ax0.grid(True, alpha=0.25)

        # Row 1: target residual
        ax1 = axes[1, col]
        for key in labels:
            rec = col_data[key]
            xs  = [h[0] for h in rec["residual"]]
            raw = [h[1] for h in rec["residual"]]
            ys, clipped = clip_for_plot(raw, plot_y_max)
            ax1.plot(xs, ys, label=labels[key], color=colors[key], lw=2.0)
            annotate_clip(ax1, clipped, plot_y_max)
        ax1.set_yscale("log")
        ax1.set_ylabel(row_labels[1])
        ax1.legend(frameon=False, fontsize=8)
        ax1.grid(True, alpha=0.25)

        # Row 2: mean per-layer ΔW norm
        ax2 = axes[2, col]
        for key in labels:
            rec = col_data[key]
            xs   = [h[0] for h in rec["dw_norms"]]
            means = [float(np.mean(h[1])) for h in rec["dw_norms"]]
            ys, clipped = clip_for_plot(means, plot_y_max)
            ax2.plot(xs, ys, label=labels[key], color=colors[key], lw=2.0)
            annotate_clip(ax2, clipped, plot_y_max)
        ax2.set_yscale("log")
        ax2.set_xlabel("Step")
        ax2.set_ylabel(row_labels[2])
        ax2.legend(frameon=False, fontsize=8)
        ax2.grid(True, alpha=0.25)

    fig.suptitle(
        f"ALS-exact vs Linearized-exact  (depth={depth}, hidden={hidden}, batch={batch}, deep linear)\n"
        f"Target residual = ||P_new − P_target|| / ||P_target||  grows as O(lr²) for the linearised objective",
        fontsize=10, y=1.01,
    )
    save_fig(fig, out_dir / "linearized_obj_comparison.png")

    cfg_section: dict[str, Any] = {
        "linearized_obj_comparison": {
            "scenario": "als_exact_vs_linearized_exact_deep_linear_mse",
            "architecture": {
                "depth": depth, "hidden": hidden, "d_in": d_in, "d_out": d_out,
                "type": "bias-free deep linear network",
            },
            "teacher": "P* = 0.3 * randn(d_out, d_in)",
            "training": {
                "batch": batch, "seed": seed,
                "dtype": str(dtype), "device": str(device),
                "loss": "MSELoss(model(x), P*x)",
                "weight_init": "nn.init.normal_(std=0.1), torch.manual_seed(seed)",
            },
            "panels": [
                {"lr": lr_small, "steps": steps_small},
                {"lr": lr_large, "steps": steps_large},
            ],
            "variants_shared": {
                "lam": lam, "n_sweeps_als": n_sweeps,
                "als_reverse_sweep": True, "als_gateperm_warmstart": False,
                "als_linear_target": True,
            },
        }
    }

    return {"linearized_obj_comparison": all_results}, cfg_section


# --- Focused MLP variant comparisons -----------------------------------------------

def _run_mlp_variants_focused(
    variant_keys: list[str],
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    batch: int,
    seed: int,
    plot_y_max: float,
    figure_name: str,
    title: str,
    kwarg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a named subset of _mlp_variants() and save a single focused comparison figure.

    ``kwarg_overrides`` is merged into every variant's kwargs before constructing
    OperatorLevelMLP — use it to disable options (e.g. warmstart) that are only
    meaningful for deep/ill-conditioned regimes.
    """
    dtype = torch.float64
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, 24, device=device, dtype=dtype, generator=g)
    y_cls = torch.randint(0, 5, (batch,), device=device, dtype=torch.long)

    torch.manual_seed(seed)
    template = make_tiny_mlp().to(device=device, dtype=dtype)
    for m in template.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.05)
    state0 = {k: v.clone() for k, v in template.state_dict().items()}

    crit = nn.CrossEntropyLoss()
    all_variants = {v["key"]: v for v in _mlp_variants()}
    selected = [all_variants[k] for k in variant_keys if k in all_variants]
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(selected), 1)))

    bundle: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    clipped_any = False

    for i, entry in enumerate(selected):
        key = entry["key"]
        kwargs = {**entry["kwargs"], **(kwarg_overrides or {})}
        model = make_tiny_mlp().to(device=device, dtype=dtype)
        model.load_state_dict(state0)
        opt = OperatorLevelMLP(list(model.parameters()), **kwargs)
        opt.attach_hooks(model)

        hist = run_mlp_loop(model, opt, lambda: crit(model(x), y_cls), steps)  # noqa: B023
        losses = [v for _, v in hist]
        bundle.append({"key": key, "losses": losses})
        ts = [h[0] for h in hist]
        label = key.replace("mlp_", "").replace("_", " ")
        color = cmap[i % len(cmap)]
        y_plot, c_any = clip_for_plot(losses, plot_y_max)
        clipped_any = clipped_any or c_any
        ax.plot(ts, y_plot, label=label, color=color, linewidth=2.0)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    annotate_clip(ax, clipped_any, plot_y_max)
    save_fig(fig, out_dir / f"{figure_name}.png")

    cfg_section: dict[str, Any] = {
        figure_name: {
            "scenario": "mlp_synthetic_cross_entropy_focused",
            "architecture": _mlp_architecture(),
            "variant_keys": variant_keys,
            "dtype": str(dtype),
            "device": str(device),
            "training": {
                "batch": batch,
                "steps": steps,
                "seed": seed,
                "plot_y_max": plot_y_max,
            },
        }
    }
    return {figure_name: bundle}, cfg_section


_NO_WARMSTART = {
    "als_gateperm_warmstart": False,
    "als_gateperm_warmstart_once": False,
}


def _run_with_step_delta(
    model: nn.Module,
    opt: Any,
    loss_fn: "Callable[[], torch.Tensor]",
    steps: int,
    catch: "tuple[type[Exception], ...]" = (RuntimeError, FloatingPointError),
) -> "tuple[list[tuple[int, float]], list[float]]":
    """Training loop that also tracks the normalized per-step loss improvement.

    Returns (hist, step_deltas) where step_deltas[t] = (L_before - L_after) / |L_before|.
    Positive ≈ step improved the model; negative = step was harmful.
    """
    hist: list[tuple[int, float]] = []
    step_deltas: list[float] = []
    for t in range(steps + 1):
        model.eval()
        with torch.no_grad():
            val = float(loss_fn().detach().cpu().item())
        model.train()
        hist.append((t, val))
        if t == steps:
            break
        l_before = val  # eval loss == loss just before step (no dropout/BN here)
        try:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            opt.step()
        except catch:
            break
        model.eval()
        with torch.no_grad():
            l_after = float(loss_fn().detach().cpu().item())
        model.train()
        denom = max(abs(l_before), 1e-30)
        step_deltas.append((l_before - l_after) / denom)
    return hist, step_deltas


def curves_dc_mlp(
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    batch: int,
    seed: int,
    plot_y_max: float,
    sweep_depths: "list[int] | None" = None,
    sweep_steps: int = 200,
    sweep_hidden: int = 16,
    sweep_d_in: int = 16,
    sweep_d_out: int = 8,
    sweep_batch: int = 32,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """D&C on MLP: shallow comparison + depth sweep showing residual growth (App. B.4b).

    Part 1 — shallow MLP (ALS-MN vs D&C): both methods run on the fixed tiny MLP;
    D&C is systematically slower due to gate heterogeneity.

    Part 2 — depth sweep (depths [2, 4, 8] with relu=True): tracks the per-step
    normalized loss improvement (L_before - L_after) / L_before as a proxy for the
    ALS target residual.  ALS-MN stays positive at every depth; D&C improvement
    degrades with depth and becomes negative (harmful steps) at depth 8, consistent
    with the operator residual exceeding 1 when gate heterogeneity compounds.
    """
    if sweep_depths is None:
        sweep_depths = [2, 4, 8]

    # --- Part 1: existing shallow comparison on tiny MLP ---
    part1_results, part1_cfg = _run_mlp_variants_focused(
        ["mlp_als_mn", "mlp_dc_mlp"],
        out_dir=out_dir,
        device=device,
        steps=steps,
        batch=batch,
        seed=seed,
        plot_y_max=plot_y_max,
        figure_name="dc_mlp_comparison",
        title="D&C on MLP: per-sample ALS average fails (App. B.4b)",
        kwarg_overrides=_NO_WARMSTART,
    )

    # --- Part 2: depth sweep ---
    dtype = torch.float64
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(device=device).manual_seed(seed)
    x_sw = torch.randn(sweep_batch, sweep_d_in, device=device, dtype=dtype, generator=g)
    y_sw = torch.randint(0, sweep_d_out, (sweep_batch,), device=device)

    als_kw = _operator_level_mlp_kwargs(
        lr=0.35, lam=1e-3, approximation="als",
        als_layer_solve="mn", n_sweeps=3, als_reverse_sweep=True,
        momentum=0.0, als_gateperm_warmstart=False, als_gateperm_warmstart_once=False,
    )
    dc_kw = _operator_level_mlp_kwargs(
        lr=0.35, lam=1e-3, approximation="dc_mlp",
        n_sweeps=3, momentum=0.0,
    )
    sweep_methods = [
        ("als_mn", als_kw, "ALS-MN (batch)", "#1f77b4"),
        ("dc_mlp", dc_kw, "D&C MLP (per-sample avg)", "#d62728"),
    ]

    n_depths = len(sweep_depths)
    fig_sw, axes_sw = plt.subplots(2, n_depths, figsize=(5.0 * n_depths, 8.0))
    if n_depths == 1:
        axes_sw = axes_sw.reshape(2, 1)
    sweep_bundle: dict[str, Any] = {}

    for di, depth in enumerate(sweep_depths):
        torch.manual_seed(seed)
        init_model = make_mlp(depth, sweep_d_in, sweep_hidden, sweep_d_out, relu=True).to(device=device, dtype=dtype)
        for m in init_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        state0 = {k: v.clone() for k, v in init_model.state_dict().items()}
        depth_bundle: dict[str, Any] = {}

        for method_key, kw, label, color in sweep_methods:
            model = make_mlp(depth, sweep_d_in, sweep_hidden, sweep_d_out, relu=True).to(device=device, dtype=dtype)
            model.load_state_dict(state0)
            opt = OperatorLevelMLP(list(model.parameters()), **kw)
            opt.attach_hooks(model)

            hist, step_deltas = _run_with_step_delta(
                model, opt, lambda: crit(model(x_sw), y_sw), sweep_steps  # noqa: B023
            )
            depth_bundle[method_key] = {"history": hist, "step_deltas": step_deltas}

            ax_loss = axes_sw[0, di]
            xs = [h[0] for h in hist]
            raw = [h[1] for h in hist]
            ys, clipped = clip_for_plot(raw, plot_y_max)
            ax_loss.plot(xs, ys, label=label, color=color, lw=2.0)
            annotate_clip(ax_loss, clipped, plot_y_max)

            ax_res = axes_sw[1, di]
            xs_d = list(range(len(step_deltas)))
            ax_res.plot(xs_d, step_deltas, label=label, color=color, lw=1.5, alpha=0.8)
            ax_res.axhline(0, color="k", lw=0.8, ls="--")

        for row, ax in enumerate([axes_sw[0, di], axes_sw[1, di]]):
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.25)
            if di == 0:
                ax.set_ylabel("Train CE loss" if row == 0 else "(L_before − L_after) / L_before")
            if row == 0:
                ax.set_yscale("log")
                ax.set_title(f"depth={depth}", fontsize=10)
                ax.legend(frameon=False, fontsize=8)
            else:
                ax.legend(frameon=False, fontsize=8)

        sweep_bundle[f"depth_{depth}"] = depth_bundle

    fig_sw.suptitle(
        "D&C on MLP: loss and per-step improvement vs depth\n"
        "Negative improvement = step is harmful (operator residual > 1)",
        fontsize=10,
    )
    save_fig(fig_sw, out_dir / "dc_mlp_depth_sweep.png")

    cfg: dict[str, Any] = {
        **part1_cfg,
        "dc_mlp_depth_sweep": {
            "scenario": "dc_mlp_vs_als_mn_depth_sweep",
            "depths": sweep_depths,
            "architecture": {"d_in": sweep_d_in, "hidden": sweep_hidden, "d_out": sweep_d_out, "relu": True},
            "training": {"steps": sweep_steps, "batch": sweep_batch, "seed": seed, "dtype": str(dtype)},
            "methods": [{"key": m[0], "label": m[2]} for m in sweep_methods],
        },
    }
    return {**part1_results, "dc_mlp_depth_sweep": sweep_bundle}, cfg


def curves_secant_mlp(
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    batch: int,
    seed: int,
    plot_y_max: float,
    depth: int = 8,
    hidden: int = 16,
    d_in: int = 16,
    d_out: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Secant gate linearization variants vs ALS-exact reference (App. B.5).

    Uses a deep ReLU MLP (default depth=8) where gate heterogeneity is large enough
    that the quality of the gradient/curvature approximation matters.  Each method has
    its lr independently tuned:
      - ALS-exact: larger operator lr (second-order per-layer exact solve converges fast)
      - secant_grad_exact: smaller weight-space lr (rank-1 curvature, stable gradient)
      - secant: same small lr (rank-1 for both gradient and curvature — most approximate)
    als_mn is omitted; the reference is als_exact_layers.
    """
    dtype = torch.float64
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch, d_in, device=device, dtype=dtype, generator=g)
    y_cls = torch.randint(0, d_out, (batch,), device=device, dtype=torch.long)

    torch.manual_seed(seed)
    init_model = make_mlp(depth, d_in, hidden, d_out, relu=True).to(device=device, dtype=dtype)
    for m in init_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
    state0 = {k: v.clone() for k, v in init_model.state_dict().items()}

    # (key, kwargs, label, color) — lr tuned independently per method.
    # At depth≥16 the rank-1 secant approximation quality degrades; secant methods
    # need a very small weight-space lr to avoid divergence and converge slowly,
    # while ALS-exact makes efficient large operator-space steps (lr=2.0).
    methods = [
        ("als_exact_layers", _operator_level_mlp_kwargs(
            lr=2.0, lam=1e-3, approximation="als",
            als_layer_solve="exact", n_sweeps=3, als_reverse_sweep=True,
            momentum=0.0,
            als_gateperm_warmstart=False, als_gateperm_warmstart_once=False,
        ), "ALS-Exact (reference)", "#9467bd"),
        ("secant_grad_exact", _operator_level_mlp_kwargs(
            lr=0.0001, lam=1e-3, approximation="secant_grad_exact", momentum=0.0,
        ), "Secant + exact gradient", "#2ca02c"),
        ("secant", _operator_level_mlp_kwargs(
            lr=0.0001, lam=1e-3, approximation="secant", momentum=0.0,
        ), "Secant (rank-1 curvature & gradient)", "#ff7f0e"),
    ]

    bundle: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    clipped_any = False

    for key, kwargs, label, color in methods:
        model = make_mlp(depth, d_in, hidden, d_out, relu=True).to(device=device, dtype=dtype)
        model.load_state_dict(state0)
        opt = OperatorLevelMLP(list(model.parameters()), **kwargs)
        opt.attach_hooks(model)

        hist = run_mlp_loop(model, opt, lambda: crit(model(x), y_cls), steps)  # noqa: B023
        losses = [v for _, v in hist]
        bundle.append({"key": key, "losses": losses})
        ts = [h[0] for h in hist]
        y_plot, c_any = clip_for_plot(losses, plot_y_max)
        clipped_any = clipped_any or c_any
        ax.plot(ts, y_plot, label=label, color=color, linewidth=2.0)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_yscale("log")
    ax.set_title(f"Secant vs ALS-Exact — depth={depth} ReLU MLP (App. B.5)", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    annotate_clip(ax, clipped_any, plot_y_max)
    save_fig(fig, out_dir / "secant_comparison.png")

    cfg_section: dict[str, Any] = {
        "secant_comparison": {
            "scenario": "mlp_synthetic_cross_entropy_secant",
            "architecture": {
                "depth": depth, "hidden": hidden, "d_in": d_in, "d_out": d_out,
                "type": "bias-free ReLU MLP",
            },
            "dtype": str(dtype),
            "device": str(device),
            "training": {
                "batch": batch, "steps": steps, "seed": seed, "plot_y_max": plot_y_max,
            },
            "methods": [
                {"key": m[0], "label": m[2], "operator_level_mlp_kwargs": jsonify(m[1])}
                for m in methods
            ],
        }
    }
    return {"secant_comparison": bundle}, cfg_section


def curves_adaptive_lam_mlp(
    *,
    out_dir: Path,
    device: torch.device,
    steps: int,
    batch: int,
    seed: int,
    plot_y_max: float,
    shallow_depth: int = 2,
    deep_depth: int = 16,
    hidden: int = 16,
    d_in: int = 16,
    d_out: int = 8,
    init_std: float = 0.1,
    lam_alpha: float = 5e-8,
    lr: float = 0.35,
    lam: float = 1e-6,
    n_sweeps: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adaptive λ vs fixed λ — shallow works, deep diverges (App. B.6a).

    Two-panel figure on a bias-free ReLU MLP with small std initialisation.
    With sub-Kaiming std, context singular values σ_max(M_l) decay with depth:
    outer layers retain signal while inner layers at large L collapse to
    σ_max ≈ 0.

    Fixed λ suppresses updates at those collapsed layers (λ dominates the
    denominator), limiting damage.  Adaptive λ_l = α·(σ_max(M_l)+σ_max(N_l))
    collapses with the context: λ_l → 0 removes all regularisation at collapsed
    layers and allows unbounded updates in numerically noisy directions —
    training diverges.

    Shallow panel (L=shallow_depth): both methods converge; α is calibrated so
    adaptive λ ≈ fixed λ at the first layer.
    Deep panel (L=deep_depth): inner-layer contexts collapse; adaptive λ diverges
    while fixed λ stays stable (though unable to make progress either, it does
    not destabilise training).
    """
    dtype = torch.float64
    crit = nn.CrossEntropyLoss()

    fixed_kwargs = _operator_level_mlp_kwargs(
        lr=lr, lam=lam, approximation="als", als_layer_solve="mn",
        n_sweeps=n_sweeps, als_reverse_sweep=True, momentum=0.0,
        als_gateperm_warmstart=False, als_gateperm_warmstart_once=False,
    )
    adaptive_kwargs = _operator_level_mlp_kwargs(
        lr=lr, lam=None, lam_alpha=lam_alpha, approximation="als",
        als_layer_solve="mn", n_sweeps=n_sweeps, als_reverse_sweep=True,
        momentum=0.0,
        als_gateperm_warmstart=False, als_gateperm_warmstart_once=False,
    )
    methods = [
        ("fixed_lam",    f"Fixed λ={lam:.0e}",            "#1f77b4", fixed_kwargs),
        ("adaptive_lam", f"Adaptive λ (α={lam_alpha})",  "#ff7f0e", adaptive_kwargs),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=False)
    bundle_all: dict[str, list[dict[str, Any]]] = {}

    for ax, depth, panel_title in [
        (axes[0], shallow_depth, f"Shallow (L={shallow_depth})"),
        (axes[1], deep_depth,    f"Deep (L={deep_depth})"),
    ]:
        g = torch.Generator(device=device).manual_seed(seed)
        x_data = torch.randn(batch, d_in, device=device, dtype=dtype, generator=g)
        y_data = torch.randint(0, d_out, (batch,), device=device, dtype=torch.long)

        torch.manual_seed(seed)
        init_model = make_mlp(depth, d_in, hidden, d_out, relu=True).to(device=device, dtype=dtype)
        for m in init_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_std)
        state0 = {k: v.clone() for k, v in init_model.state_dict().items()}

        panel_bundle: list[dict[str, Any]] = []
        clipped_panel = False

        for key, label, color, kwargs in methods:
            model = make_mlp(depth, d_in, hidden, d_out, relu=True).to(device=device, dtype=dtype)
            model.load_state_dict(state0)
            opt = OperatorLevelMLP(list(model.parameters()), **kwargs)
            opt.attach_hooks(model)

            hist = run_mlp_loop(model, opt, lambda: crit(model(x_data), y_data), steps)  # noqa: B023
            losses = [v for _, v in hist]
            panel_bundle.append({"key": key, "losses": losses})
            ts = [h[0] for h in hist]
            y_plot, c_any = clip_for_plot(losses, plot_y_max)
            clipped_panel = clipped_panel or c_any
            ax.plot(ts, y_plot, label=label, color=color, linewidth=2.0)

        import math
        random_ce = math.log(d_out)
        ax.axhline(random_ce, color="gray", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"Random init CE = log({d_out})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_yscale("log")
        ax.set_title(panel_title, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
        annotate_clip(ax, clipped_panel, plot_y_max)
        bundle_all[f"depth_{depth}"] = panel_bundle

    fig.suptitle(
        f"Adaptive λ (α={lam_alpha}) vs Fixed λ — shallow overlap, deep failure (App. B.6a)",
        fontsize=11,
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "adaptive_lam_mlp.png")

    cfg_section: dict[str, Any] = {
        "adaptive_lam_mlp": {
            "scenario": "mlp_synthetic_cross_entropy_adaptive_lam_depth",
            "architecture": {
                "shallow_depth": shallow_depth, "deep_depth": deep_depth,
                "hidden": hidden, "d_in": d_in, "d_out": d_out,
                "type": "bias-free ReLU MLP", "init": f"normal(std={init_std})",
            },
            "dtype": str(dtype),
            "device": str(device),
            "training": {
                "batch": batch, "steps": steps, "seed": seed,
                "lr": lr, "lam": lam, "lam_alpha": lam_alpha,
                "n_sweeps": n_sweeps, "plot_y_max": plot_y_max,
            },
        }
    }
    return {"adaptive_lam_mlp": bundle_all}, cfg_section


def main() -> None:
    _ALL_FIGURES = [
        "deep_linear", "mlp_all", "fgln", "kfac_depth", "mean_field", "linobj",
        "dc_mlp", "secant", "adaptive_lam",
    ]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str, default="outputs/variant_curves/run")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--figures",
        type=str,
        default="all",
        help=(
            "Comma-separated subset of figures to generate, or 'all'. "
            f"Valid keys: {', '.join(_ALL_FIGURES)}."
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps_deeplinear", type=int, default=200)
    ap.add_argument("--steps_mlp", type=int, default=150)
    ap.add_argument("--steps_fgln", type=int, default=100)
    ap.add_argument("--deeplinear_depth", type=int, default=8)
    ap.add_argument("--deeplinear_d", type=int, default=4)
    ap.add_argument("--deeplinear_n", type=int, default=32)
    ap.add_argument("--mlp_batch", type=int, default=64)
    ap.add_argument("--fgln_d", type=int, default=8)
    ap.add_argument("--fgln_depth", type=int, default=16)
    ap.add_argument("--fgln_n", type=int, default=32)
    ap.add_argument("--fgln_p", type=float, default=0.85)
    ap.add_argument("--steps_kfac_depth", type=int, default=300)
    ap.add_argument("--kfac_depths", type=str, default="2,4,8,16",
                    help="Comma-separated list of depths for operator_kfac depth sweep.")
    ap.add_argument("--kfac_hidden", type=int, default=16)
    ap.add_argument("--kfac_d_in", type=int, default=16)
    ap.add_argument("--kfac_d_out", type=int, default=8)
    ap.add_argument("--kfac_batch", type=int, default=32)
    ap.add_argument("--steps_mean_field", type=int, default=300)
    ap.add_argument("--mf_depth", type=int, default=8)
    ap.add_argument("--mf_hidden", type=int, default=32)
    ap.add_argument("--mf_d_in", type=int, default=32)
    ap.add_argument("--mf_d_out", type=int, default=8)
    ap.add_argument("--mf_batch", type=int, default=64)
    ap.add_argument("--mf_lr", type=float, default=0.15)
    ap.add_argument("--mf_lam", type=float, default=1e-3)
    ap.add_argument("--mf_n_sweeps", type=int, default=3)
    ap.add_argument("--secant_depth", type=int, default=16,
                    help="Depth of the ReLU MLP used for the secant comparison (B.5).")
    ap.add_argument("--secant_hidden", type=int, default=16)
    ap.add_argument("--secant_d_in", type=int, default=16)
    ap.add_argument("--secant_d_out", type=int, default=8)
    ap.add_argument("--dc_sweep_depths", type=str, default="2,4,8",
                    help="Comma-separated depths for D&C depth sweep (dc_mlp figure).")
    ap.add_argument("--dc_sweep_steps", type=int, default=200)
    ap.add_argument("--dc_sweep_hidden", type=int, default=16)
    ap.add_argument("--dc_sweep_d_in", type=int, default=16)
    ap.add_argument("--dc_sweep_d_out", type=int, default=8)
    ap.add_argument("--dc_sweep_batch", type=int, default=32)
    ap.add_argument("--linobj_steps_small", type=int, default=200)
    ap.add_argument("--linobj_steps_large", type=int, default=60)
    ap.add_argument("--linobj_depth", type=int, default=16)
    ap.add_argument("--linobj_hidden", type=int, default=32)
    ap.add_argument("--linobj_d_in", type=int, default=16)
    ap.add_argument("--linobj_d_out", type=int, default=8)
    ap.add_argument("--linobj_batch", type=int, default=64)
    ap.add_argument("--linobj_lr_small", type=float, default=0.05)
    ap.add_argument("--linobj_lr_large", type=float, default=2.0)
    ap.add_argument("--linobj_lam", type=float, default=1e-4)
    ap.add_argument("--linobj_n_sweeps", type=int, default=4)
    ap.add_argument("--adaptive_lam_alpha", type=float, default=5e-8,
                    help="α for adaptive λ (B.6a): λ_l = α·(σ_max(M_l)+σ_max(N_l)).")
    ap.add_argument("--adaptive_lam_shallow_depth", type=int, default=2)
    ap.add_argument("--adaptive_lam_deep_depth", type=int, default=16)
    ap.add_argument("--adaptive_lam_hidden", type=int, default=16)
    ap.add_argument("--adaptive_lam_d_in", type=int, default=16)
    ap.add_argument("--adaptive_lam_d_out", type=int, default=8)
    ap.add_argument("--adaptive_lam_init_std", type=float, default=0.1,
                    help="Weight init std for adaptive λ experiment. Sub-Kaiming so contexts "
                         "decay with depth and expose the collapse failure at large L.")
    ap.add_argument("--adaptive_lam_lr", type=float, default=0.35)
    ap.add_argument("--adaptive_lam_lam", type=float, default=1e-6)
    ap.add_argument("--adaptive_lam_n_sweeps", type=int, default=3)
    ap.add_argument(
        "--plot_y_max",
        type=float,
        default=10.0,
        help="Upper cap on plotted Y values (MSE, CE, rel.error) so diverging traces "
        "do not compress the rest; set <=0 to disable clipping.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    plot_y_max = float(args.plot_y_max)

    if args.figures.strip().lower() == "all":
        selected_figs: set[str] = set(_ALL_FIGURES)
    else:
        selected_figs = {k.strip() for k in args.figures.split(",") if k.strip()}
        unknown = selected_figs - set(_ALL_FIGURES)
        if unknown:
            ap.error(f"Unknown --figures keys: {', '.join(sorted(unknown))}. Valid: {', '.join(_ALL_FIGURES)}")

    def _want(key: str) -> bool:
        return key in selected_figs

    run_config: dict[str, Any] = {
        "script_module": "operator_level_optimization.scripts.train.variant_loss_curves",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "cli_args": vars(args),
        "resolved": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "outputs": {
            "directory": str(out_dir.resolve()),
            "figures": "PNG files in out_dir (deep_linear_*, loss_mlp_*, fgln_*, mlp_cross_entropy_*)",
            "metrics_json": "variant_curves.json",
            "reproducibility_json": "config.json",
        },
    }

    payload: dict[str, Any] = {}

    if _want("deep_linear"):
        p_dl, c_dl = curves_deep_linear(
            out_dir=out_dir,
            device=device,
            steps=args.steps_deeplinear,
            depth=args.deeplinear_depth,
            d=args.deeplinear_d,
            n=args.deeplinear_n,
            seed=args.seed,
            plot_y_max=plot_y_max,
        )
        payload.update(p_dl)
        run_config.update(c_dl)

    if _want("mlp_all"):
        p_mlp, c_mlp = curves_mlp(
            out_dir=out_dir,
            device=device,
            steps=args.steps_mlp,
            batch=args.mlp_batch,
            seed=args.seed,
            plot_y_max=plot_y_max,
        )
        payload.update(p_mlp)
        run_config.update(c_mlp)

    if _want("fgln"):
        p_fg, c_fg = curves_fgln(
            out_dir=out_dir,
            device=device,
            steps=args.steps_fgln,
            d=args.fgln_d,
            depth=args.fgln_depth,
            n=args.fgln_n,
            p_gate=args.fgln_p,
            seed=args.seed,
            plot_y_max=plot_y_max,
        )
        payload.update(p_fg)
        run_config.update(c_fg)

    if _want("kfac_depth"):
        kfac_depths = [int(d) for d in args.kfac_depths.split(",") if d.strip()]
        p_kd, c_kd = curves_operator_kfac_depth_sweep(
            out_dir=out_dir,
            device=device,
            steps=args.steps_kfac_depth,
            depths=kfac_depths,
            hidden=args.kfac_hidden,
            d_in=args.kfac_d_in,
            d_out=args.kfac_d_out,
            batch=args.kfac_batch,
            seed=args.seed,
            plot_y_max=plot_y_max,
        )
        payload.update(p_kd)
        run_config.update(c_kd)

    if _want("mean_field"):
        p_mf, c_mf = curves_mean_field_comparison(
            out_dir=out_dir,
            device=device,
            steps=args.steps_mean_field,
            depth=args.mf_depth,
            hidden=args.mf_hidden,
            d_in=args.mf_d_in,
            d_out=args.mf_d_out,
            batch=args.mf_batch,
            lr=args.mf_lr,
            lam=args.mf_lam,
            n_sweeps=args.mf_n_sweeps,
            seed=args.seed,
            plot_y_max=plot_y_max,
        )
        payload.update(p_mf)
        run_config.update(c_mf)

    if _want("linobj"):
        p_lo, c_lo = curves_linearized_obj_comparison(
            out_dir=out_dir,
            device=device,
            steps_small=args.linobj_steps_small,
            steps_large=args.linobj_steps_large,
            depth=args.linobj_depth,
            hidden=args.linobj_hidden,
            d_in=args.linobj_d_in,
            d_out=args.linobj_d_out,
            batch=args.linobj_batch,
            lr_small=args.linobj_lr_small,
            lr_large=args.linobj_lr_large,
            lam=args.linobj_lam,
            n_sweeps=args.linobj_n_sweeps,
            seed=args.seed,
            plot_y_max=plot_y_max,
        )
        payload.update(p_lo)
        run_config.update(c_lo)

    if _want("dc_mlp"):
        dc_sweep_depths = [int(d) for d in args.dc_sweep_depths.split(",") if d.strip()]
        p_dc, c_dc = curves_dc_mlp(
            out_dir=out_dir,
            device=device,
            steps=args.steps_mlp,
            batch=args.mlp_batch,
            seed=args.seed,
            plot_y_max=plot_y_max,
            sweep_depths=dc_sweep_depths,
            sweep_steps=args.dc_sweep_steps,
            sweep_hidden=args.dc_sweep_hidden,
            sweep_d_in=args.dc_sweep_d_in,
            sweep_d_out=args.dc_sweep_d_out,
            sweep_batch=args.dc_sweep_batch,
        )
        payload.update(p_dc)
        run_config.update(c_dc)

    if _want("secant"):
        p_sc, c_sc = curves_secant_mlp(
            out_dir=out_dir,
            device=device,
            steps=args.steps_mlp,
            batch=args.mlp_batch,
            seed=args.seed,
            plot_y_max=plot_y_max,
            depth=args.secant_depth,
            hidden=args.secant_hidden,
            d_in=args.secant_d_in,
            d_out=args.secant_d_out,
        )
        payload.update(p_sc)
        run_config.update(c_sc)

    if _want("adaptive_lam"):
        p_al, c_al = curves_adaptive_lam_mlp(
            out_dir=out_dir,
            device=device,
            steps=args.steps_mlp,
            batch=args.mlp_batch,
            seed=args.seed,
            plot_y_max=plot_y_max,
            shallow_depth=args.adaptive_lam_shallow_depth,
            deep_depth=args.adaptive_lam_deep_depth,
            hidden=args.adaptive_lam_hidden,
            d_in=args.adaptive_lam_d_in,
            d_out=args.adaptive_lam_d_out,
            init_std=args.adaptive_lam_init_std,
            lam_alpha=args.adaptive_lam_alpha,
            lr=args.adaptive_lam_lr,
            lam=args.adaptive_lam_lam,
            n_sweeps=args.adaptive_lam_n_sweeps,
        )
        payload.update(p_al)
        run_config.update(c_al)

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, float):
            return obj if obj == obj else None  # nan -> null
        return obj

    (out_dir / "config.json").write_text(
        json.dumps(jsonify(run_config), indent=2, sort_keys=True)
    )

    payload["reproducibility"] = {
        "config_json": "config.json",
        "note": "Full CLI, environment, data, model, and optimizer hyperparameters for this run.",
    }
    (out_dir / "variant_curves.json").write_text(json.dumps(_json_safe(payload), indent=2))
    print(f"[done] wrote {out_dir / 'config.json'} and figures under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
