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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from operator_level_optimization.core.optim.operator import OperatorLevelMLP
from operator_level_optimization.models.fgln import FGLN, MaskedOperatorALS, compute_P_fgln, init_weights
from operator_level_optimization.scripts.train.deep_linear_compare import run_method as deep_linear_run


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _y_for_plot(y: list[float], plot_y_max: float) -> tuple[np.ndarray, bool]:
    """Clip finite Y values at ``plot_y_max`` for display (JSON keeps raw).

    Returns ``(y_display, clipped)`` where ``clipped`` is True if any finite
    value exceeded ``plot_y_max``. NaNs are left unchanged.
    """
    a = np.asarray(y, dtype=float)
    if plot_y_max <= 0 or a.size == 0:
        return a, False
    finite = np.isfinite(a)
    over = finite & (a > plot_y_max)
    if not np.any(over):
        return a, False
    out = a.copy()
    out[over] = plot_y_max
    return out, True


def _annotate_clip(ax: Any, clipped: bool, plot_y_max: float) -> None:
    if clipped and plot_y_max > 0:
        ax.text(
            0.01,
            0.99,
            f"values > {plot_y_max:g} clipped for display",
            transform=ax.transAxes,
            fontsize=7,
            color="0.35",
            va="top",
            ha="left",
        )


def _tiny_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Linear(24, 8, bias=False),
        nn.ReLU(inplace=False),
        nn.Linear(8, 5, bias=False),
    )


def _mlp_architecture() -> dict[str, Any]:
    """Fixed synthetic MLP used for variant comparison (must match ``_tiny_mlp``)."""
    return {
        "description": "bias-free ReLU MLP for synthetic cross-entropy task",
        "layers": [
            {"type": "Linear", "in_features": 24, "out_features": 8, "bias": False},
            {"type": "ReLU", "inplace": False},
            {"type": "Linear", "in_features": 8, "out_features": 5, "bias": False},
        ],
        "num_classes": 5,
        "total_trainable_params": sum(p.numel() for p in _tiny_mlp().parameters()),
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


def _jsonify_run_value(obj: Any) -> Any:
    """Convert objects to JSON-serializable forms for ``config.json``."""
    if isinstance(obj, dict):
        return {str(k): _jsonify_run_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_run_value(v) for v in obj]
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


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
        als_sweeps=4,
        dc_lam=1e-4,
        dc_alt=3,
        dc_init_mode="identity_delta",
        dc_init_scale=0.01,
        spec_every=1_000_000,
        spec_steps=None,
        early_stop_patience=None,
        early_stop_rel_tol=1e-4,
        early_stop_abs_tol=1e-10,
        early_stop_min_steps=10_000,
    )
    configs: list[tuple[str, float, dict[str, Any]]] = [
        ("als_exact", 1.0, {}),
        ("dc", 1.0, {"dc_init_mode": "identity_delta", "dc_init_scale": 0.01}),
        ("dc", 1.0, {"dc_init_mode": "plain", "dc_init_scale": 1.0, "dc_lam": 1e-4}),
    ]
    # Disambiguate duplicate method keys for plotting
    labels = [
        "ALS-exact",
        "D&C (identity_delta)",
        "D&C (plain)",
    ]
    series: list[dict[str, Any]] = []
    fig_mse, ax_mse = plt.subplots(figsize=(8.0, 4.8))
    fig_rel, ax_rel = plt.subplots(figsize=(8.0, 4.8))
    colors = ["#9467bd", "#17becf", "#bcbd22"]
    slugs = ["als_exact", "dc_identity_delta", "dc_plain"]
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
        y_mse, cm = _y_for_plot(mses, plot_y_max)
        y_rel, cr = _y_for_plot(rels, plot_y_max)
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
        _annotate_clip(ax_i, cm, plot_y_max)
        fig_i.tight_layout()
        fig_i.savefig(out_dir / f"deep_linear_mse_{slug}.png", dpi=160)
        plt.close(fig_i)

    for ax, ylabel, fname, clipped_any in (
        (ax_mse, "Train MSE", "deep_linear_train_mse.png", clipped_mse_any),
        (ax_rel, r"$\|P-P^\star\|_F / \|P^\star\|_F$", "deep_linear_rel_operator_error.png", clipped_rel_any),
    ):
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, loc="best")
        _annotate_clip(ax, clipped_any, plot_y_max)
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)

    per_method_cfg: list[dict[str, Any]] = []
    for slug, label, (method, lr, extra) in zip(slugs, labels, configs):
        kw = {**common, **extra, "lr": lr}
        per_method_cfg.append(
            {
                "slug": slug,
                "plot_label": label,
                "method": method,
                "operator_step_lr": lr,
                "run_kwargs_overrides_vs_shared": _jsonify_run_value(extra),
                "merged_run_kwargs_passed_to_deep_linear_run": _jsonify_run_value(kw),
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
        "shared_run_kwargs": _jsonify_run_value(common),
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
            lr=0.08, lam=1e-3, approximation="secant", momentum=0.0,
        )},
        {"key": "mlp_secant_r3", "kwargs": _operator_level_mlp_kwargs(
            lr=0.08, lam=1e-3, approximation="secant_r", momentum=0.0, rank=3,
        )},
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
    template = _tiny_mlp().to(device=device, dtype=dtype)
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
        model = _tiny_mlp().to(device=device, dtype=dtype)
        model.load_state_dict(state0)
        params = list(model.parameters())
        opt = OperatorLevelMLP(params, **kwargs)
        opt.attach_hooks(model)
        losses: list[float] = []
        with torch.no_grad():
            losses.append(float(crit(model(x), y).detach().cpu().item()))
        try:
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
        except (RuntimeError, FloatingPointError) as e:
            bundle.append({"key": key, "error": str(e), "losses": losses})
            continue

        bundle.append({"key": key, "losses": losses})
        ts = list(range(0, len(losses)))
        label = key.replace("mlp_", "").replace("_", " ")
        color = cmap[i % len(cmap)]
        y_plot, c_any = _y_for_plot(losses, plot_y_max)
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
        _annotate_clip(ax_i, c_any, plot_y_max)
        fig_i.tight_layout()
        fig_i.savefig(out_dir / f"loss_{key}.png", dpi=160)
        plt.close(fig_i)

    for ax, fname, clipped_any in (
        (ax_lin, "mlp_cross_entropy_linear.png", clipped_lin_any),
        (ax_log, "mlp_cross_entropy_logy.png", clipped_log_any),
    ):
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-entropy loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best", fontsize=8, ncol=2)
        fig = ax.figure
        if ax is ax_log:
            ax.set_yscale("log")
        _annotate_clip(ax, clipped_any, plot_y_max)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)

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
            {"key": v["key"], "operator_level_mlp_kwargs": _jsonify_run_value(v["kwargs"])}
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
    g = torch.Generator().manual_seed(seed)
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
        y_plot, clipped = _y_for_plot(raw, plot_y_max)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(xs, y_plot, color=color, lw=2.0)
        ax.set_xlabel("Step")
        ax.set_ylabel("Train MSE")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        _annotate_clip(ax, clipped, plot_y_max)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)
        return clipped

    _ = _fgln_single_plot(h_fixed, "FGLN ALS-exact (fixed λ)", "fgln_mse_als_fixed_lam.png", "#9467bd")
    _ = _fgln_single_plot(h_adapt, "FGLN ALS-exact + adaptive λ step", "fgln_mse_als_adaptive_lam.png", "#ff7f0e")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    yf, cf = _y_for_plot([a[1] for a in h_fixed], plot_y_max)
    ya, ca = _y_for_plot([a[1] for a in h_adapt], plot_y_max)
    ax.plot([a[0] for a in h_fixed], yf, label="ALS-exact (fixed λ)", color="#9467bd", lw=2.0)
    ax.plot([a[0] for a in h_adapt], ya, label="ALS-exact + adaptive λ step", color="#ff7f0e", lw=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    _annotate_clip(ax, cf or ca, plot_y_max)
    fig.tight_layout()
    fig.savefig(out_dir / "fgln_train_mse.png", dpi=160)
    plt.close(fig)

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
                "masked_operator_als_shared_kwargs": _jsonify_run_value(masked_als_kwargs_base),
                "runs": [
                    {
                        "key": "als_fixed_lam",
                        "adaptive_lambda_step": False,
                        "MaskedOperatorALS_kwargs": _jsonify_run_value(
                            {**masked_als_kwargs_base, "adaptive_lambda_step": False}
                        ),
                    },
                    {
                        "key": "als_adaptive_lambda_step",
                        "adaptive_lambda_step": True,
                        "MaskedOperatorALS_kwargs": _jsonify_run_value(
                            {**masked_als_kwargs_base, "adaptive_lambda_step": True}
                        ),
                    },
                ],
                "plot_y_max": plot_y_max,
            }
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=str, default="outputs/variant_curves/run")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
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
    device = _device(args.device)
    plot_y_max = float(args.plot_y_max)

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

    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, float):
            return obj if obj == obj else None  # nan -> null
        return obj

    (out_dir / "config.json").write_text(
        json.dumps(_jsonify_run_value(run_config), indent=2, sort_keys=True)
    )

    payload["reproducibility"] = {
        "config_json": "config.json",
        "note": "Full CLI, environment, data, model, and optimizer hyperparameters for this run.",
    }
    (out_dir / "variant_curves.json").write_text(json.dumps(_json_safe(payload), indent=2))
    print(f"[done] wrote {out_dir / 'config.json'} and figures under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
