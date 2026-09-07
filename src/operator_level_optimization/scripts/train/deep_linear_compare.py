from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from operator_level_optimization.core.optim.kfac import KFAC
from operator_level_optimization.core.optim.muon import Muon
from operator_level_optimization.core.optim.shampoo import Shampoo
from operator_level_optimization.core.optim.soap import SOAP
from operator_level_optimization.scripts.utils.io import get_device
from operator_level_optimization.scripts.utils.plotting import save_fig
from operator_level_optimization.scripts.utils.training import EarlyStopping


def ginibre_sn1(shape: tuple[int, int], device: torch.device, dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:
    w = torch.randn(shape, device=device, dtype=dtype, generator=g)
    smax = torch.linalg.svdvals(w).max().clamp(min=1e-12)
    return w / smax


def xavier_gaussian(shape: tuple[int, int], device: torch.device, dtype: torch.dtype, g: torch.Generator) -> torch.Tensor:
    fan_in = float(shape[1])
    return torch.randn(shape, device=device, dtype=dtype, generator=g) / fan_in**0.5


def compose_operator(weights: list[torch.Tensor]) -> torch.Tensor:
    p = weights[-1]
    for i in range(len(weights) - 2, -1, -1):
        p = p @ weights[i]
    return p


def _product_excluding(weights: list[torch.Tensor], k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (A_k, B_k) with P = A_k @ W_k @ B_k."""
    d = weights[0].shape[0]
    A = torch.eye(d, device=weights[0].device, dtype=weights[0].dtype)
    for i in range(len(weights) - 1, k, -1):
        A = A @ weights[i]
    B = torch.eye(d, device=weights[0].device, dtype=weights[0].dtype)
    for i in range(k - 1, -1, -1):
        B = B @ weights[i]
    return A, B


def _solve_delta_exact(M: torch.Tensor, N: torch.Tensor, G: torch.Tensor, lam: float) -> torch.Tensor:
    """Solve M Δ N + lam Δ = G via modewise SVD filtering."""
    # M,N are symmetric PSD (A^T A and B B^T).
    # Regularize before eigh: scale eps to the matrix norm so the shift is
    # meaningful under extreme ill-conditioning (depth-256 xavier, float64).
    eps_rel = 1e-6
    eye_m = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    eye_n = torch.eye(N.shape[0], device=N.device, dtype=N.dtype)
    eval_m, U_m = torch.linalg.eigh(M + eps_rel * float(M.diagonal().max().clamp(min=1e-30)) * eye_m)
    eval_n, U_n = torch.linalg.eigh(N + eps_rel * float(N.diagonal().max().clamp(min=1e-30)) * eye_n)
    G_t = U_m.T @ G @ U_n
    denom = eval_m.unsqueeze(1) * eval_n.unsqueeze(0) + lam
    # Moore-Penrose style for lam=0: zero out near-null modes.
    eps = 1e-12
    X_t = torch.where(denom.abs() > eps, G_t / denom, torch.zeros_like(G_t))
    return U_m @ X_t @ U_n.T


def als_collapse_metric(M: torch.Tensor | None, N: torch.Tensor | None) -> float:
    """Degeneracy monitor for the modewise solve: min-eig(M) * min-eig(N).

    The filter (6) loses its operator-correcting structure exactly when
    min_i sigma_A,i^2 * min_j sigma_B,j^2 << lam, where the update degenerates
    to G / lam. A None context is an identity (edge layer), contributing 1.
    """
    out = 1.0
    for C in (M, N):
        if C is not None:
            if not bool(torch.isfinite(C).all()):
                return 0.0
            out *= float(torch.linalg.eigvalsh(C)[0].clamp(min=0.0).item())
    return out


def als_exact_shared_target_step(
    weights: list[torch.Tensor],
    p_tgt: torch.Tensor,
    lam: float,
    n_sweeps: int,
    reverse: bool = True,
    warmstart_identity: bool = True,
    collapse_monitor: list | None = None,
) -> list[torch.Tensor]:
    """Deep-linear ALS step for a single shared operator target.

    If collapse_monitor is a list, one dict per layer solve is appended with
    the per-layer degeneracy metric min-eig(M_k) * min-eig(N_k) vs lam.
    """
    w_work = [w.clone() for w in weights]
    if warmstart_identity:
        for k in range(len(w_work)):
            d_out, d_in = w_work[k].shape
            I = torch.zeros_like(w_work[k])
            m = min(d_out, d_in)
            idx = torch.arange(m, device=w_work[k].device)
            I[idx, idx] = 1.0
            # Delta-init semantics: W + Delta starts at identity-like map.
            w_work[k] = I
    order = list(range(len(w_work) - 1, -1, -1)) if reverse else list(range(len(w_work)))
    for _ in range(n_sweeps):
        for k in order:
            A_k, B_k = _product_excluding(w_work, k)
            R_k = p_tgt - A_k @ w_work[k] @ B_k
            M_k = A_k.T @ A_k
            N_k = B_k @ B_k.T
            G_k = A_k.T @ R_k @ B_k.T
            if collapse_monitor is not None:
                m_min = als_collapse_metric(M_k, N_k)
                collapse_monitor.append(
                    {"layer": k, "min_gram_product": m_min, "lam": float(lam),
                     "degenerate": bool(m_min < 10.0 * lam)}
                )
            dW = _solve_delta_exact(M_k, N_k, G_k, lam)
            w_work[k] = w_work[k] + dW
    return w_work


class DeepLinearModel(nn.Module):
    def __init__(self, depth: int, d: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def mse_grad_target(p: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    d_out = p.shape[0]
    pred = x @ p.T
    # For loss = mean((pred - y)^2) over batch and output dimensions:
    # dL/dP = (2 / (n * d_out)) * (pred - y)^T X
    grad_p = (2.0 / (n * d_out)) * (pred - y).T @ x
    return grad_p


def dc_node_solve(
    u0: torch.Tensor,
    v0: torch.Tensor,
    t: torch.Tensor,
    lam: float,
    n_alt: int,
    init_mode: str,
    init_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if init_mode == "identity_delta":
        p, r = u0.shape
        r2, q = v0.shape
        i_pr = torch.zeros((p, r), device=u0.device, dtype=u0.dtype)
        i_rq = torch.zeros((r2, q), device=v0.device, dtype=v0.dtype)
        m1 = min(p, r)
        m2 = min(r2, q)
        i_pr[torch.arange(m1), torch.arange(m1)] = 1.0
        i_rq[torch.arange(m2), torch.arange(m2)] = 1.0
        u = u0 + init_scale * torch.norm(u0, p="fro") * i_pr
        v = v0 + init_scale * torch.norm(v0, p="fro") * i_rq
    else:
        u = u0.clone()
        v = v0.clone()
    eye_u = torch.eye(u.shape[0], device=u.device, dtype=u.dtype)
    eye_v = torch.eye(v.shape[0], device=v.device, dtype=v.dtype)
    for _ in range(n_alt):
        # U update
        rhs_u = t @ v.T + lam * u0
        lhs_u = v @ v.T + lam * eye_v
        u = rhs_u @ torch.linalg.pinv(lhs_u)
        # V update
        rhs_v = u.T @ t + lam * v0
        lhs_v = u.T @ u + lam * eye_u
        v = torch.linalg.pinv(lhs_v) @ rhs_v
    return u, v


def product(weights: list[torch.Tensor], a: int, b: int) -> torch.Tensor:
    p = weights[b]
    for i in range(b - 1, a - 1, -1):
        p = p @ weights[i]
    return p


def dc_apply_recursive(
    weights: list[torch.Tensor],
    a: int,
    b: int,
    t: torch.Tensor,
    lam: float,
    n_alt: int,
    init_mode: str,
    init_scale: float,
) -> None:
    if a == b:
        weights[a] = t
        return
    m = (a + b) // 2
    u0 = product(weights, m + 1, b)
    v0 = product(weights, a, m)
    u, v = dc_node_solve(
        u0=u0, v0=v0, t=t, lam=lam, n_alt=n_alt, init_mode=init_mode, init_scale=init_scale
    )
    dc_apply_recursive(weights, m + 1, b, u, lam, n_alt, init_mode, init_scale)
    dc_apply_recursive(weights, a, m, v, lam, n_alt, init_mode, init_scale)


def run_method(
    method: str,
    depth: int,
    d: int,
    n: int,
    steps: int,
    lr: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    target_mode: str,
    dc_lam: float,
    dc_alt: int,
    dc_init_mode: str,
    dc_init_scale: float,
    spec_every: int,
    spec_steps: list[int] | None,
    target_kind: str,
    init_mode: str,
    als_lam: float,
    als_sweeps: int,
    early_stop_patience: int | None = None,
    early_stop_rel_tol: float = 1e-4,
    early_stop_abs_tol: float = 1e-10,
    early_stop_min_steps: int = 200,
    target_scale: float = 1.0,
    init_diag_scale: float = 1.0,
) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, d, device=device, dtype=dtype, generator=g)
    if target_kind == "orth":
        q1, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
        q2, _ = torch.linalg.qr(torch.randn((d, d), device=device, dtype=dtype, generator=g))
        p_star = q1 @ q2.T
    else:
        p_star = ginibre_sn1((d, d), device=device, dtype=dtype, g=g)
    p_star = p_star * target_scale
    y = x @ p_star.T

    model = DeepLinearModel(depth=depth, d=d).to(device=device, dtype=dtype)
    with torch.no_grad():
        if init_mode == "identity":
            for layer in model.layers:
                layer.weight.copy_(
                    init_diag_scale * torch.eye(d, device=device, dtype=dtype)
                )
        else:
            init_fn = ginibre_sn1 if init_mode == "ginibre_sn1" else xavier_gaussian
            for layer in model.layers:
                layer.weight.copy_(init_fn((d, d), device=device, dtype=dtype, g=g))
    params = list(model.parameters())

    if method == "heavyball":
        optim = torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif method == "adam":
        optim = torch.optim.Adam(params, lr=lr)
    elif method == "muon":
        optim = Muon(params, lr=lr)
    elif method == "shampoo":
        optim = Shampoo(params, lr=lr, beta=0.9, momentum=0.0, weight_decay=0.0, eps=1e-4)
    elif method == "soap":
        optim = SOAP(params, lr=lr, weight_decay=0.0, correct_bias=True)
    elif method == "kfac":
        optim = KFAC(
            params,
            lr=lr,
            factor_decay=0.95,
            damping=0.1,
            momentum=0.0,
            weight_decay=0.0,
            inv_floor=1e-2,
        )
    elif method == "als_exact":
        optim = None  # handled directly via als_exact_shared_target_step in training loop
    elif method == "dc":
        optim = None
    else:
        raise ValueError(f"unknown method {method}")
    if optim is not None and hasattr(optim, "attach_hooks"):
        optim.attach_hooks(model)

    hist = []
    spectra = {}
    target_residual_history = []
    early_stop_triggered_at: int | None = None
    es = EarlyStopping(early_stop_patience, early_stop_rel_tol, early_stop_abs_tol, early_stop_min_steps)
    for t in range(steps + 1):
        p_tgt_step = None
        with torch.no_grad():
            p = compose_operator([layer.weight.data for layer in model.layers])
            rel = torch.norm(p - p_star) / torch.norm(p_star)
            loss = torch.mean((x @ p.T - y) ** 2)
            hist.append((t, float(rel.item()), float(loss.item())))
            capture = (t % spec_every == 0 or t == steps)
            if spec_steps is not None and len(spec_steps) > 0:
                capture = capture or (t in spec_steps)
            if capture and bool(torch.isfinite(p).all()):
                spectra[int(t)] = torch.linalg.svdvals(p).detach().cpu().numpy()

        mse = float(loss.item())
        if es.check(mse, t):
            early_stop_triggered_at = es.triggered_at
            print(f"[early_stop] method={method} t={t} stall={es._stall} best_mse={es._best:.6e}")
            break

        # ALS reaches numerical zero quickly; no need to wait on stall logic.
        if method == "als_exact" and mse < 1e-20 and t >= max(50, early_stop_min_steps // 2):
            early_stop_triggered_at = t
            print(f"[early_stop] als_exact numerical floor t={t} mse={mse:.3e}")
            break

        if t == steps:
            break

        if method == "dc":
            with torch.no_grad():
                p_curr = compose_operator([layer.weight.data for layer in model.layers])
                if target_mode != "mse_grad":
                    raise ValueError("only mse_grad target_mode is implemented")
                grad_p = mse_grad_target(p_curr, x, y)
                p_tgt = p_curr - lr * grad_p
                w_list = [layer.weight.data.clone() for layer in model.layers]
                dc_apply_recursive(
                    w_list,
                    0,
                    depth - 1,
                    p_tgt,
                    lam=dc_lam,
                    n_alt=dc_alt,
                    init_mode=dc_init_mode,
                    init_scale=dc_init_scale,
                )
                for layer, w_new in zip(model.layers, w_list):
                    layer.weight.data.copy_(w_new)
        elif method == "als_exact":
            with torch.no_grad():
                p_curr = compose_operator([layer.weight.data for layer in model.layers])
                if target_mode != "mse_grad":
                    raise ValueError("only mse_grad target_mode is implemented")
                grad_p = mse_grad_target(p_curr, x, y)
                p_tgt_step = p_curr - lr * grad_p
                w_new = als_exact_shared_target_step(
                    [layer.weight.data for layer in model.layers],
                    p_tgt=p_tgt_step,
                    lam=als_lam,
                    n_sweeps=als_sweeps,
                    reverse=True,
                    warmstart_identity=False,
                )
                for layer, w_k in zip(model.layers, w_new):
                    layer.weight.data.copy_(w_k)
                p_after = compose_operator([layer.weight.data for layer in model.layers])
                tgt_abs = torch.norm(p_after - p_tgt_step)
                tgt_norm = torch.norm(p_tgt_step).clamp(min=1e-30)
                tgt_rel = tgt_abs / tgt_norm
                target_residual_history.append(
                    (t + 1, float(tgt_rel.item()), float(tgt_abs.item()), float(tgt_norm.item()))
                )
        else:
            with torch.no_grad():
                p_curr = compose_operator([layer.weight.data for layer in model.layers])
                if target_mode != "mse_grad":
                    raise ValueError("only mse_grad target_mode is implemented")
                grad_p = mse_grad_target(p_curr, x, y)
                p_tgt_step = p_curr - lr * grad_p
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            optim.step()
            if method == "als_exact":
                with torch.no_grad():
                    p_after = compose_operator([layer.weight.data for layer in model.layers])
                    tgt_abs = torch.norm(p_after - p_tgt_step)
                    tgt_norm = torch.norm(p_tgt_step).clamp(min=1e-30)
                    tgt_rel = tgt_abs / tgt_norm
                    target_residual_history.append(
                        (t + 1, float(tgt_rel.item()), float(tgt_abs.item()), float(tgt_norm.item()))
                    )

    # Always store singular values at the **final** trained operator $P$ (overwrites
    # duplicate step if spec_every already captured it).  Without this, early-stopped
    # runs often have no spectrum at $t>0$ when spec_every is large.
    with torch.no_grad():
        p_final = compose_operator([layer.weight.data for layer in model.layers])
        final_t = int(hist[-1][0])
        if bool(torch.isfinite(p_final).all()):
            spectra[final_t] = torch.linalg.svdvals(p_final).detach().cpu().numpy()

    out = {
        "method": method,
        "history": hist,
        "spectra": {k: v.tolist() for k, v in spectra.items()},
    }
    if early_stop_triggered_at is not None:
        out["early_stop_triggered_at"] = early_stop_triggered_at
        out["early_stop_steps_run"] = len(hist) - 1
    if target_residual_history:
        out["target_residual_history"] = target_residual_history
    return out


# (method_key, display label, color) — used for the auto-generated loss figure
_METHOD_PLOT = {
    "heavyball": ("Heavy Ball", "#1f77b4"),
    "adam": ("Adam", "#ff7f0e"),
    "muon": ("Muon", "#2ca02c"),
    "shampoo": ("Shampoo", "#d62728"),
    "soap": ("SOAP", "#8c564b"),
    "kfac": ("K-FAC", "#e377c2"),
    "als_exact": ("ALS-Exact", "#9467bd"),
    "dc": ("D&C", "#7f7f7f"),
}


def save_loss_over_training_figure(
    out_dir: Path,
    results: dict,
    method_order: list[str],
    *,
    filename: str = "loss_over_training.png",
    yscale: str = "log",
    clip_max: float | None = None,
) -> Path | None:
    """Plot training MSE vs step; save PNG under ``out_dir / filename``.

    ``yscale`` is ``\"log\"`` or ``\"linear\"`` (linear helps compare baselines in a
    narrow MSE band when ALS is omitted).

    ``clip_max``: if > 0, displayed values are ``min(MSE, clip_max)`` only for drawing,
    and the y-axis top is limited so spikes do not compress other curves.

    Returns ``None`` if nothing was plotted (empty ``method_order`` or no histories).
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    n_plotted = 0
    for m in method_order:
        if m not in results:
            continue
        hist = results[m].get("history", [])
        if not hist:
            continue
        arr = np.array(hist, dtype=float)
        steps = arr[:, 0]
        loss = np.clip(arr[:, 2], 1e-30, None)
        if clip_max is not None and clip_max > 0:
            loss = np.minimum(loss, clip_max)
        label, color = _METHOD_PLOT.get(m, (m, None))
        ax.plot(
            steps,
            loss,
            label=label,
            color=color,
            linewidth=2.0,
        )
        n_plotted += 1
    if n_plotted == 0:
        plt.close(fig)
        return None
    if yscale == "log":
        ax.set_yscale("log")
    elif yscale == "linear":
        ax.set_yscale("linear")
    else:
        raise ValueError(f"yscale must be 'log' or 'linear', got {yscale!r}")
    if clip_max is not None and clip_max > 0:
        if yscale == "linear":
            ax.set_ylim(bottom=0.0, top=clip_max * 1.02)
        else:
            ax.set_ylim(top=clip_max * 1.02)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training MSE (mean over batch and output dims)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    path = out_dir / filename
    save_fig(fig, path, dpi=180, bbox_inches="tight")
    return path


def _run_method_kwargs(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    return dict(
        depth=args.depth,
        d=args.d,
        n=args.n,
        steps=args.steps,
        seed=args.seed,
        device=device,
        dtype=dtype,
        target_mode=args.target_mode,
        dc_lam=args.dc_lam,
        dc_alt=args.dc_alt,
        dc_init_mode=args.dc_init_mode,
        dc_init_scale=args.dc_init_scale,
        spec_every=args.spec_every,
        spec_steps=args.spec_steps if args.spec_steps else None,
        target_kind=args.target_kind,
        init_mode=args.init_mode,
        als_lam=args.als_lam,
        als_sweeps=args.als_sweeps,
        early_stop_patience=args.early_stop_patience,
        early_stop_rel_tol=args.early_stop_rel_tol,
        early_stop_abs_tol=args.early_stop_abs_tol,
        early_stop_min_steps=args.early_stop_min_steps,
    )


def run_weight_lr_sweep(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict, dict, dict, dict]:
    """Sweep learning rates for weight-space optimizers; keep best curve per method.

    Reuses ``als_exact`` from ``out_dir / results.json`` when present (no ALS re-run).
    Returns ``(results, method_cfg, sweep_raw, sweep_details)``.
    """
    als_key = "als_exact"
    sweeps = list(args.weight_lr_sweep)
    weight_methods = [m for m in args.methods if m != als_key]
    if not weight_methods:
        raise ValueError(
            "--weight_lr_sweep requires at least one method other than als_exact."
        )

    prev_results: dict | None = None
    prev_methods: dict | None = None
    res_path = out_dir / "results.json"
    if res_path.exists():
        prev = json.loads(res_path.read_text())
        prev_results = prev.get("results")
        prev_methods = prev.get("methods")

    als_block: dict | None = None
    als_cfg: dict | None = None
    if als_key in args.methods:
        if prev_results and als_key in prev_results:
            als_block = copy.deepcopy(prev_results[als_key])
            if prev_methods and als_key in prev_methods:
                als_cfg = copy.deepcopy(prev_methods[als_key])
        else:
            print(
                f"[warn] --weight_lr_sweep: no prior {als_key} in {res_path}; "
                "ALS omitted from merged results and plot."
            )

    sweep_mse: dict[str, dict[str, float]] = {}
    sweep_raw: dict[str, dict[str, dict]] = {}
    results: dict = {}
    method_cfg: dict = {}

    for m in weight_methods:
        sweep_mse[m] = {}
        sweep_raw[m] = {}
        best_loss = float("inf")
        best_lr: float | None = None
        best_res: dict | None = None
        base_kw = _run_method_kwargs(args, device, dtype)
        for lr in sweeps:
            print(f"[sweep] method={m} lr={lr}")
            res = run_method(method=m, lr=lr, **base_kw)
            sweep_raw[m][str(lr)] = res
            fl = float(res["history"][-1][2])
            sweep_mse[m][str(lr)] = fl
            if math.isfinite(fl) and fl < best_loss:
                best_loss = fl
                best_lr = lr
                best_res = res
        if best_res is None or best_lr is None:
            raise RuntimeError(
                f"No finite final MSE for method={m!r} across lr sweep {sweeps}."
            )
        results[m] = best_res
        method_cfg[m] = {
            "optimizer": m,
            "effective_lr": best_lr,
            "weight_lr_sweep_candidates": sweeps,
            "weight_lr_sweep_final_mse_by_lr": sweep_mse[m],
        }

    if als_block is not None:
        results[als_key] = als_block
        method_cfg[als_key] = als_cfg or {
            "optimizer": als_key,
            "effective_lr": args.lr_als if args.lr_als is not None else args.lr,
        }

    details = {
        "candidates": sweeps,
        "per_method_final_mse": sweep_mse,
    }
    return results, method_cfg, sweep_raw, details


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_gd", type=float, default=None)
    ap.add_argument("--lr_adam", type=float, default=None)
    ap.add_argument("--lr_muon", type=float, default=None)
    ap.add_argument("--lr_shampoo", type=float, default=None)
    ap.add_argument("--lr_soap", type=float, default=None)
    ap.add_argument("--lr_kfac", type=float, default=None)
    ap.add_argument(
        "--lr_als",
        type=float,
        default=1.0,
        help="Outer operator-step size for als_exact.",
    )
    ap.add_argument("--lr_dc", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_mode", choices=["mse_grad"], default="mse_grad")
    ap.add_argument("--target_kind", choices=["orth", "ginibre"], default="orth")
    ap.add_argument("--init_mode", choices=["ginibre_sn1", "xavier", "identity"], default="ginibre_sn1")
    ap.add_argument("--methods", nargs="+", default=["heavyball", "adam", "muon", "kfac", "shampoo", "soap", "als_exact"])
    ap.add_argument("--als_lam", type=float, default=1e-4)
    ap.add_argument("--als_sweeps", type=int, default=4)
    ap.add_argument("--dc_lam", type=float, default=0.0)
    ap.add_argument("--dc_alt", type=int, default=1)
    ap.add_argument("--dc_init_mode", choices=["plain", "identity_delta"], default="identity_delta")
    ap.add_argument("--dc_init_scale", type=float, default=1.0)
    ap.add_argument("--spec_every", type=int, default=250)
    ap.add_argument("--spec_steps", type=int, nargs="*", default=[])
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument(
        "--no_auto_plot",
        action="store_true",
        help="Disable writing loss_over_training.png (plots are on by default).",
    )
    ap.add_argument(
        "--weight_lr_sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="LR",
        help=(
            "For each non-ALS method, run this many learning rates and keep the run "
            "with lowest final training MSE. Merges als_exact from existing results.json "
            "(no ALS re-run). Writes weight_lr_sweep_raw.json with all trajectories."
        ),
    )
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help=(
            "Stop after this many steps without relative MSE improvement (disabled if unset). "
            "Uses --early_stop_rel_tol / --early_stop_abs_tol vs best-so-far MSE."
        ),
    )
    ap.add_argument(
        "--early_stop_rel_tol",
        type=float,
        default=1e-4,
        help="Relative improvement threshold vs best MSE to reset stall counter.",
    )
    ap.add_argument(
        "--early_stop_abs_tol",
        type=float,
        default=1e-10,
        help="Absolute improvement threshold (combined with relative).",
    )
    ap.add_argument(
        "--early_stop_min_steps",
        type=int,
        default=200,
        help="Minimum steps before early stopping can trigger.",
    )
    ap.add_argument(
        "--loss_plot_yscale",
        choices=["linear", "log"],
        default="linear",
        help="Y-axis scale for loss_over_training.png (default: linear).",
    )
    ap.add_argument(
        "--loss_plot_clip_max",
        type=float,
        default=1.0,
        help=(
            "Plotting only: cap displayed MSE at this value so spikes / divergent traces "
            "(e.g. K-FAC) do not shrink the rest of the curves; set <=0 to disable clipping."
        ),
    )
    ap.add_argument(
        "--also_plot_baselines_only",
        action="store_true",
        help="Also write loss_over_training_baselines_only.png (linear y).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    dtype = torch.float64

    sweep_raw: dict | None = None
    sweep_details: dict | None = None
    if args.weight_lr_sweep is not None:
        results, method_cfg, sweep_raw, sweep_details = run_weight_lr_sweep(
            args, out_dir, device, dtype
        )
    else:
        results = {}
        method_cfg = {}
        base_kw = _run_method_kwargs(args, device, dtype)
        for m in args.methods:
            print(f"[run] method={m}")
            lr = args.lr
            if m == "heavyball" and args.lr_gd is not None:
                lr = args.lr_gd
            elif m == "adam" and args.lr_adam is not None:
                lr = args.lr_adam
            elif m == "muon" and args.lr_muon is not None:
                lr = args.lr_muon
            elif m == "shampoo" and args.lr_shampoo is not None:
                lr = args.lr_shampoo
            elif m == "soap" and args.lr_soap is not None:
                lr = args.lr_soap
            elif m == "kfac" and args.lr_kfac is not None:
                lr = args.lr_kfac
            elif m == "als_exact" and args.lr_als is not None:
                lr = args.lr_als
            elif m == "dc" and args.lr_dc is not None:
                lr = args.lr_dc
            method_cfg[m] = {
                "optimizer": m,
                "effective_lr": lr,
                "dc_lam": args.dc_lam if m == "dc" else None,
                "dc_alt": args.dc_alt if m == "dc" else None,
                "dc_init_mode": args.dc_init_mode if m == "dc" else None,
                "dc_init_scale": args.dc_init_scale if m == "dc" else None,
            }
            results[m] = run_method(method=m, lr=lr, **base_kw)

    payload = {
        "config": {
            "depth": args.depth,
            "d": args.d,
            "n": args.n,
            "steps": args.steps,
            "lr": args.lr,
            "seed": args.seed,
            "target_mode": args.target_mode,
            "target_kind": args.target_kind,
            "init_mode": args.init_mode,
            "methods": args.methods,
            "lr_als": args.lr_als,
            "als_lam": args.als_lam,
            "als_sweeps": args.als_sweeps,
            "dc_lam": args.dc_lam,
            "dc_alt": args.dc_alt,
            "dc_init_mode": args.dc_init_mode,
            "dc_init_scale": args.dc_init_scale,
            "spec_every": args.spec_every,
            "spec_steps": args.spec_steps,
            "device": str(device),
            "dtype": str(dtype),
            "weight_lr_sweep": args.weight_lr_sweep,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_rel_tol": args.early_stop_rel_tol,
            "early_stop_abs_tol": args.early_stop_abs_tol,
            "early_stop_min_steps": args.early_stop_min_steps,
            "loss_plot_yscale": args.loss_plot_yscale,
            "loss_plot_clip_max": args.loss_plot_clip_max,
        },
        "methods": method_cfg,
        "results": results,
    }
    if sweep_details is not None:
        payload["weight_lr_sweep_details"] = sweep_details
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    # Explicit reproducibility file requested by paper workflow.
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "distribution": "Gaussian X with synthetic linear labels Y = X @ P_star^T",
                    "n_samples": args.n,
                    "dimension": args.d,
                    "target_kind": args.target_kind,
                    "target_mode": args.target_mode,
                    "seed": args.seed,
                },
                "model": {
                    "type": "deep_linear",
                    "depth": args.depth,
                    "init": args.init_mode,
                    "dtype": str(dtype),
                    "device": str(device),
                },
                "global_optimization": {
                    "steps": args.steps,
                    "spec_every": args.spec_every,
                    "spec_steps": args.spec_steps,
                },
                "methods": method_cfg,
            },
            indent=2,
        )
    )
    print(f"[done] wrote {out_dir / 'results.json'}")
    if sweep_raw is not None:
        (out_dir / "weight_lr_sweep_raw.json").write_text(
            json.dumps(sweep_raw, indent=2)
        )
        print(f"[done] wrote {out_dir / 'weight_lr_sweep_raw.json'}")
    if not args.no_auto_plot:
        plot_order = [m for m in args.methods if m in results]
        clip = (
            args.loss_plot_clip_max if args.loss_plot_clip_max > 0 else None
        )
        plot_path = save_loss_over_training_figure(
            out_dir,
            results,
            plot_order,
            yscale=args.loss_plot_yscale,
            clip_max=clip,
        )
        if plot_path is not None:
            print(f"[done] wrote {plot_path}")
        if args.also_plot_baselines_only:
            baseline_order = [m for m in plot_order if m != "als_exact"]
            if baseline_order:
                p_bl = save_loss_over_training_figure(
                    out_dir,
                    results,
                    baseline_order,
                    filename="loss_over_training_baselines_only.png",
                    yscale="linear",
                    clip_max=clip,
                )
                if p_bl is not None:
                    print(f"[done] wrote {p_bl}")


if __name__ == "__main__":
    main()
