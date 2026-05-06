from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from operator_level_optimization.models.fgln import FGLN, MaskedOperatorALS, compute_P_fgln, init_weights
from operator_level_optimization.core.optim.kfac import KFAC
from operator_level_optimization.core.optim.muon import Muon
from operator_level_optimization.core.optim.shampoo import Shampoo
from operator_level_optimization.core.optim.soap import SOAP


_METHOD_PLOT = {
    "heavyball": ("Heavy Ball", "#1f77b4"),
    "adam": ("Adam", "#ff7f0e"),
    "muon": ("Muon", "#2ca02c"),
    "shampoo": ("Shampoo", "#d62728"),
    "soap": ("SOAP", "#8c564b"),
    "kfac": ("K-FAC", "#e377c2"),
    "als_exact": ("ALS-Exact", "#9467bd"),
}


def make_dataset(d: int, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    u, _, vh = torch.linalg.svd(torch.randn(d, d, generator=g))
    p_star = u @ vh  # isotropic (all singular values 1)
    y = x @ p_star.T
    return x, y, p_star


def save_loss_plot(
    out_dir: Path,
    results: dict,
    method_order: list[str],
    *,
    filename: str = "loss_over_training.png",
    yscale: str = "linear",
    clip_max: float | None = 1.0,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for m in method_order:
        hist = results[m]["history"]
        arr = np.asarray(hist, dtype=float)
        steps = arr[:, 0]
        loss = np.clip(arr[:, 2], 1e-30, None)
        if clip_max is not None and clip_max > 0:
            loss = np.minimum(loss, clip_max)
        label, color = _METHOD_PLOT[m]
        ax.plot(steps, loss, label=label, color=color, linewidth=2.0)
    ax.set_yscale(yscale)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training MSE (mean over batch and output dims)")
    if clip_max is not None and clip_max > 0:
        if yscale == "linear":
            ax.set_ylim(bottom=0.0, top=clip_max * 1.02)
        else:
            ax.set_ylim(top=clip_max * 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    out = out_dir / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def run_method(
    *,
    method: str,
    net0: FGLN,
    p_star: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    als_lam: float,
    als_sweeps: int,
    als_warmstart_once: bool,
    als_lam_anchor_post_warmstart: bool,
    early_stop_patience: int | None,
    early_stop_rel_tol: float,
    early_stop_abs_tol: float,
    early_stop_min_steps: int,
    spec_every: int,
) -> dict:
    model = copy.deepcopy(net0)
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
        optim = MaskedOperatorALS(
            model,
            P_star=p_star,
            lr=lr,
            lam=als_lam,
            n_sweeps=als_sweeps,
            als_reverse_sweep=True,
            als_gateperm_warmstart=True,
            als_gateperm_warmstart_once=als_warmstart_once,
            als_lam_anchor_post_warmstart=als_lam_anchor_post_warmstart,
            normalize_contexts=True,
            damping=True,
            damping_beta=0.5,
            damping_min=1e-6,
            damping_max_trials=20,
            gauge_balance=True,
            gauge_balance_passes=1,
        )
    else:
        raise ValueError(f"unknown method {method}")

    if method != "als_exact" and hasattr(optim, "attach_hooks"):
        optim.attach_hooks(model)

    p_star_norm = torch.norm(p_star).clamp(min=1e-30)
    hist = []
    spectra: dict[int, list[float]] = {}
    best_mse = float("inf")
    stall = 0
    early_stop_triggered_at: int | None = None
    for t in range(steps + 1):
        with torch.no_grad():
            p = compute_P_fgln(model)
            rel = torch.norm(p - p_star) / p_star_norm
            loss = torch.mean((x @ p.T - y) ** 2)
            hist.append((t, float(rel.item()), float(loss.item())))
            if t == 0 or (spec_every > 0 and t % spec_every == 0):
                spectra[int(t)] = torch.linalg.svdvals(p).detach().cpu().tolist()
        if t == steps:
            break
        mse_val = float(loss.item())
        if best_mse == float("inf"):
            best_mse = mse_val
        else:
            thr = max(early_stop_rel_tol * best_mse, early_stop_abs_tol)
            if mse_val < best_mse - thr:
                best_mse = mse_val
                stall = 0
            elif t >= early_stop_min_steps and early_stop_patience is not None:
                stall += 1
        if (
            early_stop_patience is not None
            and t >= early_stop_min_steps
            and stall >= early_stop_patience
        ):
            early_stop_triggered_at = t
            break
        if method == "als_exact":
            try:
                n = x.shape[0]
                d_out = p.shape[0]
                pred = x @ p.T
                grad_p = (2.0 / (n * d_out)) * (pred - y).T @ x
                p_tgt_step = p - float(lr) * grad_p
                optim.step(P_tgt_override=p_tgt_step)
            except (FloatingPointError, RuntimeError):
                # Keep a complete figure even if ALS becomes numerically unstable at some step.
                with torch.no_grad():
                    p = compute_P_fgln(model)
                    rel = torch.norm(p - p_star) / p_star_norm
                    loss = torch.mean((x @ p.T - y) ** 2)
                    hist.append((t + 1, float(rel.item()), float(loss.item())))
                break
        else:
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            mse = torch.mean((pred - y) ** 2)
            mse.backward()
            optim.step()
    # Always save final spectrum (including early-stop/failure endpoint).
    with torch.no_grad():
        p_final = compute_P_fgln(model)
        final_t = int(hist[-1][0])
        spectra[final_t] = torch.linalg.svdvals(p_final).detach().cpu().tolist()

    out = {"method": method, "history": hist, "spectra": {str(k): v for k, v in sorted(spectra.items())}}
    if early_stop_triggered_at is not None:
        out["early_stop_triggered_at"] = early_stop_triggered_at
        out["early_stop_steps_run"] = len(hist) - 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate_seed", type=int, default=0)
    ap.add_argument("--init_seed", type=int, default=0)
    ap.add_argument("--init_mode", choices=["orth", "xavier"], default="xavier")
    ap.add_argument("--methods", nargs="+", default=["heavyball", "adam", "muon", "kfac", "shampoo", "soap", "als_exact"])
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr_gd", type=float, default=1e-5)
    ap.add_argument("--lr_adam", type=float, default=1e-5)
    ap.add_argument("--lr_muon", type=float, default=1e-5)
    ap.add_argument("--lr_kfac", type=float, default=1e-6)
    ap.add_argument("--lr_shampoo", type=float, default=1e-5)
    ap.add_argument("--lr_soap", type=float, default=1e-5)
    ap.add_argument("--lr_als", type=float, default=1.0)
    ap.add_argument("--als_lam", type=float, default=1e-4)
    ap.add_argument("--als_sweeps", type=int, default=4)
    ap.add_argument("--als_gateperm_warmstart_once", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--als_lam_anchor_post_warmstart", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    ap.add_argument("--early_stop_patience", type=int, default=500)
    ap.add_argument("--early_stop_rel_tol", type=float, default=1e-4)
    ap.add_argument("--early_stop_abs_tol", type=float, default=1e-10)
    ap.add_argument("--early_stop_min_steps", type=int, default=500)
    ap.add_argument("--spec_every", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float64

    x, y, p_star = make_dataset(args.d, args.n, args.seed)
    x, y, p_star = x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype), p_star.to(device=device, dtype=dtype)

    net0 = FGLN(args.d, depth=args.depth, p_gate=args.p, seed=args.gate_seed).to(device=device, dtype=dtype)
    init_weights(net0, init=args.init_mode, seed=args.init_seed, target_norm=float(p_star.norm().item()), rescale_mode="all")

    per_method_lr = {
        "heavyball": args.lr_gd,
        "adam": args.lr_adam,
        "muon": args.lr_muon,
        "kfac": args.lr_kfac,
        "shampoo": args.lr_shampoo,
        "soap": args.lr_soap,
        "als_exact": args.lr_als,
    }

    results = {}
    method_cfg = {}
    for m in args.methods:
        lr = per_method_lr.get(m, args.lr)
        print(f"[run] method={m} lr={lr}")
        results[m] = run_method(
            method=m,
            net0=net0,
            p_star=p_star,
            x=x,
            y=y,
            steps=args.steps,
            lr=lr,
            als_lam=args.als_lam,
            als_sweeps=args.als_sweeps,
            als_warmstart_once=args.als_gateperm_warmstart_once,
            als_lam_anchor_post_warmstart=args.als_lam_anchor_post_warmstart,
            early_stop_patience=args.early_stop_patience,
            early_stop_rel_tol=args.early_stop_rel_tol,
            early_stop_abs_tol=args.early_stop_abs_tol,
            early_stop_min_steps=args.early_stop_min_steps,
            spec_every=args.spec_every,
        )
        method_cfg[m] = {"optimizer": m, "effective_lr": lr}

    payload = {
        "config": {
            "d": args.d,
            "depth": args.depth,
            "n": args.n,
            "steps": args.steps,
            "p": args.p,
            "seed": args.seed,
            "gate_seed": args.gate_seed,
            "init_seed": args.init_seed,
            "init_mode": args.init_mode,
            "als_lam": args.als_lam,
            "als_sweeps": args.als_sweeps,
            "als_gateperm_warmstart_once": args.als_gateperm_warmstart_once,
            "als_lam_anchor_post_warmstart": args.als_lam_anchor_post_warmstart,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_rel_tol": args.early_stop_rel_tol,
            "early_stop_abs_tol": args.early_stop_abs_tol,
            "early_stop_min_steps": args.early_stop_min_steps,
            "spec_every": args.spec_every,
            "device": str(device),
            "dtype": str(dtype),
        },
        "methods": method_cfg,
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "config.json").write_text(json.dumps(payload["config"], indent=2))
    fig = save_loss_plot(out_dir, results, [m for m in args.methods if m in results], yscale="linear", clip_max=1.0)
    print(f"[done] wrote {out_dir / 'results.json'}")
    print(f"[done] wrote {fig}")


if __name__ == "__main__":
    main()
