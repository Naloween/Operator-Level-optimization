"""Smoke-test operator / ALS variants used in the paper appendix.

Exits non-zero if any implemented variant raises. Prints a summary table and
writes ``outputs/smoke/variants/smoke_variants.json`` (under gitignored outputs)
and a sibling ``config.json`` with CLI, environment, and a summary of fixed defaults.

Not implemented in this repository (no entry points / code paths):
  * D&C appendix-only heuristics: cross-sample target penalty, node linearized
    schedule, multi-pass tree (discussed in the paper; only baseline ``dc`` exists).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from operator_level_optimization.core.optim.operator import OperatorLevelMLP
from operator_level_optimization.models.fgln import FGLN, MaskedOperatorALS, compute_P_fgln, init_weights
from operator_level_optimization.scripts.train.deep_linear_compare import run_method as deep_linear_run


@dataclass
class SmokeResult:
    name: str
    ok: bool
    detail: str
    skipped: bool = False


def _tiny_mlp(d_in: int = 24, h: int = 8, n_cls: int = 5) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_in, h, bias=False),
        nn.ReLU(inplace=False),
        nn.Linear(h, n_cls, bias=False),
    )


def smoke_mlp(
    name: str,
    *,
    steps: int,
    batch: int,
    build_opt: Callable[[list[nn.Parameter]], OperatorLevelMLP],
) -> SmokeResult:
    device = torch.device("cpu")
    dtype = torch.float64
    torch.manual_seed(0)
    model = _tiny_mlp().to(device=device, dtype=dtype)
    params = list(model.parameters())
    opt = build_opt(params)
    opt.attach_hooks(model)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(batch, 24, device=device, dtype=dtype)
    y = torch.randint(0, 5, (batch,), device=device)
    try:
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
        last = float(loss.detach().cpu().item())
        return SmokeResult(name, True, f"last_loss={last:.6e}")
    except Exception as e:  # noqa: BLE001 — smoke harness
        return SmokeResult(name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def smoke_deep_linear_pair(method: str, *, lr: float | None = None) -> SmokeResult:
    name = f"deep_linear:{method}"
    try:
        kw = dict(
            depth=8,
            d=4,
            n=8,
            steps=3,
            lr=1.0 if lr is None else lr,
            seed=0,
            device=torch.device("cpu"),
            dtype=torch.float64,
            target_mode="mse_grad",
            dc_lam=1e-4,
            dc_alt=2,
            dc_init_mode="identity_delta",
            dc_init_scale=0.01,
            spec_every=10_000,
            spec_steps=None,
            target_kind="orth",
            init_mode="xavier",
            als_lam=1e-4,
            als_sweeps=2,
            early_stop_patience=None,
            early_stop_rel_tol=1e-4,
            early_stop_abs_tol=1e-10,
            early_stop_min_steps=200,
        )
        out = deep_linear_run(method=method, **kw)
        rel = out["history"][-1][1]
        return SmokeResult(name, True, f"final_rel_op_err={rel:.6e}")
    except Exception as e:
        return SmokeResult(name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def smoke_fgln_adaptive_lambda() -> SmokeResult:
    name = "fgln:als_exact+adaptive_lambda_step"
    try:
        torch.manual_seed(0)
        device = torch.device("cpu")
        dtype = torch.float64
        d, depth, n = 8, 12, 16
        g = torch.Generator().manual_seed(0)
        x = torch.randn(n, d, generator=g, device=device, dtype=dtype)
        u, _, vh = torch.linalg.svd(torch.randn(d, d, generator=g, device=device, dtype=dtype))
        p_star = u @ vh
        y = x @ p_star.T
        net = FGLN(d, depth=depth, p_gate=0.85, seed=0).to(device=device, dtype=dtype)
        init_weights(net, init="xavier", seed=1, target_norm=float(p_star.norm().item()), rescale_mode="all")
        opt = MaskedOperatorALS(
            net,
            P_star=p_star,
            lr=0.5,
            lam=1e-4,
            n_sweeps=2,
            als_reverse_sweep=True,
            adaptive_lambda_step=True,
            adaptive_lambda_mode="max",
            als_gateperm_warmstart=True,
            als_gateperm_warmstart_once=True,
            als_lam_anchor_post_warmstart=True,
            normalize_contexts=True,
            damping=False,
        )
        for _ in range(2):
            with torch.no_grad():
                p = compute_P_fgln(net)
                pred = x @ p.T
                grad_p = (2.0 / (n * d)) * (pred - y).T @ x
                p_tgt = p - 0.5 * grad_p
            opt.step(P_tgt_override=p_tgt)
        with torch.no_grad():
            p = compute_P_fgln(net)
            loss = torch.mean((x @ p.T - y) ** 2).item()
        return SmokeResult(name, True, f"last_mse={loss:.6e}")
    except Exception as e:
        return SmokeResult(name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_json", type=str, default="outputs/smoke/variants/smoke_variants.json")
    args = ap.parse_args()

    results: list[SmokeResult] = []

    # --- Not implemented (explicit skips) ---
    for label in (
        "dnc:cross_sample_target_penalty",
        "dnc:node_linearized_schedule",
        "dnc:multi_tree_passes",
    ):
        results.append(
            SmokeResult(
                label,
                ok=True,
                skipped=True,
                detail="No code path in this repository (paper appendix ideas only).",
            )
        )

    # --- Deep linear: ALS-exact vs divide-and-conquer ---
    results.append(smoke_deep_linear_pair("als_exact"))
    results.append(smoke_deep_linear_pair("dc", lr=1.0))

    def _dc_alt_init() -> SmokeResult:
        name = "deep_linear:dc(plain_init_mode)"
        try:
            kw = dict(
                depth=8,
                d=4,
                n=8,
                steps=2,
                lr=1.0,
                seed=0,
                device=torch.device("cpu"),
                dtype=torch.float64,
                target_mode="mse_grad",
                dc_lam=1e-4,
                dc_alt=2,
                dc_init_mode="plain",
                dc_init_scale=1.0,
                spec_every=10_000,
                spec_steps=None,
                target_kind="orth",
                init_mode="xavier",
                als_lam=1e-4,
                als_sweeps=2,
                early_stop_patience=None,
                early_stop_rel_tol=1e-4,
                early_stop_abs_tol=1e-10,
                early_stop_min_steps=200,
            )
            deep_linear_run(method="dc", **kw)
            return SmokeResult(name, True, "dc_init_mode=plain ok")
        except Exception as e:
            return SmokeResult(name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    results.append(_dc_alt_init())

    # --- MLP (OperatorLevelMLP): linearised exact / BD / Operator-KFAC / secant / CG fallback ---
    results.append(
        smoke_mlp(
            "mlp:linearized_exact(coupled)",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.05,
                lam=1e-3,
                approximation="exact",
                momentum=0.0,
            ),
        )
    )
    results.append(
        smoke_mlp(
            "mlp:block_diagonal(LOPP)",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.05,
                lam=1e-3,
                approximation="block_diagonal",
                momentum=0.0,
            ),
        )
    )
    results.append(
        smoke_mlp(
            "mlp:block_diagonal+adaptive_lam(None)",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.05,
                lam=None,
                lam_alpha=0.5,
                approximation="block_diagonal",
                momentum=0.0,
            ),
        )
    )
    results.append(
        smoke_mlp(
            "mlp:operator_kfac",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.05,
                lam=1e-3,
                approximation="operator_kfac",
                momentum=0.0,
            ),
        )
    )
    results.append(
        smoke_mlp(
            "mlp:secant(rank1+coupled_kernel)",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.05,
                lam=1e-3,
                approximation="secant",
                momentum=0.0,
            ),
        )
    )
    results.append(
        smoke_mlp(
            "mlp:secant_r(rank-3)",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.05,
                lam=1e-3,
                approximation="secant_r",
                momentum=0.0,
                rank=3,
            ),
        )
    )
    results.append(
        smoke_mlp(
            "mlp:cg(fallback→block_diagonal)",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.05,
                lam=1e-3,
                approximation="cg",
                momentum=0.0,
            ),
        )
    )

    # --- MLP: full frozen-gate ALS (nonlinear EOPP) ---
    results.append(
        smoke_mlp(
            "mlp:als+layer_solve=mn(baseline)",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.3,
                lam=1e-3,
                approximation="als",
                als_layer_solve="mn",
                n_sweeps=2,
                als_reverse_sweep=True,
                momentum=0.0,
                als_gateperm_warmstart=True,
                als_gateperm_warmstart_once=True,
                als_lam_anchor_post_warmstart=False,
            ),
        )
    )
    results.append(
        smoke_mlp(
            "mlp:als+layer_solve=exact(small_layers)",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.3,
                lam=1e-3,
                approximation="als",
                als_layer_solve="exact",
                n_sweeps=2,
                als_reverse_sweep=True,
                momentum=0.0,
                als_gateperm_warmstart=True,
                als_gateperm_warmstart_once=True,
                als_lam_anchor_post_warmstart=False,
            ),
        )
    )
    results.append(
        smoke_mlp(
            "mlp:als+layer_solve=per_sample(mean_field)",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.3,
                lam=1e-3,
                approximation="als",
                als_layer_solve="per_sample",
                n_sweeps=2,
                als_reverse_sweep=True,
                momentum=0.0,
                als_gateperm_warmstart=True,
                als_gateperm_warmstart_once=True,
                als_lam_anchor_post_warmstart=False,
            ),
        )
    )
    results.append(
        smoke_mlp(
            "mlp:als+adaptive_lam(None)_per_layer",
            steps=2,
            batch=8,
            build_opt=lambda p: OperatorLevelMLP(
                p,
                lr=0.3,
                lam=None,
                lam_alpha=0.5,
                approximation="als",
                als_layer_solve="mn",
                n_sweeps=2,
                als_reverse_sweep=True,
                momentum=0.0,
                als_gateperm_warmstart=True,
                als_gateperm_warmstart_once=True,
                als_lam_anchor_post_warmstart=False,
            ),
        )
    )

    # --- FGLN ---
    results.append(smoke_fgln_adaptive_lambda())

    # --- Report ---
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repro = {
        "script_module": "operator_level_optimization.scripts.smoke_variants",
        "cli_args": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "outputs": {
            "results_json": str(out_path.resolve()),
            "config_json": str((out_path.parent / "config.json").resolve()),
        },
        "implementation_note": (
            "Each smoke case builds ``OperatorLevelMLP`` / ``deep_linear_run`` / ``MaskedOperatorALS`` "
            "with literals in this file. Search for ``OperatorLevelMLP(``, ``deep_linear_run(``, and "
            "``MaskedOperatorALS(`` for exact kwargs, shapes, and seeds."
        ),
        "fixed_defaults": {
            "mlp_architecture": "24 -> 8 -> 5 (bias-free), ReLU, CrossEntropyLoss, cpu float64, torch.manual_seed(0)",
            "mlp_smoke_steps": 2,
            "mlp_smoke_batch": 8,
            "deep_linear": "depth=8, d=4, n=8, cpu float64, seed=0, target_kind=orth, mse_grad (see smoke_deep_linear_pair)",
            "fgln_smoke": "d=8, depth=12, n=16, p_gate=0.85, outer lr=0.5, lam=1e-4, n_sweeps=2, adaptive_lambda_step=True",
        },
    }
    (out_path.parent / "config.json").write_text(json.dumps(repro, indent=2, sort_keys=True))

    payload: dict[str, Any] = {"results": [asdict(r) for r in results]}
    out_path.write_text(json.dumps(payload, indent=2))

    w_name = max(len(r.name) for r in results)
    for r in results:
        tag = "SKIP" if r.skipped else ("OK  " if r.ok else "FAIL")
        first_line = r.detail.strip().split("\n")[0]
        print(f"{tag}  {r.name:{w_name}s}  {first_line}")

    failed = [r for r in results if not r.skipped and not r.ok]
    print(f"\nWrote {out_path} and {out_path.parent / 'config.json'}")
    if failed:
        print(f"\n{len(failed)} failure(s):")
        for r in failed:
            print(f"--- {r.name} ---\n{r.detail}\n")
        raise SystemExit(1)
    print("All implemented variants completed without exception.")


if __name__ == "__main__":
    main()
