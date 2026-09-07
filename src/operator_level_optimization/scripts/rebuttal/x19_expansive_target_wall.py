"""E19 -- the genuine, eta-independent wall: expansive (non-contractive) targets.

Derivation (verified numerically in-conversation): the local operator gradient
at identity init is G = c*(I-P*)*Sigma_x. For isotropic Sigma_x=I, G's
eigenvalues always have non-negative real part for ANY orthogonal (isometric)
P*, because 2I-(P*+P*^T) is PSD whenever P*'s singular values equal 1 -- this
is what guarantees a comfortable safe-eta window at ANY depth, regardless of
how "far" the target rotation is (verified: even P*=-I, the maximally rotated
isometry, keeps this guarantee). Anisotropic Sigma_x shrinks the safe window
but does not break the sign (verified numerically up to cond=1e4).

What DOES break the guarantee: target singular values EXCEEDING 1 (expansive,
not just non-isometric). Then min Re(eig G) goes negative and the
first-unstable eta collapses to ~1e-7 (vs ~8 for contractive targets) --
technically eta-independent instability. At realistic eta (1e-2 to 1e-1) and
depth (hundreds to thousands of layers) this compounds into dramatic,
verified blowup in the toy matrix-power model (e.g. scale=2, eta=3e-2,
L=4096 -> factor 4.5e6). This experiment tests whether that toy prediction
survives REAL iterative training (gradient recomputed every step, not
step-1-repeated), using each method's own L=128-tuned lr held FIXED across
depth and target scale -- so a previously-working setup is tested against a
purely target-side change (not a fresh lr search).

ALS-exact is the reference: its modewise solve should correctly distribute
the expansion as scale^(1/L) per layer rather than have every layer try to
realize the full scale factor at once (the mechanism argued to cause the
blowup for gradient methods), so it should remain robust regardless of scale.

Run from repo root:
  PYTHONPATH=labs venv/bin/python -m operator_level_optimization.scripts.rebuttal.x19_expansive_target_wall \
      --out_dir outputs/oplevel_rebuttal/e19_expansive_wall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from operator_level_optimization.scripts.train.deep_linear_compare import run_method
from operator_level_optimization.scripts.utils.io import get_device, jsonify

# Each method's own L=128-tuned lr from E13's probe (held fixed across
# depth AND target scale -- this is the point: a previously-working lr).
FIXED_LR_L128 = {
    "adam": 3e-4,
    "heavyball": 1e-3,
    "kfac": 1e-4,
    "shampoo": 3e-4,
}


def run_cfg(method, lr, d, depth, steps, seed, device, spec_every, target_scale):
    try:
        res = run_method(
            method=method, depth=depth, d=d, n=4 * d, steps=steps, lr=lr,
            seed=seed, device=device, dtype=torch.float64,
            target_mode="mse_grad", dc_lam=0.0, dc_alt=1,
            dc_init_mode="plain", dc_init_scale=1.0,
            spec_every=spec_every, spec_steps=None,
            target_kind="orth", init_mode="identity",
            als_lam=1e-4, als_sweeps=4, target_scale=target_scale,
        )
    except Exception as exc:
        return {"failed": str(exc), "final_rel_err": float("inf")}
    res["final_rel_err"] = float(res["history"][-1][1])
    hist = res.get("history", [])
    res["peak_rel_err"] = max((h[1] for h in hist), default=float("nan"))
    res["diverged"] = (not torch.isfinite(torch.tensor(res["final_rel_err"]))) or res["final_rel_err"] > 1e6
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--depths", type=int, nargs="+", default=[128, 512, 1024, 2048])
    ap.add_argument("--target_scales", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--als_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec_every", type=int, default=100)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    results: dict = {}
    for scale in args.target_scales:
        for depth in args.depths:
            for method, lr in FIXED_LR_L128.items():
                res = run_cfg(method, lr, args.d, depth, args.steps, args.seed,
                              device, args.spec_every, scale)
                res["lr"] = lr
                print(f"[e19] scale={scale:g} L={depth:5d} {method:10s} "
                      f"lr={lr:.1e} (fixed): final={res['final_rel_err']:.3e} "
                      f"peak={res['peak_rel_err']:.3e} diverged={res['diverged']}",
                      flush=True)
                results[f"{method}_s{scale:g}_L{depth}"] = res

            als_res = run_cfg("als_exact", 1.0, args.d, depth, args.als_steps,
                              args.seed, device, args.spec_every, scale)
            als_res["lr"] = 1.0
            print(f"[e19] scale={scale:g} L={depth:5d} als_exact: "
                  f"final={als_res['final_rel_err']:.3e}", flush=True)
            results[f"als_exact_s{scale:g}_L{depth}"] = als_res

    summary = {
        k: {"lr": r.get("lr"), "final_rel_err": r.get("final_rel_err"),
            "peak_rel_err": r.get("peak_rel_err"), "diverged": r.get("diverged")}
        for k, r in results.items()
    }
    (out_dir / "results.json").write_text(
        json.dumps(jsonify({"results": results, "summary": summary}), indent=2))
    (out_dir / "config.json").write_text(json.dumps(jsonify(vars(args)), indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
