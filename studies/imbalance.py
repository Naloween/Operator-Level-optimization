"""Drop balancedness. Which hypothesis was actually doing the work?

`theory/05-imbalance.md` factors the mode gain as `c_k = s_k^{2-2/L} K_k` with
`K_k = sum_l (a_{l,k} s_k^{-1/L})^{-2} >= L`, so the cross-mode bias exponent is exactly

    d log c / d log s  =  (2 - 2/L)  +  d log K / d log s,

and the correction vanishes precisely when the imbalance does not depend on the mode. That
turns "balancedness" -- which no real network satisfies -- into the far weaker
"mode-independence", and makes the difference measurable rather than assumed.

Three parts:

* **static** -- the decomposition itself, over depths and imbalance patterns, on diagonal
  deep linear networks where alignment holds exactly and only balance varies. Mode-independent
  patterns (including a 20x bottleneck layer) must leave the exponent at `2 - 2/L` exactly;
  mode-dependent ones must shift it by exactly `d log K/d log s`.
* **init** -- what real initializations do, across depths, widths and seeds, for deep linear
  and CReLU. Xavier is mode-dependent, so the question is which way and by how much.
* **trained** -- whether the correction moves during training, since `K` is not conserved.

Writes `runs/theory/imbalance.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from olo.models.crelu_mlp import CReLUMLP
from olo.models.deep_linear import DeepLinear
from olo.tasks.base import mse
from olo.theory.imbalance import balanced_exponent, imbalance_norm, mode_gains

MODELS = {"deep_linear": DeepLinear, "crelu_mlp": CReLUMLP}


def diagonal_net(w: torch.Tensor, spread: float, width: int) -> DeepLinear:
    """Deep linear, diagonal. `w` is (L, width) log layer-scale offsets, per layer per mode."""
    L = w.shape[0]
    a = torch.linspace(-0.5, 0.5, width, dtype=torch.float64) * spread
    net = DeepLinear(d_in=width, d_out=width, width=width, depth=L).double()
    with torch.no_grad():
        for l, W in enumerate(net.weights):
            W.copy_(torch.diag(torch.exp(a / L + w[l])))
    return net


def patterns(L: int, width: int) -> dict[str, torch.Tensor]:
    """Imbalance patterns, split by whether they depend on the mode."""
    z = torch.zeros(L, width, dtype=torch.float64)
    lin = torch.linspace(-0.5, 0.5, L, dtype=torch.float64)
    bott = torch.zeros(L, dtype=torch.float64)
    bott[0] = -3.0
    rnd = torch.randn(L, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    per_mode = torch.linspace(0.0, 2.0, width, dtype=torch.float64)
    return {
        "balanced": z,
        "mode-indep: linear in l": (lin * 4.0).unsqueeze(1).expand(L, width).contiguous(),
        "mode-indep: bottleneck": bott.unsqueeze(1).expand(L, width).contiguous(),
        "mode-indep: random": rnd.unsqueeze(1).expand(L, width).contiguous(),
        "mode-DEP: linear in k": lin.unsqueeze(1) * per_mode.unsqueeze(0),
        "mode-DEP: bottleneck on big modes":
            torch.cat([-3.0 * torch.linspace(0, 1, width, dtype=torch.float64).unsqueeze(0),
                       torch.zeros(L - 1, width, dtype=torch.float64)]),
    }


def static(depths, width, spread) -> list[dict]:
    out = []
    for L in depths:
        for name, w in patterns(L, width).items():
            net = diagonal_net(w, spread, width)
            mg = mode_gains(net, torch.eye(width, dtype=torch.float64))
            slope, dK = mg.exponent()
            out.append({
                "part": "static", "depth": L, "pattern": name,
                "balanced": balanced_exponent(L), "slope": slope, "dlogK": dK,
                "residual": abs(slope - balanced_exponent(L) - dK),
                "K_median": float(np.median(mg.imbalance_factor)),
                "imbalance": imbalance_norm(net),
                "mode_dependent": name.startswith("mode-DEP"),
            })
    return out


def at_init(pairs, depths, widths, seeds) -> list[dict]:
    out = []
    for model, init in pairs:
        if True:
            for L in depths:
                for width in widths:
                    for seed in seeds:
                        net = MODELS[model](d_in=width, d_out=width, width=width,
                                            depth=L).double()
                        net.initialize(init, seed=seed)
                        g = torch.Generator().manual_seed(seed + 1)
                        X = torch.randn(8, width, generator=g, dtype=torch.float64)
                        mg = mode_gains(net, X)
                        slope, dK = mg.exponent()
                        out.append({
                            "part": "init", "model": model, "init": init, "depth": L,
                            "width": width, "seed": seed,
                            "balanced": balanced_exponent(L), "slope": slope, "dlogK": dK,
                            "K_median": float(np.median(mg.imbalance_factor)),
                            "imbalance": imbalance_norm(net),
                            "live_modes": int((mg.s > 1e-10).sum()),
                        })
    return out


def trained(model, init, depths, width, steps, lr_c, batch, eval_every, seed) -> list[dict]:
    """Does the correction move as the network trains? `K` is not conserved, so it may."""
    out = []
    for L in depths:
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        P_star = torch.linalg.qr(
            torch.randn(width, width, generator=g, dtype=torch.float64))[0] * 2.0
        net = MODELS[model](d_in=width, d_out=width, width=width, depth=L).double()
        net.initialize(init, seed=seed)
        opt = torch.optim.SGD(net.parameters(), lr=lr_c / L)
        gg = torch.Generator().manual_seed(seed + 7)
        for step in range(steps + 1):
            X = torch.randn(batch, width, generator=gg, dtype=torch.float64)
            if step % eval_every == 0:
                mg = mode_gains(net, X[:8])
                slope, dK = mg.exponent()
                out.append({
                    "part": "trained", "model": model, "init": init, "depth": L,
                    "step": step, "balanced": balanced_exponent(L),
                    "slope": slope, "dlogK": dK,
                    "K_median": float(np.median(mg.imbalance_factor)),
                    "sbar": float(np.exp(np.log(np.clip(mg.s, 1e-300, None)).mean())),
                    "sep": float(np.log(mg.s.max() / max(mg.s.min(), 1e-300))),
                    "imbalance": imbalance_norm(net),
                })
            if step == steps:
                break
            opt.zero_grad(set_to_none=True)
            mse(net(X), X @ P_star.T).backward()
            opt.step()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--models", nargs="+", default=["deep_linear", "crelu_mlp"])
    ap.add_argument("--inits", nargs="+", default=["xavier", "haar", "looks_linear"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--spread", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--lr-c", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default="runs/theory/imbalance.json")
    a = ap.parse_args()

    t0 = time.time()
    rows = static(a.depths, max(a.widths), a.spread)
    worst = max(r["residual"] for r in rows)
    print(f"static: {len(rows)} cases, worst |slope - (2-2/L) - dlogK| = {worst:.2e}")
    for r in rows:
        if r["depth"] == a.depths[1 % len(a.depths)]:
            print(f"    {r['pattern']:<36} slope {r['slope']:>7.4f}  "
                  f"balanced {r['balanced']:>6.4f}  dlogK {r['dlogK']:>8.4f}  "
                  f"K/L {r['K_median'] / r['depth']:>6.2f}")

    # looks_linear has no generic form: only the rectifier models build it.
    pairs = [(m, i) for m in a.models for i in a.inits
             if not (m == "deep_linear" and i == "looks_linear")]
    rows += at_init(pairs, a.depths, a.widths, a.seeds)
    print(f"init: done ({time.time()-t0:.0f}s)")

    for model in a.models:
        rows += trained(model, "xavier", a.depths, a.widths[-1], a.steps, a.lr_c,
                        a.batch, a.eval_every, a.seeds[0])
    print(f"trained: done ({time.time()-t0:.0f}s)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}))
    print(f"wrote {a.out}  ({len(rows)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
