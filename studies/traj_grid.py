"""Operator-space trajectories for every optimiser, initialisation and architecture.

This is the previous submission's two-dimensional figure, widened into a complete view. The model
is a bias-free chain of depth L and width h mapping two features to one output, so however large
the weight space is, the operator P(x) has exactly two components and its path can be drawn
exactly. The loss is quadratic in P,

    L(P) = (1/N) || X P^T - y ||^2,

so the contours are exact ellipses and P* is the ordinary least-squares solution.

Why operator space and not weight space. A model whose *weight* space is two-dimensional has a
rank-one step map, so M^+ is a multiple of M^T and gradient descent coincides with the reference
exactly (measured at cos = 1.0000000000). Nothing about the parameterisation can show up. With L
layers into a two-dimensional operator, M has rank two and the rules separate. Every difference
visible in these panels is therefore attributable to *how each rule converts a weight step into an
operator step*, since all of them descend the same quadratic in P.

Input-dependent architectures. For `linear` and `fgln` the gates do not depend on x, so P is a
single 1x2 matrix and the trajectory is exact. For `relu`, `leaky` and `crelu` the operator varies
across inputs, and what is plotted is the data-averaged operator E_x[P(x)]. Each such panel is
annotated with the spread of P(x) across inputs at the end of training, because the average alone
cannot distinguish "every input sees the same map" from "the average of two very different maps"
-- and the second is what a nonlinear network is for. The contours are then a fixed frame of
reference rather than the model's true loss surface, since a nonlinear model is not confined to a
single linear map, so on those panels do not read "reaches P*" as "fits the data".

Rules come from `exp.step_for`, so these are the same implementations the main experiments use,
including the comparators verified against their published forms in `tests_baselines.py`.

Run: `python traj_grid.py --depth 8 --hidden 32`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import exp
import gpu

# Each arm with the learning rates tried for it. The grid is swept per panel and the rate that
# reaches the lowest operator loss is the one drawn: a trajectory shown at a badly scaled step
# size says nothing about the rule, only about the tuning, and the arms differ in natural scale by
# orders of magnitude (the reference has no learning rate at all, it has a target size eta).
ARMS = {
    "gd":        [1e-4, 1e-3, 0.003, 0.01, 0.03, 0.1, 0.3],
    "adam":      [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1],
    "op":        [0.1, 0.3, 1.0],
    "muon":      [3e-4, 0.001, 0.003, 0.01, 0.03, 0.1],
    "shampoo":   [3e-4, 0.001, 0.003, 0.01, 0.03, 0.1],
    "soap":      [1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03],
    "kfac":      [0.0003, 0.001, 0.003, 0.01, 0.03],
    "heavyball": [0.0003, 0.001, 0.003, 0.01, 0.03],
}
LABEL = {"gd": "gradient descent", "adam": "Adam", "op": "reference (operator)", "muon": "Muon",
         "shampoo": "Shampoo", "soap": "SOAP", "kfac": "K-FAC", "heavyball": "heavy ball"}
COLOR = {"gd": "#4c78a8", "adam": "#59a14f", "op": "#e15759", "muon": "#b07aa1",
         "shampoo": "#f28e2b", "soap": "#76b7b2", "kfac": "#9c755f", "heavyball": "#bab0ac"}
ARCHS = ["linear", "fgln", "relu", "leaky", "crelu"]
INITS = ["he", "xavier", "orthogonal", "looks_linear"]


def dataset(n, seed, dev, scales=(1.0, 0.35)):
    """Two anisotropic features and a linear target: the anisotropy is what bends the paths.

    `n` is kept small because for an input-dependent architecture the step map's codomain is the
    stack of per-sample operators, so its size drives the LSQR cost of the reference arm -- the
    one arm here that is not nearly free. A rank-2 intuition does not apply: only `linear` and
    `fgln` have a genuinely two-dimensional target.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 2, generator=g, dtype=torch.float64) * torch.tensor(scales, dtype=torch.float64)
    w = torch.randn(2, generator=g, dtype=torch.float64)
    return X.to(dev), (X @ w).to(dev)


def operators(Ws, X, arch):
    """P(x) for every input, shape (n, 1, 2)."""
    _, gs = gpu.forward(Ws, X, arch)
    A, B = gpu.contexts(Ws, gs, X, arch)
    return A[0] @ Ws[0] @ B[0]


def op_loss(P, X, y):
    """(1/N)||X P^T - y||^2 evaluated at a single 1x2 operator.

    Coerces to numpy: the callers mix a numpy path with torch data, and `torch @ numpy` happens
    to work through `__array_wrap__` while warning that it will stop.
    """
    X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    y = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
    r = X @ np.asarray(P).reshape(2) - y
    return float((r ** 2).mean())


def run(arch, init, arm, hyper, depth, hidden, seed, X, y, steps, dev):
    """One trajectory: the mean operator at every step, plus the pointwise spread."""
    Ws = gpu.init_net(2, hidden, 1, depth, arch, seed, dev, dtype=torch.float64, init=init)
    state = {"m": [torch.zeros_like(w) for w in Ws], "v": [torch.zeros_like(w) for w in Ws]}
    path, spread = [], []
    n = X.shape[0]
    for t in range(steps):
        P = operators(Ws, X, arch)
        mean = P.mean(0).reshape(2)
        path.append(mean.detach().cpu().numpy().copy())
        spread.append(float(P.reshape(-1, 2).std(0).norm()))
        pred = (X.unsqueeze(1) @ P.transpose(1, 2)).reshape(-1)
        R = (2.0 * (pred - y)).reshape(n, 1)
        h = hyper if arm != "op" else (hyper, 50)
        # Deliberately not wrapped in try/except: a rule that raises is a bug to see, not a short
        # trajectory to plot. Divergence is caught below by the finiteness and magnitude checks.
        d = exp.step_for(arm, Ws, X, R, h, arch, state, t)
        Ws = [w + dw for w, dw in zip(Ws, d)]
        if not all(torch.isfinite(w).all() for w in Ws):
            break
        if abs(float(mean.abs().max())) > 1e4:
            break
    return np.array(path), np.array(spread)


def best_for(arch, init, arm, depth, hidden, seed, X, y, steps, dev):
    """Sweep the arm's learning rates, keep the trajectory that ends at the lowest loss."""
    best = None
    for lr in ARMS[arm]:
        # Divergence is an expected outcome of sweeping learning rates, and at depth 32 most of
        # each grid diverges. It is caught HERE, per learning rate, and never inside `run`: a rule
        # that raises mid-trajectory for any other reason is a bug that must surface.
        try:
            p, s = run(arch, init, arm, lr, depth, hidden, seed, X, y, steps, dev)
        except (ValueError, RuntimeError, torch._C._LinAlgError) as exc:
            print(f"    {arch}/{init}/{arm} lr={lr:g} diverged ({type(exc).__name__})", flush=True)
            continue
        if len(p) < 2:
            continue
        end = op_loss(p[-1], X.cpu(), y.cpu())
        if not np.isfinite(end):
            continue
        if best is None or end < best[0]:
            best = (end, lr, p, s)
    return best


def build(depth=8, hidden=32, n=128, seed=0, steps=300, dev=None, archs=None, inits=None,
          cache=None):
    """Sweep every (architecture, initialisation, optimiser) cell, resuming from `cache`.

    Each completed cell is written out immediately. This machine's OOM reaper has killed this job
    three times at unpredictable points, and a sweep that starts from zero after every kill never
    finishes; with a cache each restart only pays for what is left.
    """
    import pickle
    dev = dev or "cpu"
    X, y = dataset(n, seed, dev)
    Xc, yc = X.cpu(), y.cpu()
    Pstar = torch.linalg.lstsq(Xc, yc.unsqueeze(1)).solution.reshape(2).numpy()
    cache = Path(cache) if cache else None
    out = {}
    scales = {}
    for arch in (archs or ARCHS):
        for init in (inits or INITS):
            Ws = gpu.init_net(2, hidden, 1, depth, arch, seed, dev, dtype=torch.float64, init=init)
            scales[(arch, init)] = float(operators(Ws, X, arch).abs().max())
    if cache is not None and cache.exists():
        out = pickle.loads(cache.read_bytes())
        print(f"resuming with {len(out)} cells already done", flush=True)
    for arch in (archs or ARCHS):
        for init in (inits or INITS):
            for arm in ARMS:
                if (arch, init, arm) in out:
                    continue
                b = best_for(arch, init, arm, depth, hidden, seed, X, y, steps, dev)
                if b is not None:
                    end, lr, p, sp = b
                    out[(arch, init, arm)] = dict(lr=lr, end=end, path=p, spread=sp)
                if cache is not None:
                    cache.write_bytes(pickle.dumps(out))
            print(f"  {arch}/{init} done ({len(out)} cells, "
                  f"|P| at init = {scales[(arch, init)]:.1e})", flush=True)
    out["_scales"] = scales
    if cache is not None:
        cache.write_bytes(pickle.dumps(out))
    return out, Pstar, Xc.numpy(), yc.numpy()


def figure(data, Pstar, X, y, depth, hidden, archs=None, inits=None):
    """One panel per (architecture, initialisation); one coloured path per optimiser."""
    import matplotlib.pyplot as plt
    archs = archs or ARCHS
    inits = inits or INITS
    fig, axes = plt.subplots(len(archs), len(inits),
                             figsize=(3.5 * len(inits), 3.2 * len(archs)), squeeze=False)
    scales = data.get("_scales", {})
    paths = [d["path"] for k, d in data.items() if k != "_scales"]
    pts = np.concatenate(paths) if paths else np.zeros((1, 2))
    lo = np.minimum(pts.min(0), Pstar) - 0.4
    hi = np.maximum(pts.max(0), Pstar) + 0.4
    lo = np.maximum(lo, Pstar - 4.0)
    hi = np.minimum(hi, Pstar + 4.0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160))
    Z = np.stack([((X @ np.array([a, b]) - y) ** 2).mean()
                  for a, b in zip(gx.ravel(), gy.ravel())]).reshape(gx.shape)
    for r, arch in enumerate(archs):
        for c, init in enumerate(inits):
            ax = axes[r][c]
            ax.contour(gx, gy, Z, levels=np.logspace(np.log10(max(Z.min(), 1e-6) + 1e-9),
                                                     np.log10(Z.max()), 12),
                       colors="#cccccc", linewidths=0.6, zorder=0)
            ax.plot(*Pstar, "k*", ms=11, zorder=5, label="$P^\\star$ (OLS)" if r == c == 0 else None)
            for arm in ARMS:
                d = data.get((arch, init, arm))
                if d is None:
                    continue
                p = d["path"]
                ax.plot(p[:, 0], p[:, 1], "-", color=COLOR[arm], lw=1.4, alpha=0.9,
                        label=f"{LABEL[arm]}" if r == 0 and c == 0 else None, zorder=3)
                ax.plot(p[0, 0], p[0, 1], "o", color=COLOR[arm], ms=3.5, zorder=4)
            drawn = [m for m in ARMS if (arch, init, m) in data]
            sc = scales.get((arch, init))
            note = f"$\\|P\\|_\\infty$ at init: {sc:.1e}" if sc is not None else ""
            if not drawn:
                # An empty panel is a result, not a gap: at this depth the initialisation itself
                # puts the operator so far from O(1) that no learning rate in the grid recovers.
                ax.text(0.5, 0.55, "no optimiser converged", transform=ax.transAxes,
                        ha="center", fontsize=8, color="#b03030")
                ax.text(0.5, 0.45, note, transform=ax.transAxes, ha="center",
                        fontsize=7, color="#b03030")
            else:
                if arch not in ("linear", "fgln"):
                    sp = [data[(arch, init, m)]["spread"][-1] for m in drawn]
                    note += f"\nspread of $P(x)$: {min(sp):.1e}-{max(sp):.1e}"
                ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=6.5, color="#555555")
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_title(f"{arch} / {init}", fontsize=9)
            if r == len(archs) - 1:
                ax.set_xlabel("$P_1$")
            if c == 0:
                ax.set_ylabel("$P_2$")
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(f"Operator-space trajectories, depth {depth}, width {hidden}. "
                 f"Solid dot = initialisation, star = OLS solution. "
                 f"For relu / leaky / crelu the path is the data-averaged operator.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="../figures/fig_traj_grid")
    a = ap.parse_args()
    data, Pstar, X, y = build(a.depth, a.hidden, a.n, a.seed, a.steps, a.device,
                              cache=f"{a.out}_L{a.depth}.cache.pkl")
    fig = figure(data, Pstar, X, y, a.depth, a.hidden)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}_L{a.depth}.{ext}", dpi=150, bbox_inches="tight")
    rows = {f"{k[0]}|{k[1]}|{k[2]}": dict(lr=v["lr"], end=v["end"], steps=len(v["path"]))
            for k, v in data.items() if k != "_scales"}
    rows["_init_scales"] = {f"{a}|{i}": v for (a, i), v in data.get("_scales", {}).items()}
    Path(f"{a.out}_L{a.depth}.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}_L{a.depth}.png  ({len(data)} trajectories)")


if __name__ == "__main__":
    main()
