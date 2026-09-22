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
    "gd":        [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1],
    "adam":      [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03],
    "op":        [0.03, 0.1, 0.3, 1.0],
    "muon":      [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03],
    "shampoo":   [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03],
    "soap":      [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01],
    "kfac":      [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01],
    "heavyball": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01],
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
    """One trajectory: the mean operator at each step, the pointwise spread, and the MODEL loss.

    The model loss is `mean((f(x) - y)^2)` from an actual forward pass, and it is the only
    trustworthy scoring signal here. An earlier version scored the loss of the *mean* operator
    E_x[P(x)], which for a nonlinear architecture is not the model: runs whose per-sample
    operators exploded to 1e74 had means that cancelled back to near P*, so they scored well and
    were plotted. Divergence is likewise detected on the per-sample operators, not on their mean.
    """
    Ws = gpu.init_net(2, hidden, 1, depth, arch, seed, dev, dtype=torch.float64, init=init)
    state = {"m": [torch.zeros_like(w) for w in Ws], "v": [torch.zeros_like(w) for w in Ws]}
    path, spread, losses = [], [], []
    n = X.shape[0]
    for t in range(steps):
        P = operators(Ws, X, arch)
        pred = (X.unsqueeze(1) @ P.transpose(1, 2)).reshape(-1)
        loss = float(((pred - y) ** 2).mean())
        if not np.isfinite(loss) or loss > 1e8:
            break
        if not torch.isfinite(P).all() or float(P.abs().max()) > 1e6:
            break
        path.append(P.mean(0).reshape(2).detach().cpu().numpy().copy())
        spread.append(float(P.reshape(-1, 2).std(0).norm()))
        losses.append(loss)
        R = (2.0 * (pred - y)).reshape(n, 1)
        if arm == "op":
            # The reference here is the EXACT minimal realisation: the damped minimum-norm solve,
            # not the k-truncated Krylov step used as a practical solver elsewhere. On a deep
            # linear chain the step map is ~83% null, and an undamped truncated solve drifts into
            # that null space rather than converging -- so the truncation is a property of the
            # solver, not of the rule, and a figure about the rule should not inherit it.
            _, gs = gpu.forward(Ws, X, arch)
            A, B = gpu.contexts(Ws, gs, X, arch)
            M = gpu.StepMap(A, B, [tuple(w.shape) for w in Ws])
            G = R.unsqueeze(2) * X.unsqueeze(1)
            x = gpu.solve_min_norm(M, (-hyper * G).reshape(-1), iters=400, lam_rel=1e-7, stall=1e-10)
            d = [x[M.offs[l]:M.offs[l + 1]].view(Ws[l].shape) for l in range(len(Ws))]
        else:
            # Deliberately not wrapped in try/except: a rule that raises is a bug to see, not a
            # short trajectory to plot. Divergence is caught per learning rate in `best_for`.
            d = exp.step_for(arm, Ws, X, R, hyper, arch, state, t)
        Ws = [w + dw for w, dw in zip(Ws, d)]
        if not all(torch.isfinite(w).all() for w in Ws):
            break
    return np.array(path), np.array(spread), np.array(losses)


def best_for(arch, init, arm, depth, hidden, seed, X, y, steps, dev, wander=5.0, slack=2.0):
    """Sweep the arm's learning rates; keep the run that descends most SMOOTHLY among those that
    converge.

    Selecting purely by lowest loss picks, for several of these rules, a step size that overshoots
    and oscillates across the optimum: it lands low, and its path is a zigzag that says nothing
    about the rule while inflating the panel's extent enough to flatten the loss contours into
    stripes. So we keep every run whose tail loss is within `slack` of the best for that arm, then
    among those choose the shortest path --- the same destination, reached most directly.
    """
    Pstar = np.linalg.lstsq(X.cpu().numpy(), y.cpu().numpy(), rcond=None)[0]
    cands = []
    for lr in ARMS[arm]:
        try:
            p, sp, ls = run(arch, init, arm, lr, depth, hidden, seed, X, y, steps, dev)
        except (ValueError, RuntimeError, torch._C._LinAlgError) as exc:
            print(f"    {arch}/{init}/{arm} lr={lr:g} diverged ({type(exc).__name__})", flush=True)
            continue
        if len(p) < 2 or len(ls) < 2 or not np.isfinite(ls[-1]):
            continue
        scale = max(np.linalg.norm(p[0] - Pstar), 1e-6)
        if np.abs(p - Pstar).max() > wander * scale:
            continue
        tail = float(np.mean(ls[max(1, int(len(ls) * 0.9)):]))
        if not np.isfinite(tail) or tail >= ls[0]:
            continue
        cands.append((tail, float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), lr, p, sp,
                      float(ls[0])))
    if not cands:
        return None
    floor = min(c[0] for c in cands)
    good = [c for c in cands if c[0] <= floor * slack or c[0] <= floor + 1e-12]
    tail, _, lr, p, sp, start = min(good, key=lambda c: c[1])
    return (tail, lr, p, sp, start)


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
                    end, lr, p, sp, start = b
                    out[(arch, init, arm)] = dict(lr=lr, end=end, path=p, spread=sp, start=start)
                if cache is not None:
                    cache.write_bytes(pickle.dumps(out))
            print(f"  {arch}/{init} done ({len(out)} cells, "
                  f"|P| at init = {scales[(arch, init)]:.1e})", flush=True)
    out["_scales"] = scales
    if cache is not None:
        cache.write_bytes(pickle.dumps(out))
    return out, Pstar, Xc.numpy(), yc.numpy()


def figure(data, Pstar, X, y, depth, hidden, archs=None, inits=None):
    """One panel per (architecture, initialisation); one coloured path per optimiser.

    Axes are scaled PER PANEL. A shared frame is unreadable here: at depth 32 the initial operator
    ranges over six orders of magnitude across panels (1.1e+04 for linear/he down to 1.1e-07 for
    fgln/orthogonal), so any global window either clips the large panels or compresses every small
    one into the marker at the origin. The contours are redrawn per panel for the same reason.
    """
    import matplotlib.pyplot as plt
    archs = archs or ARCHS
    inits = inits or INITS
    scales = data.get("_scales", {})
    fig, axes = plt.subplots(len(archs), len(inits),
                             figsize=(3.6 * len(inits), 3.3 * len(archs)), squeeze=False)
    for r, arch in enumerate(archs):
        for c, init in enumerate(inits):
            ax = axes[r][c]
            drawn = [m for m in ARMS if (arch, init, m) in data]
            sc = scales.get((arch, init))
            note = f"$\\|P\\|_\\infty$ at init: {sc:.1e}" if sc is not None else ""
            if not drawn:
                ax.text(0.5, 0.55, "no optimiser converged", transform=ax.transAxes,
                        ha="center", fontsize=8, color="#b03030")
                ax.text(0.5, 0.45, note, transform=ax.transAxes, ha="center",
                        fontsize=7, color="#b03030")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{arch} / {init}", fontsize=9)
                continue
            pts = np.concatenate([data[(arch, init, m)]["path"] for m in drawn])
            pts = np.vstack([pts, Pstar])
            lo, hi = pts.min(0), pts.max(0)
            pad = np.maximum((hi - lo) * 0.15, np.maximum(np.abs(hi), 1e-3) * 0.05)
            lo, hi = lo - pad, hi + pad
            gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 120), np.linspace(lo[1], hi[1], 120))
            Z = np.stack([((X @ np.array([a, b]) - y) ** 2).mean()
                          for a, b in zip(gx.ravel(), gy.ravel())]).reshape(gx.shape)
            ax.contour(gx, gy, Z, levels=12, colors="#d0d0d0", linewidths=0.6, zorder=0)
            for arm in drawn:
                d = data[(arch, init, arm)]
                p = d["path"]
                ax.plot(p[:, 0], p[:, 1], "-", color=COLOR[arm], lw=1.3, alpha=0.85,
                        label=LABEL[arm] if r == 0 and c == 1 else None, zorder=3)
                ax.plot(p[0, 0], p[0, 1], "o", color=COLOR[arm], ms=3.5, zorder=4)
            ax.plot(*Pstar, "k*", ms=12, zorder=5)
            if arch not in ("linear", "fgln"):
                sp = [data[(arch, init, m)]["spread"][-1] for m in drawn]
                note += f"\nspread of $P(x)$: {min(sp):.1e}-{max(sp):.1e}"
            ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=6.5, color="#555555")
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_title(f"{arch} / {init}  ({len(drawn)}/{len(ARMS)} arms)", fontsize=9)
            ax.tick_params(labelsize=6.5)
            if r == len(archs) - 1:
                ax.set_xlabel("$P_1$")
            if c == 0:
                ax.set_ylabel("$P_2$")
    axes[0][1].legend(fontsize=6.5, frameon=False, loc="best", ncol=2)
    fig.suptitle(f"Operator-space trajectories, depth {depth}, width {hidden}. "
                 f"Dot = initialisation, star = OLS solution; axes scale per panel. "
                 f"For relu / leaky / crelu the path is the data-averaged operator.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
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


def build_depths(arch="linear", depths=(2, 32), hidden=32, n=128, seed=0, steps=300,
                 dev=None, inits=("xavier", "orthogonal"), cache=None):
    """The same sweep for one architecture at several depths, for a depth-contrast figure."""
    import pickle
    dev = dev or "cpu"
    X, y = dataset(n, seed, dev)
    Xc, yc = X.cpu(), y.cpu()
    Pstar = torch.linalg.lstsq(Xc, yc.unsqueeze(1)).solution.reshape(2).numpy()
    cache = Path(cache) if cache else None
    out = pickle.loads(cache.read_bytes()) if (cache and cache.exists()) else {}
    scales = out.setdefault("_scales", {})
    for depth in depths:
        for init in (inits or INITS):
            Ws = gpu.init_net(2, hidden, 1, depth, arch, seed, dev, dtype=torch.float64, init=init)
            scales[(depth, init)] = float(operators(Ws, X, arch).abs().max())
            for arm in ARMS:
                if (depth, init, arm) in out:
                    continue
                b = best_for(arch, init, arm, depth, hidden, seed, X, y, steps, dev)
                if b is not None:
                    end, lr, p, sp, start = b
                    out[(depth, init, arm)] = dict(lr=lr, end=end, path=p, spread=sp, start=start)
                if cache is not None:
                    cache.write_bytes(pickle.dumps(out))
            print(f"  L={depth} {init}: {sum(1 for k in out if k != '_scales' and k[0]==depth and k[1]==init)}"
                  f"/{len(ARMS)} arms, |P| at init = {scales[(depth, init)]:.1e}", flush=True)
    return out, Pstar, Xc.numpy(), yc.numpy()


def figure_depths(data, Pstar, X, y, arch="linear", depths=(2, 32),
                  inits=("xavier", "orthogonal")):
    """Rows are depths, columns are initialisations; one coloured path per optimiser."""
    import matplotlib.pyplot as plt
    inits = inits or INITS
    scales = data.get("_scales", {})
    fig, axes = plt.subplots(len(depths), len(inits),
                             figsize=(3.6 * len(inits), 3.3 * len(depths)), squeeze=False)
    for r, depth in enumerate(depths):
        for c, init in enumerate(inits):
            ax = axes[r][c]
            drawn = [m for m in ARMS if (depth, init, m) in data]
            sc = scales.get((depth, init))
            note = f"$\\|P\\|_\\infty$ at init: {sc:.1e}" if sc is not None else ""
            if not drawn:
                ax.text(0.5, 0.55, "no optimiser converged", transform=ax.transAxes,
                        ha="center", fontsize=8, color="#b03030")
                ax.text(0.5, 0.45, note, transform=ax.transAxes, ha="center", fontsize=7,
                        color="#b03030")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"$L={depth}$ / {init}", fontsize=9)
                continue
            pts = np.vstack([np.concatenate([data[(depth, init, m)]["path"] for m in drawn]), Pstar])
            lo, hi = pts.min(0), pts.max(0)
            pad = np.maximum((hi - lo) * 0.15, np.maximum(np.abs(hi), 1e-3) * 0.05)
            lo, hi = lo - pad, hi + pad
            gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 120), np.linspace(lo[1], hi[1], 120))
            Z = np.stack([((X @ np.array([a, b]) - y) ** 2).mean()
                          for a, b in zip(gx.ravel(), gy.ravel())]).reshape(gx.shape)
            ax.contour(gx, gy, Z, levels=12, colors="#d0d0d0", linewidths=0.6, zorder=0)
            for arm in drawn:
                p = data[(depth, init, arm)]["path"]
                ax.plot(p[:, 0], p[:, 1], "-", color=COLOR[arm], lw=1.3, alpha=0.85,
                        label=LABEL[arm] if r == 0 and c == 1 else None, zorder=3)
                ax.plot(p[0, 0], p[0, 1], "o", color=COLOR[arm], ms=3.5, zorder=4)
            ax.plot(*Pstar, "k*", ms=12, zorder=5)
            ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=6.5, color="#555555")
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_title(f"$L={depth}$ / {init}  ({len(drawn)}/{len(ARMS)} arms)", fontsize=9)
            ax.tick_params(labelsize=6.5)
            if r == len(depths) - 1:
                ax.set_xlabel("$P_1$")
            if c == 0:
                ax.set_ylabel("$P_2$")
    axes[0][1].legend(fontsize=6.5, frameon=False, loc="best", ncol=2)
    fig.suptitle(f"Operator-space trajectories, {arch} chain, width 32. "
                 f"Dot = initialisation, star = OLS solution; axes scale per panel.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig
