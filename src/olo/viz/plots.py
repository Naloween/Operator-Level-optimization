"""Figure builders, one per question the experiments ask.

Each function takes runs and returns an axes, so notebooks stay thin and the figures a
paper ships are the same objects a notebook shows. Conventions held throughout: one y
axis per plot (never two scales), a legend whenever more than one series is drawn, direct
labels when four or fewer, and identity carried by color *and* line style.
"""
from __future__ import annotations

import numpy as np

from olo.viz import style as S
from olo.viz.loading import Run, best_per, group_by, sweep_table


def convergence(runs: list[Run], ax=None, metric: str = "eval_primary",
                label_key: str = "method", title: str | None = None):
    """Metric against step, one line per run. The basic "did it train?" figure."""
    ax = _axes(ax)
    for r in sorted(runs, key=lambda r: r.method):
        steps, vals = r.series(metric)
        if not len(steps):
            continue
        ls, mk = S.style(r.method)
        ax.plot(steps, vals, ls, color=S.color(r.method), marker=mk, markevery=max(1, len(steps) // 8),
                label=_label(r, label_key))
    ax.set_xlabel("step")
    ax.set_ylabel(_metric_label(metric))
    ax.set_yscale("log")
    _finish(ax, title, n_series=len(runs))
    return ax


def lr_grid(runs: list[Run], ax=None, metric: str = "eval_primary", title: str | None = None,
            cap: float | None = None):
    """Best metric against learning rate, one line per method.

    The figure that answers "is this a real difference between methods, or a difference in
    step size?" -- each method's whole curve is shown, so its best is visible rather than
    asserted, and a method that simply needed a smaller step says so.

    Learning rates that blow up produce values like 1e78, which would compress every
    meaningful difference into one pixel of a log axis. Those cells are clipped to a cap
    and drawn as hollow markers: still shown, still counted against the method, but no
    longer setting the scale.

    The cap defaults to an order of magnitude above the *smallest* first-eval value in the
    sweep. Every run shares an initialization, so that value is the initialization level
    itself -- and a run whose very first eval already sits far above it has diverged
    before the first measurement, which is precisely the case the scale must not be set
    by.
    """
    ax = _axes(ax)
    if cap is None:
        starts = [r.series(metric)[1][0] for r in runs if len(r.series(metric)[1])]
        finite = [s for s in starts if np.isfinite(s)]
        cap = 10 * min(finite) if finite else np.inf

    for (method,), group in sorted(group_by(runs, "method").items()):
        pts = sorted((r.lr, r.best(metric)) for r in group)
        lrs = np.array([p[0] for p in pts])
        raw = np.array([p[1] for p in pts])
        over = ~np.isfinite(raw) | (raw > cap)
        vals = np.where(over, cap, raw)
        ls, mk = S.style(method)
        ax.plot(lrs, vals, ls, color=S.color(method), marker=mk, label=S.label(method))
        if over.any():                       # diverged: shown, but not to scale
            ax.plot(lrs[over], vals[over], mk, color=S.color(method),
                    markerfacecolor="none", markersize=9, linestyle="none")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"best {_metric_label(metric)}")
    if np.isfinite(cap):
        ax.axhline(cap, color="#8a8a85", lw=0.8, ls=":", zorder=0)
        ax.annotate("diverged (clipped)", xy=(ax.get_xlim()[0], cap), fontsize=7,
                    color="#8a8a85", va="bottom", ha="left")
    _finish(ax, title, n_series=len(set(r.method for r in runs)))
    return ax


def depth_wall(runs: list[Run], ax=None, metric: str = "eval_primary",
               title: str | None = None):
    """Tuned performance against depth, one line per method.

    Every point is that method's best over the whole learning-rate grid at that depth, so
    a rise in the curve is a depth effect and not an untuned step size.
    """
    ax = _axes(ax)
    best = best_per(runs, "method", "depth", metric=metric)
    for method in sorted({m for m, _ in best}):
        pts = sorted((d, best[(m, d)].best(metric)) for m, d in best if m == method)
        ls, mk = S.style(method)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=S.color(method),
                marker=mk, label=S.label(method))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("depth $L$")
    ax.set_ylabel(f"best {_metric_label(metric)}")
    _finish(ax, title, n_series=len({m for m, _ in best}))
    return ax


def mismatch_vs_depth(runs: list[Run], ax=None, key: str = "cos_gd_first_order",
                      title: str | None = None):
    """Operator-step alignment against depth: the paper's central quantity, measured.

    Linear y axis, because this is a cosine in [-1, 1] and a log scale would misrepresent
    how far from 1 a value sits.
    """
    ax = _axes(ax)
    for (method,), group in sorted(group_by(runs, "method").items()):
        pts = []
        for r in sorted(group, key=lambda r: r.depth):
            _, v = r.series(key)
            if len(v):
                pts.append((r.depth, float(v[0])))
        if not pts:
            continue
        ls, mk = S.style(method)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=S.color(method),
                marker=mk, label=S.label(method))
    ax.axhline(1.0, color="#8a8a85", lw=0.8, ls=":", zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("depth $L$")
    ax.set_ylabel(r"$\cos(\Delta P,\ -\eta G)$")
    _finish(ax, title, n_series=len(set(r.method for r in runs)))
    return ax


def spectrum(run: Run, ax=None, steps: list[int] | None = None, title: str | None = None):
    """Operator singular values at several points in training, for one run.

    An ordered ramp rather than categorical hues: the series are the same quantity at
    different times, not different entities.
    """
    ax = _axes(ax)
    all_steps, sv = run.array("singular_values")
    if not len(all_steps):
        return ax
    pick = list(range(len(all_steps))) if steps is None else \
        [int(np.argmin(np.abs(all_steps - s))) for s in steps]
    colors = S.ramp(len(pick))
    for c, i in zip(colors, pick):
        ax.plot(np.arange(1, sv.shape[1] + 1), sv[i], "-", color=c, marker="o",
                label=f"step {all_steps[i]}")
    ax.set_yscale("log")
    ax.set_xlabel("index")
    ax.set_ylabel(r"$\sigma_i(P)$")
    _finish(ax, title or f"{S.label(run.method)}, {S.MODEL_LABELS.get(run.model, run.model)}",
            n_series=len(pick))
    return ax


def cost(runs: list[Run], ax=None, title: str | None = None):
    """Measured seconds per step against depth -- the price of the exact solve."""
    ax = _axes(ax)
    for (method,), group in sorted(group_by(runs, "method").items()):
        pts = []
        for r in sorted(group, key=lambda r: r.depth):
            _, v = r.series("step_seconds_mean")
            if len(v):
                pts.append((r.depth, float(v[-1])))
        if not pts:
            continue
        ls, mk = S.style(method)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=S.color(method),
                marker=mk, label=S.label(method))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("depth $L$")
    ax.set_ylabel("seconds / step")
    _finish(ax, title, n_series=len(set(r.method for r in runs)))
    return ax


# ---------------------------------------------------------------------------


def _axes(ax):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    S.apply_style()
    return plt.subplots(figsize=(5.2, 3.4))[1]


def _finish(ax, title, n_series: int) -> None:
    if title:
        ax.set_title(title, loc="left", pad=8)
    if n_series >= 2:                      # identity is never color-alone
        ax.legend(fontsize=8, ncol=1 if n_series <= 4 else 2)


def _label(run: Run, key: str) -> str:
    if key == "method":
        return S.label(run.method)
    if key == "depth":
        return f"$L={run.depth}$"
    if key == "lr":
        return f"lr={run.lr:g}"
    return str(getattr(run, key, key))


def _metric_label(metric: str) -> str:
    return {
        "eval_primary": "primary metric",
        "eval_rel_operator_error": r"$\|P-P^\star\|_F/\|P^\star\|_F$",
        "eval_loss": "loss",
        "eval_val_loss": "validation loss",
        "eval_val_accuracy": "validation accuracy",
        "loss": "training loss",
    }.get(metric, metric)
