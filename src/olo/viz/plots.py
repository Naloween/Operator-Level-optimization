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
            cap: float | None = None, mode: str | None = None, yscale: str = "log"):
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
    mode = _infer_mode(metric) if mode is None else mode
    if cap is None:
        cap = _divergence_cap(runs, metric, mode)

    clipped = False
    pick = min if mode == "min" else max
    for (method,), group in sorted(group_by(runs, "method").items()):
        # Aggregate per learning rate. A sweep that also varies depth (or seed) has
        # several runs at each rate, and plotting them as separate points would draw the
        # same x twice and let the line double back on itself.
        by_lr: dict[float, list[float]] = {}
        for r in group:
            by_lr.setdefault(r.lr, []).append(r.value(metric, mode))
        pts = sorted((lr, pick(v)) for lr, v in by_lr.items())
        lrs = np.array([p[0] for p in pts])
        raw = np.array([p[1] for p in pts])
        over = ~np.isfinite(raw) | ((raw > cap) if cap is not None else False)
        vals = np.where(over, cap, raw) if cap is not None else raw
        ls, mk = S.style(method)
        ax.plot(lrs, vals, ls, color=S.color(method), marker=mk, label=S.label(method))
        if np.any(over):                     # diverged: shown, but not to scale
            clipped = True
            ax.plot(lrs[over], vals[over], mk, color=S.color(method),
                    markerfacecolor="none", markersize=9, linestyle="none")

    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"best {_metric_label(metric)}")
    _mark_cap(ax, cap, clipped)
    _finish(ax, title, n_series=len(set(r.method for r in runs)))
    return ax


def depth_wall(runs: list[Run], ax=None, metric: str = "eval_primary",
               title: str | None = None, mode: str | None = None,
               yscale: str = "linear", cap: float | None = None,
               select_metric: str | None = None):
    """Tuned performance against depth, one line per method.

    Every point is that method's best over the whole learning-rate grid at that depth, so
    a rise in the curve is a depth effect and not an untuned step size.

    `select_metric` is what the learning rate is chosen by; `metric` is what gets plotted.
    They differ whenever the reported number is a test metric: choosing the learning rate
    on test and then plotting test would make every point the best of nine attempts at the
    thing being reported. Pass `metric="test_accuracy"` with
    `select_metric="eval_val_accuracy"` and the selection stays on validation, where it
    belongs. Defaults to `metric`, which is correct as long as `metric` is a validation
    quantity.

    `mode` says which direction is "best" and defaults to the metric's own: minimize a
    loss or an error, maximize an accuracy. Getting this wrong is silent and total -- a
    grid read with the wrong sense reports each cell's *worst* learning rate as its
    result -- so it is inferred rather than left to the caller to remember.

    On a linear `yscale` a single diverged cell (values of 1e100 are routine in a
    learning-rate sweep) flattens every real difference to the axis. Values beyond `cap`
    are clipped and drawn hollow, as in `lr_grid`.
    """
    ax = _axes(ax)
    mode = _infer_mode(metric) if mode is None else mode
    select = metric if select_metric is None else select_metric
    best = best_per(runs, "method", "depth", metric=select, mode=_infer_mode(select))
    if cap is None and yscale == "linear":
        cap = _divergence_cap(runs, metric, mode)

    clipped = False
    for method in sorted({m for m, _ in best}):
        pts = sorted((d, best[(m, d)].value(metric, mode)) for m, d in best if m == method)
        xs = np.array([p[0] for p in pts], dtype=float)
        raw = np.array([p[1] for p in pts], dtype=float)
        over = ~np.isfinite(raw) | ((raw > cap) if cap is not None else False)
        vals = np.where(over, cap, raw) if cap is not None else raw
        ls, mk = S.style(method)
        ax.plot(xs, vals, ls, color=S.color(method), marker=mk, label=S.label(method))
        if np.any(over):
            clipped = True
            ax.plot(xs[over], vals[over], mk, color=S.color(method),
                    markerfacecolor="none", markersize=9, linestyle="none")

    ax.set_xscale("log", base=2)
    ax.set_yscale(yscale)
    ax.set_xlabel("depth $L$")
    ax.set_ylabel(f"best {_metric_label(metric)}")
    _mark_cap(ax, cap, clipped)
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


def _infer_mode(metric: str) -> str:
    """Which direction is "best" for this metric.

    Everything logged here is minimized -- losses, errors, residuals -- except accuracy.
    Inferring it keeps a caller from silently reading a grid backwards.
    """
    return "max" if "accuracy" in metric or metric.endswith("_acc") else "min"


def _divergence_cap(runs: list[Run], metric: str, mode: str) -> float | None:
    """Where to clip runs that blew up, so they do not set the axis scale.

    An order of magnitude past the initialization level. Runs in a sweep share an
    initialization, so the *best* first-eval value is that level; anything an order of
    magnitude worse has already failed, and its exact magnitude carries no information
    worth an axis decade.

    Returns None for a bounded metric -- an accuracy cannot run away, so clipping it would
    only hide real differences at the chance-level end, which for a depth experiment is
    the end that matters.
    """
    if mode == "max" or _is_bounded(metric):
        return None
    starts = [r.series(metric)[1][0] for r in runs if len(r.series(metric)[1])]
    finite = [s for s in starts if np.isfinite(s)]
    return 10 * min(finite) if finite else None


def _is_bounded(metric: str) -> bool:
    return "accuracy" in metric or metric.endswith("_acc")


def _mark_cap(ax, cap: float | None, clipped: bool) -> None:
    """Draw the clipping line only when something was actually clipped.

    Drawing it unconditionally forces the axis to include the cap, so a sweep where
    nothing diverged got a y-range stretched to ten times its initialization value with no
    data anywhere near the top -- the line itself became the outlier it was meant to
    contain.
    """
    if clipped and cap is not None and np.isfinite(cap):
        ax.axhline(cap, color="#8a8a85", lw=0.8, ls=":", zorder=0)
        ax.annotate("diverged (clipped)", xy=(ax.get_xlim()[0], cap), fontsize=7,
                    color="#8a8a85", va="bottom", ha="left")


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
        "test_accuracy": "test accuracy",
        "test_loss": "test loss",
        "eval_val_accuracy": "validation accuracy",
        "eval_rel_operator_error": r"$\|P-P^\star\|_F/\|P^\star\|_F$",
        "eval_loss": "loss",
        "eval_val_loss": "validation loss",
        "eval_val_accuracy": "validation accuracy",
        "loss": "training loss",
    }.get(metric, metric)
