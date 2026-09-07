"""Figure builders must not misrepresent the data they are handed.

A wrong number in a plot is worse than a crash: it is read, believed, and acted on. These
tests cover the ways a plotting helper can silently lie -- selecting the wrong end of a
grid, letting an annotation set the axis range, or reporting a metric chosen on the same
split it is reported from.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from olo.viz import depth_wall, load_runs, lr_grid  # noqa: E402
from olo.viz.plots import _divergence_cap, _infer_mode  # noqa: E402


def _cap_lines(ax):
    """Horizontal reference lines added by axhline, which spans the axes in x."""
    return [l for l in ax.get_lines()
            if tuple(l.get_xdata()) == (0, 1) and len(set(l.get_ydata())) == 1]


def _make_run(root, name, depth, method, lr, primary, accuracy, test_accuracy=None):
    d = root / name / "seed0"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({
        "name": name, "seed": 0,
        "model": {"type": "crelu_mlp", "depth": depth, "width": 8, "init": "looks_linear"},
        "task": {"type": "mnist"}, "optim": {"type": method, "lr": lr},
    }))
    (d / "meta.json").write_text("{}")
    with (d / "metrics.jsonl").open("w") as fh:
        for step, v in enumerate(primary):
            fh.write(json.dumps({"step": step, "eval_primary": v,
                                 "eval_val_accuracy": accuracy[step]}) + "\n")
    if test_accuracy is not None:
        (d / "test.json").write_text(json.dumps({"test_accuracy": test_accuracy}))


@pytest.fixture
def sweep(tmp_path):
    """Two methods x two depths x two learning rates, nothing diverging."""
    for depth in (2, 4):
        for method, base in (("adam", 0.30), ("muon", 0.50)):
            for lr, off in ((0.1, 0.0), (0.01, -0.10)):
                p = [base + off + 0.02 * i for i in range(3)][::-1]
                a = [0.90 - off - 0.01 * i for i in range(3)]
                _make_run(tmp_path, f"e_d{depth}_{method}_lr{lr}", depth, method, lr,
                          p, a, test_accuracy=a[-1] - 0.005)
    return load_runs(tmp_path, "e_*")


def test_axis_is_not_stretched_by_a_cap_nothing_reached(sweep):
    """The clipping line must not appear -- and must not set the range -- when unused.

    The cap sits an order of magnitude past the initialization value. Drawing it
    unconditionally stretched a plot whose data spanned 0.2 to 4 out to 14, making the
    annotation the only thing near the top of its own axis.
    """
    ax = depth_wall(sweep, ax=plt.subplots()[1], metric="eval_primary")
    lo, hi = ax.get_ylim()
    values = [r.value("eval_primary") for r in sweep]
    assert hi < 2 * max(values), f"axis {lo:.3f}..{hi:.3f} far exceeds data max {max(values):.3f}"
    assert not _cap_lines(ax), "cap line drawn when nothing was clipped"


def test_cap_line_appears_when_a_run_actually_diverges(tmp_path):
    """A cell whose whole trajectory is past the cap is clipped, and says so.

    The diverged run is bad from its first evaluation onward: `best` takes the minimum
    over training, so a run that starts healthy and only later explodes is still scored by
    its healthy moment -- which is the behaviour early stopping would have given it.
    """
    _make_run(tmp_path, "ok", 2, "adam", 0.01, [1.0, 0.5, 0.2], [0.5, 0.7, 0.9])
    _make_run(tmp_path, "boom", 4, "adam", 0.1, [1e6, 1e11, 1e12], [0.1, 0.1, 0.1])
    runs = load_runs(tmp_path, "*")
    ax = depth_wall(runs, ax=plt.subplots()[1], metric="eval_primary")
    assert _cap_lines(ax), "no cap line for a diverged run"
    assert ax.get_ylim()[1] < 1e6


def test_accuracy_is_never_clipped(sweep):
    """A bounded metric cannot run away, and clipping it would hide the chance-level end."""
    assert _divergence_cap(sweep, "eval_val_accuracy", "max") is None
    ax = depth_wall(sweep, ax=plt.subplots()[1], metric="eval_val_accuracy")
    assert not _cap_lines(ax)


def test_best_direction_follows_the_metric(sweep):
    """Accuracy maximizes, loss minimizes -- read backwards, a grid reports its worst cell."""
    assert _infer_mode("eval_val_accuracy") == "max"
    assert _infer_mode("test_accuracy") == "max"
    assert _infer_mode("eval_primary") == "min"

    ax = depth_wall(sweep, ax=plt.subplots()[1], metric="eval_val_accuracy")
    plotted = {tuple(l.get_ydata()) for l in ax.get_lines() if len(l.get_ydata()) == 2}
    best = max(r.value("eval_val_accuracy", "max") for r in sweep)
    assert any(np.isclose(max(y), best) for y in plotted), "the best cell was not plotted"


def test_reporting_test_metrics_selects_on_validation(sweep):
    """`metric` and `select_metric` must be able to differ, or test becomes a selected set."""
    ax = depth_wall(sweep, ax=plt.subplots()[1], metric="test_accuracy",
                    select_metric="eval_val_accuracy")
    ys = np.concatenate([l.get_ydata() for l in ax.get_lines() if len(l.get_ydata()) == 2])
    assert np.all(ys <= 1.0) and np.all(ys >= 0.0), ys
    # every plotted value must be a real test number, not a best-of over training
    known = {round(r.value("test_accuracy", "max"), 6) for r in sweep}
    assert {round(float(v), 6) for v in ys} <= known


def test_lr_grid_plots_every_learning_rate_tried(sweep):
    ax = lr_grid(sweep, ax=plt.subplots()[1], metric="eval_val_accuracy", yscale="linear")
    for line in ax.get_lines():
        xs = list(line.get_xdata())
        if len(xs) > 1:
            # one point per learning rate, even though the sweep spans two depths
            assert sorted(xs) == [0.01, 0.1], xs
