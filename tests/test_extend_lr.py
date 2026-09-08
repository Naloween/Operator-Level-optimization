"""Grid-edge detection: is a method's reported score tuned, or just the best of a bad range?

A cell whose optimum sits at the smallest learning rate tried has not been tuned -- its
score is a lower bound. At depth that distinction decides whether a curve shows a depth
wall or a grid edge, because the usable step size falls as O(1/L) (olo.theory.deep_linear)
and a range that brackets the optimum at depth 2 can sit entirely above it at depth 256.
"""
from __future__ import annotations

import json

import pytest
import yaml

from olo.extend_lr import edge_cells, next_lr
from olo.viz.loading import load_runs


def _run(root, name, depth, method, lr, accuracy, model="crelu_mlp", init="looks_linear"):
    d = root / name / "seed0"
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({
        "name": name, "seed": 0,
        "model": {"type": model, "init": init, "depth": depth, "width": 32},
        "task": {"type": "mnist"}, "optim": {"type": method, "lr": lr},
    }))
    (d / "meta.json").write_text("{}")
    (d / "metrics.jsonl").write_text(
        json.dumps({"step": 0, "eval_val_accuracy": accuracy}) + "\n")


def test_optimum_at_the_smallest_rate_is_flagged(tmp_path):
    """Monotone improvement toward the bottom of the range means the optimum is below it."""
    for lr, acc in ((0.1, 0.20), (0.01, 0.50), (0.001, 0.80)):
        _run(tmp_path, f"g_adam_lr{lr}", 32, "adam", lr, acc)
    cells = edge_cells(load_runs(tmp_path, "g_*"), "eval_val_accuracy", "max")
    assert len(cells) == 1
    assert cells[0]["method"] == "adam" and cells[0]["best_lr"] == 0.001
    assert next_lr(cells[0]) == pytest.approx(1e-4)


def test_interior_optimum_is_not_flagged(tmp_path):
    """A peak inside the range is a tuned result and must not trigger more runs."""
    for lr, acc in ((0.1, 0.20), (0.01, 0.90), (0.001, 0.55)):
        _run(tmp_path, f"g_adam_lr{lr}", 32, "adam", lr, acc)
    assert edge_cells(load_runs(tmp_path, "g_*"), "eval_val_accuracy", "max") == []


def test_each_depth_and_method_is_judged_separately(tmp_path):
    """Depth is the axis of interest, so a shallow cell being fine says nothing about a deep one."""
    for lr, acc in ((0.1, 0.2), (0.01, 0.9), (0.001, 0.5)):      # L=2 interior
        _run(tmp_path, f"g_L2_adam_lr{lr}", 2, "adam", lr, acc)
    for lr, acc in ((0.1, 0.1), (0.01, 0.3), (0.001, 0.7)):      # L=256 at the edge
        _run(tmp_path, f"g_L256_adam_lr{lr}", 256, "adam", lr, acc)
    for lr, acc in ((0.1, 0.2), (0.01, 0.8), (0.001, 0.4)):      # L=256 other method, fine
        _run(tmp_path, f"g_L256_muon_lr{lr}", 256, "muon", lr, acc)

    cells = edge_cells(load_runs(tmp_path, "g_*"), "eval_val_accuracy", "max")
    assert [(c["depth"], c["method"]) for c in cells] == [(256, "adam")]


def test_model_families_are_kept_apart(tmp_path):
    """The two families are different experiments; one must not mask the other's edge."""
    for lr, acc in ((0.01, 0.9), (0.001, 0.5)):
        _run(tmp_path, f"g_crelu_lr{lr}", 32, "adam", lr, acc)
    for lr, acc in ((0.01, 0.3), (0.001, 0.6)):
        _run(tmp_path, f"g_relu_lr{lr}", 32, "adam", lr, acc,
             model="relu_mlp", init="xavier")
    cells = edge_cells(load_runs(tmp_path, "g_*"), "eval_val_accuracy", "max")
    assert [c["model"] for c in cells] == ["relu_mlp"]


def test_a_loss_metric_extends_in_the_right_direction(tmp_path):
    """With a minimized metric the best cell is the smallest, not the largest."""
    for lr, loss in ((0.1, 2.0), (0.01, 1.0), (0.001, 0.3)):
        d = tmp_path / f"g_adam_lr{lr}" / "seed0"
        d.mkdir(parents=True)
        (d / "config.yaml").write_text(yaml.safe_dump({
            "name": f"g_adam_lr{lr}", "seed": 0,
            "model": {"type": "crelu_mlp", "init": "looks_linear", "depth": 32, "width": 32},
            "task": {"type": "mnist"}, "optim": {"type": "adam", "lr": lr}}))
        (d / "meta.json").write_text("{}")
        (d / "metrics.jsonl").write_text(json.dumps({"step": 0, "eval_primary": loss}) + "\n")
    cells = edge_cells(load_runs(tmp_path, "g_*"), "eval_primary", "min")
    assert len(cells) == 1 and cells[0]["best_lr"] == 0.001


def test_extension_stops_at_the_floor(tmp_path):
    for lr, acc in ((1e-7, 0.5), (1e-8, 0.8)):
        _run(tmp_path, f"g_adam_lr{lr}", 32, "adam", lr, acc)
    cells = edge_cells(load_runs(tmp_path, "g_*"), "eval_val_accuracy", "max")
    assert len(cells) == 1
    assert next_lr(cells[0]) is None, "must not extend below the floor forever"


def test_single_rate_cells_are_ignored(tmp_path):
    """One point is not a grid; there is no edge to be at."""
    _run(tmp_path, "g_adam_only", 32, "adam", 0.01, 0.9)
    assert edge_cells(load_runs(tmp_path, "g_*"), "eval_val_accuracy", "max") == []


def test_generated_names_stay_distinct_across_cells(tmp_path):
    """Two cells extended in the same round must not be handed the same run name.

    Rebuilding the name from the sweep's base drops every axis but the learning rate, so
    depth 8 and depth 32 both became "<base>__typeadam_lr1em05" -- one silently standing
    in for the other, or refused as a clash. The template's own name already encodes the
    axes; only its lr suffix should change.
    """
    from olo.extend_lr import edge_cells, launch

    for depth in (8, 32):
        for lr, acc in ((0.01, 0.2), (0.001, 0.5), (0.0001, 0.9)):
            _run(tmp_path, f"e_depth{depth}_lr{lr}", depth, "adam", lr, acc)

    cells = edge_cells(load_runs(tmp_path, "e_*"), "eval_val_accuracy", "max")
    assert len(cells) == 2

    names = []
    import olo.extend_lr as m
    original = m.subprocess.run
    m.subprocess.run = lambda cmd, **kw: names.append(
        cmd[cmd.index("--set") + 1]) or type("R", (), {"returncode": 0})()
    try:
        for c in cells:
            launch(c, 1e-5, str(tmp_path), dry_run=False)
    finally:
        m.subprocess.run = original

    assert len(set(names)) == 2, f"names collided: {names}"
    assert any("8" in n for n in names) and any("32" in n for n in names)
