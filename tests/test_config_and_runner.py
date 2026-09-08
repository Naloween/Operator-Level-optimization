"""Config parsing, sweep expansion, and one end-to-end run.

The harness is what makes comparisons fair, so its failure modes are the ones that would
silently corrupt results rather than crash: a learning rate parsed as a string, a sweep
axis that does not vary, per-method params leaking into the wrong optimizer. Those are
what these tests are for.
"""
from __future__ import annotations

import json

import pytest
import yaml

from olo.config import ConfigError, RunCfg, expand, load_yaml, parse_scalar, set_path


def _minimal(**over) -> dict:
    raw = {
        "name": "t",
        "model": {"type": "deep_linear", "width": 4, "depth": 2, "init": "identity"},
        "task": {"type": "teacher_student", "d": 4, "n": 8},
        "optim": {"type": "als", "lr": 0.5, "als": {"lam": 1e-6}},
        "train": {"steps": 3, "eval_every": 1},
        "diagnostics": {"every": 1},
        "device": "cpu",
        "dtype": "float64",
    }
    raw.update(over)
    return raw


# -- parsing ---------------------------------------------------------------


def test_scientific_notation_parses_as_a_number():
    """YAML 1.1 would hand back the string "1e-3"; a string learning rate is a silent bug."""
    assert parse_scalar("1e-3") == pytest.approx(1e-3)
    assert isinstance(parse_scalar("1e-3"), float)
    assert parse_scalar("1.0e-3") == pytest.approx(1e-3)
    assert parse_scalar("true") is True
    assert parse_scalar("null") is None
    assert parse_scalar("als") == "als"


def test_scientific_notation_in_config_files_too():
    """The same trap applies to values written in a config, where it is easier to miss.

    `stop_above: 1.0e8` is a string under YAML 1.1 (no sign on the exponent), and compared
    against a float it raises from deep inside the run loop rather than at load.
    """
    raw = load_yaml("a: 1e-3\nb: 1.0e8\nc: 1e6\nd: -2.5e-7\ne: not_a_number\nf: '1e-3'\n")
    for key in "abcd":
        assert isinstance(raw[key], float), (key, raw[key])
    assert raw["b"] == pytest.approx(1e8)
    assert raw["e"] == "not_a_number"
    assert raw["f"] == "1e-3", "an explicitly quoted value must stay a string"


def test_every_shipped_config_loads_and_validates():
    """A config that only fails at run time wastes whatever queue it was submitted to."""
    from pathlib import Path

    paths = sorted(Path("configs/experiments").glob("*.yaml"))
    assert paths, "no experiment configs found"
    for p in paths:
        cfg = RunCfg.load(p)
        assert isinstance(cfg.optim.params.get("lr", 0.0), (int, float)), p
        for key in ("stop_below", "stop_above"):
            v = getattr(cfg.train, key)
            assert v is None or isinstance(v, (int, float)), (p, key, v)


def test_unknown_keys_are_rejected():
    """A typo'd key must fail loudly rather than be ignored for a whole sweep."""
    with pytest.raises(ConfigError, match="unknown top-level"):
        RunCfg.from_dict(_minimal(stpes=10))
    with pytest.raises(ConfigError, match="train"):
        RunCfg.from_dict(_minimal(train={"stpes": 10}))


def test_missing_required_section_is_named():
    raw = _minimal()
    del raw["task"]
    with pytest.raises(ConfigError, match="task"):
        RunCfg.from_dict(raw)


def test_roundtrip_through_yaml(tmp_path):
    cfg = RunCfg.from_dict(_minimal())
    cfg.save(tmp_path / "c.yaml")
    again = RunCfg.load(tmp_path / "c.yaml")
    assert again.to_dict() == cfg.to_dict()


# -- sweeps ----------------------------------------------------------------


def test_sweep_expands_and_names_only_the_varied_axes():
    out = list(expand(_minimal(), {"optim.lr": [0.1, 0.2], "model.depth": [4]}))
    assert len(out) == 2
    assert [c["optim"]["lr"] for c in out] == [0.1, 0.2]
    assert all(c["model"]["depth"] == 4 for c in out)
    names = [c["name"] for c in out]
    assert "depth" not in names[0], "a single-valued axis should not enter the run name"
    assert names[0] != names[1], "swept runs must land in distinct directories"


def test_sweep_values_actually_reach_nested_keys():
    out = list(expand(_minimal(), {"optim.als.lam": [1e-8]}))
    assert out[0]["optim"]["als"]["lam"] == 1e-8


def test_set_path_creates_missing_levels():
    raw = {}
    set_path(raw, "a.b.c", 1)
    assert raw == {"a": {"b": {"c": 1}}}


# -- end to end ------------------------------------------------------------


def test_run_writes_a_complete_directory(tmp_path):
    from olo.runner import run

    cfg = RunCfg.from_dict(_minimal(out_dir=str(tmp_path)))
    out = run(cfg, progress=False)

    for name in ("config.yaml", "meta.json", "metrics.jsonl", "ckpt_final.pt"):
        assert (out / name).exists(), name

    rows = [json.loads(l) for l in (out / "metrics.jsonl").open()]
    assert rows and "loss" in rows[0]
    assert "eval_primary" in rows[0]
    assert "cos_target" in rows[0], "diagnostics were requested but not logged"

    meta = json.loads((out / "meta.json").read_text())
    assert meta["model"]["depth"] == 2
    assert meta["init"] == "identity"


def test_per_method_blocks_do_not_leak_between_optimizers(tmp_path):
    """`--sweep optim.type=...` must work from one config: this is the fairness mechanism.

    Adam would raise a TypeError if ALS's `lam` or `n_sweeps` were forwarded to it.
    """
    from olo.runner import run

    raw = _minimal(out_dir=str(tmp_path))
    raw["optim"] = {"type": "adam", "lr": 1e-3,
                    "als": {"lam": 1e-6, "n_sweeps": 4}, "adam": {"betas": [0.9, 0.99]}}
    out = run(RunCfg.from_dict(raw), progress=False)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["optim"]["type"] == "adam"
    assert meta["optim"]["betas"] == [0.9, 0.99]


def test_all_registered_components_import():
    """A registry entry pointing at a module that does not exist fails only at run time."""
    from olo.registry import MODELS, OPTIMS, TARGETS, TASKS, resolve

    for kind, table in (("model", MODELS), ("optim", OPTIMS),
                        ("task", TASKS), ("target", TARGETS)):
        for name in table:
            assert callable(resolve(kind, name)), f"{kind}:{name}"


def test_unknown_component_names_list_the_alternatives():
    from olo.registry import resolve

    with pytest.raises(KeyError, match="Available"):
        resolve("optim", "adamw")


# -- splits, stopping, and post-hoc evaluation ------------------------------


def test_validation_and_test_splits_are_disjoint():
    """Early stopping reads validation; if test leaked into it, test would be selected on."""
    from olo.tasks.teacher_student import TeacherStudent

    task = TeacherStudent(d=4, n=16, n_val=8, n_test=8, seed=0)
    for a, b in ((task.X, task.X_val), (task.X, task.X_test), (task.X_val, task.X_test)):
        for i in range(b.shape[0]):
            assert not (a == b[i]).all(dim=1).any(), "splits share a sample"


def test_task_test_metrics_are_not_logged_during_training(tmp_path):
    """Nothing in the loop may call `test()`; it is written once, afterwards."""
    from olo.runner import run

    seen = []
    from olo.tasks.teacher_student import TeacherStudent

    original = TeacherStudent.test
    TeacherStudent.test = lambda self, net: (seen.append(1), original(self, net))[1]
    try:
        cfg = RunCfg.from_dict(_minimal(out_dir=str(tmp_path), train={"steps": 6, "eval_every": 1}))
        out = run(cfg, progress=False)
    finally:
        TeacherStudent.test = original

    assert len(seen) == 1, f"test() called {len(seen)} times; it must be called once, at the end"
    assert (out / "test.json").exists()
    payload = json.loads((out / "test.json").read_text())
    assert "test_loss" in payload


def test_stall_detection_stops_a_run_that_is_not_improving(tmp_path):
    """A collapsed network sitting at its initialization loss should not burn the budget."""
    from olo.runner import run

    raw = _minimal(out_dir=str(tmp_path))
    raw["optim"] = {"type": "adam", "lr": 0.0}          # cannot improve, by construction
    raw["train"] = {"steps": 200, "eval_every": 1, "stop_patience": 3, "stop_min_delta": 1e-3}
    out = run(RunCfg.from_dict(raw), progress=False)

    rows = [json.loads(l) for l in (out / "metrics.jsonl").open()]
    assert rows[-1].get("stopped") == "stalled", rows[-1]
    assert rows[-1]["step"] < 20, "stall should be caught within a few evaluations"


def test_divergence_and_convergence_are_labelled_differently(tmp_path):
    from olo.runner import run

    raw = _minimal(out_dir=str(tmp_path))
    raw["train"] = {"steps": 50, "eval_every": 1, "stop_below": 1e6}   # trivially satisfied
    out = run(RunCfg.from_dict(raw), progress=False)
    rows = [json.loads(l) for l in (out / "metrics.jsonl").open()]
    assert rows[-1].get("stopped") == "converged"


def test_evaluate_reproduces_the_runs_own_test_metrics(tmp_path):
    """Rebuilding from config.yaml must give back exactly what the run wrote."""
    from olo.evaluate import evaluate_run
    from olo.runner import run

    cfg = RunCfg.from_dict(_minimal(out_dir=str(tmp_path)))
    out = run(cfg, progress=False)
    written = json.loads((out / "test.json").read_text())
    recomputed = evaluate_run(out)

    assert recomputed["test_test_loss"] == pytest.approx(written["test_loss"], rel=1e-9)


def test_diagnostics_run_on_a_subsample_not_the_training_batch(tmp_path):
    """The context stack is the loop's largest allocation and grows with depth.

    At MNIST's batch of 128 it reaches 3.4 GB at depth 64 and ~14 GB at depth 256 -- enough
    to take the machine down before it takes the run down, which is exactly what happened
    to the first depth sweep. The diagnostics must see only `diagnostics.batch_size`
    samples regardless of how large a batch training uses.
    """
    from olo.models.base import FactoredNet
    from olo.runner import run

    seen = []
    original = FactoredNet.all_contexts
    FactoredNet.all_contexts = lambda self, x, gates=None: (
        seen.append(x.shape[0]), original(self, x, gates))[1]
    try:
        raw = _minimal(out_dir=str(tmp_path))
        raw["task"] = {"type": "teacher_student", "d": 4, "n": 64}
        raw["train"] = {"steps": 4, "eval_every": 1, "batch_size": 32}
        raw["diagnostics"] = {"every": 1, "batch_size": 3}
        run(RunCfg.from_dict(raw), progress=False)
    finally:
        FactoredNet.all_contexts = original

    assert seen, "diagnostics never built contexts"
    assert set(seen) == {3}, f"contexts built on {set(seen)} samples, expected 3"


def test_reusing_a_run_directory_for_a_different_config_is_refused(tmp_path, capsys):
    """Silently skipping a name collision fills a sweep with the wrong runs.

    A value set with --set does not enter the run name, so a shell loop over depths that
    uses --set writes every depth to one directory. The first depth wins, every later one
    is skipped as "already exists", and the sweep reports success while containing a
    single depth repeated. That happened; this makes it loud.
    """
    from olo.run import main

    cfg_path = tmp_path / "c.yaml"
    RunCfg.from_dict(_minimal(out_dir=str(tmp_path / "runs"))).save(cfg_path)

    assert main([str(cfg_path), "--quiet"]) == 0                      # first depth
    rc = main([str(cfg_path), "--quiet", "--set", "model.depth=7"])    # second, same name

    assert rc == 1, "a config clash must fail, not be skipped"
    assert "REFUSED" in capsys.readouterr().err


def test_identical_reruns_are_still_skipped_quietly(tmp_path, capsys):
    """Resuming an interrupted sweep must stay cheap: same config, same name, no work."""
    from olo.run import main

    cfg_path = tmp_path / "c.yaml"
    RunCfg.from_dict(_minimal(out_dir=str(tmp_path / "runs"))).save(cfg_path)
    assert main([str(cfg_path), "--quiet"]) == 0
    assert main([str(cfg_path), "--quiet"]) == 0
    assert "skip" in capsys.readouterr().out
