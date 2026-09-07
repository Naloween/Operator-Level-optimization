"""Reading runs back off disk.

Notebooks should ask for "every run in this sweep, indexed by method and depth" and get
it, rather than each notebook growing its own parsing of directory names. A run is
identified by its `meta.json` and `config.yaml`, never by string-matching its folder --
the folder name is a convenience for humans, and a sweep that adds an axis would silently
break any code that parsed it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from olo.config import load_yaml


@dataclass
class Run:
    """One run directory: its config, its metadata, its metrics, its arrays."""

    path: Path
    config: dict[str, Any]
    meta: dict[str, Any]
    rows: list[dict[str, Any]]
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    # -- identity -----------------------------------------------------------

    @property
    def method(self) -> str:
        return self.config["optim"]["type"]

    @property
    def model(self) -> str:
        return self.config["model"]["type"]

    @property
    def depth(self) -> int:
        return int(self.config["model"].get("depth", 1))

    @property
    def init(self) -> str:
        return self.config["model"].get("init", "xavier")

    @property
    def lr(self) -> float:
        return float(self.config["optim"]["lr"])

    @property
    def seed(self) -> int:
        return int(self.config.get("seed", 0))

    # -- series -------------------------------------------------------------

    def series(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """(steps, values) for a metric, skipping steps where it was not logged."""
        pairs = [(r["step"], r[key]) for r in self.rows if key in r and r[key] is not None]
        if not pairs:
            return np.array([]), np.array([])
        return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs], dtype=float)

    def final(self, key: str) -> float:
        _, v = self.series(key)
        return float(v[-1]) if len(v) else float("nan")

    def best(self, key: str, mode: str = "min") -> float:
        """Best finite value of a metric. Diverged runs report inf rather than nan.

        A run that blew up must not silently win a `min` by producing nan -- it has to
        compare as the worst possible value, or a sweep summary would report the diverged
        cell as the best one.
        """
        _, v = self.series(key)
        finite = v[np.isfinite(v)]
        if not len(finite):
            return float("inf") if mode == "min" else float("-inf")
        return float(finite.min() if mode == "min" else finite.max())

    @property
    def diverged(self) -> bool:
        _, v = self.series("eval_primary")
        return len(v) > 0 and not np.isfinite(v[-1])

    def array(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(steps, values) for a logged array such as the singular-value spectrum."""
        if name not in self.arrays:
            return np.array([]), np.array([])
        return self.arrays[f"{name}_steps"], self.arrays[name]

    @property
    def test_metrics(self) -> dict[str, float]:
        """What `test.json` holds: the split scored once, after training."""
        path = self.path / "test.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def value(self, key: str, mode: str = "min") -> float:
        """One number for this run, from wherever that metric lives.

        A `test_*` key comes from `test.json` -- a single measurement of the final model,
        with no best-over-training to take, because taking one would be selecting on test.
        Everything else is a training-time series and reports its best.
        """
        if key.startswith("test_"):
            v = self.test_metrics.get(key, float("nan"))
            return float(v) if isinstance(v, (int, float)) else float("nan")
        return self.best(key, mode)


def load_run(path: str | Path) -> Run:
    path = Path(path)
    metrics = path / "metrics.jsonl"
    rows = [json.loads(l) for l in metrics.open()] if metrics.exists() else []
    arrays = {}
    if (path / "arrays.npz").exists():
        with np.load(path / "arrays.npz") as z:
            arrays = {k: z[k] for k in z.files}
    return Run(
        path=path,
        config=load_yaml((path / "config.yaml").read_text()),
        meta=json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {},
        rows=rows,
        arrays=arrays,
    )


def load_runs(root: str | Path = "runs", pattern: str = "*") -> list[Run]:
    """Every completed run under `root` whose directory name matches `pattern`.

    Runs without metrics are skipped: a directory holding only a config is one that
    crashed before its first log, and including it would put an empty series into every
    plot that touches the sweep.
    """
    out = []
    for cfg in sorted(Path(root).glob(f"{pattern}/*/config.yaml")):
        run = load_run(cfg.parent)
        if run.rows:
            out.append(run)
    return out


def group_by(runs: list[Run], *keys: str) -> dict[tuple, list[Run]]:
    """Index runs by one or more of their identity properties."""
    out: dict[tuple, list[Run]] = {}
    for r in runs:
        out.setdefault(tuple(getattr(r, k) for k in keys), []).append(r)
    return out


def best_per(runs: list[Run], *keys: str, metric: str = "eval_primary",
             mode: str = "min") -> dict[tuple, Run]:
    """The best run per group -- the tuned-hyperparameter view of a sweep.

    This is how a learning-rate sweep should be read: every method at its own best, which
    is the only comparison that says something about the methods rather than about the
    step size they happened to share.
    """
    out = {}
    for key, group in group_by(runs, *keys).items():
        pick = min if mode == "min" else max
        out[key] = pick(group, key=lambda r: r.best(metric, mode))
    return out


def sweep_table(runs: list[Run], row: str, col: str, metric: str = "eval_primary",
                mode: str = "min") -> tuple[list, list, np.ndarray]:
    """(row_values, col_values, matrix) for a two-axis sweep, e.g. method x lr."""
    rows = sorted({getattr(r, row) for r in runs}, key=_sortable)
    cols = sorted({getattr(r, col) for r in runs}, key=_sortable)
    M = np.full((len(rows), len(cols)), np.nan)
    for r in runs:
        i, j = rows.index(getattr(r, row)), cols.index(getattr(r, col))
        v = r.best(metric, mode)
        if np.isnan(M[i, j]) or (v < M[i, j] if mode == "min" else v > M[i, j]):
            M[i, j] = v
    return rows, cols, M


def _sortable(v):
    return (0, v) if isinstance(v, (int, float)) else (1, str(v))
