"""Check a set of runs for the ways a sweep quietly stops meaning what it claims.

    python -m olo.audit "e02_*"

Three separate bugs during one depth sweep each removed cells without any error: a
skipped-but-unfinished run, a crash that left no directory, and a name collision that made
eight depths write to one place. All three produced output that looked entirely normal,
because a missing cell and a never-requested cell are the same thing to any analysis that
groups over what it finds.

So the checks here are about *coverage and consistency*, not about results:

  incomplete     a directory with no metrics.jsonl -- started, never finished
  empty          metrics written but containing nothing
  unscored       no test.json, so the run cannot enter a test-metric figure
  duplicated     one experimental cell (model, init, depth, method, lr) in two directories
  inconsistent   runs differing in settings that were supposed to be held fixed
  failed         runs that stopped on divergence or numerical breakdown

Exit status is nonzero when anything would distort a comparison, so this can gate the
analysis rather than being a thing to remember to run.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from olo.config import load_yaml
from olo.viz.loading import Run, load_runs

#: settings a sweep varies on purpose; everything else should match across runs
SWEPT = {"name", "seed", "model.depth", "model.type", "model.init", "optim.type", "optim.lr"}


def cell(run: Run) -> tuple:
    """The experimental cell a run occupies. Two runs sharing one are duplicates."""
    m = run.config["model"]
    return (m["type"], m.get("init"), run.depth, run.method, run.lr, run.seed)


def audit(root: str | Path, pattern: str) -> dict:
    root = Path(root)
    dirs = sorted(p.parent for p in root.glob(f"{pattern}/*/config.yaml"))
    runs = load_runs(root, pattern)

    report: dict = {
        "n_dirs": len(dirs), "n_runs": len(runs),
        "incomplete": [], "empty": [], "unscored": [],
        "duplicated": [], "inconsistent": [], "failed": Counter(),
    }

    for d in dirs:
        if not (d / "metrics.jsonl").exists():
            report["incomplete"].append(str(d))

    seen: dict[tuple, list[str]] = defaultdict(list)
    fixed: dict[str, set] = defaultdict(set)
    for r in runs:
        if not r.rows:
            report["empty"].append(str(r.path))
        if not (r.path / "test.json").exists():
            report["unscored"].append(str(r.path))
        seen[cell(r)].append(str(r.path))
        report["failed"][r.rows[-1].get("stopped") if r.rows else None] += 1
        for key, value in _flat(r.config).items():
            if key not in SWEPT:
                fixed[key].add(repr(value))

    report["duplicated"] = [(k, v) for k, v in seen.items() if len(v) > 1]
    report["inconsistent"] = {k: sorted(v) for k, v in fixed.items() if len(v) > 1}
    report["coverage"] = _coverage(runs)
    return report


def _flat(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, f"{key}."))
        else:
            out[key] = v
    return out


def _coverage(runs: list[Run]) -> dict:
    """Which learning rates exist for each (family, depth, method)."""
    grid: dict[tuple, set] = defaultdict(set)
    for r in runs:
        grid[(r.config["model"]["type"], r.depth, r.method)].add(r.lr)
    return {k: sorted(v) for k, v in grid.items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="olo.audit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pattern", nargs="?", default="*")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--coverage", action="store_true", help="print the per-cell lr coverage")
    args = p.parse_args(argv)

    rep = audit(args.runs_dir, args.pattern)
    print(f"{rep['n_dirs']} directories, {rep['n_runs']} with metrics\n")

    bad = 0
    for key, label in (("incomplete", "started but never finished"),
                       ("empty", "metrics file is empty"),
                       ("unscored", "no test.json")):
        items = rep[key]
        if items:
            bad += len(items) if key != "unscored" else 0
            print(f"{key.upper()} ({len(items)}) -- {label}")
            for s in items[:8]:
                print(f"   {s}")
            if len(items) > 8:
                print(f"   ... and {len(items) - 8} more")
            print()

    if rep["duplicated"]:
        bad += len(rep["duplicated"])
        print(f"DUPLICATED ({len(rep['duplicated'])}) -- one cell in several directories")
        for key, paths in rep["duplicated"][:8]:
            print(f"   {key}: {', '.join(paths)}")
        print()

    if rep["inconsistent"]:
        bad += len(rep["inconsistent"])
        print(f"INCONSISTENT ({len(rep['inconsistent'])}) -- held-fixed settings that differ")
        for k, vals in list(rep["inconsistent"].items())[:8]:
            print(f"   {k}: {', '.join(vals[:4])}")
        print()

    stops = {k: v for k, v in rep["failed"].items() if k}
    if stops:
        print("stop reasons: " + ", ".join(f"{k}={v}" for k, v in sorted(stops.items())))
    print(f"ran to completion: {rep['failed'].get(None, 0)}")

    if args.coverage:
        print("\nlearning rates per cell:")
        for (model, depth, method), lrs in sorted(rep["coverage"].items()):
            print(f"   {model:10s} L={depth:<4d} {method:10s} {len(lrs)} lrs: "
                  f"{', '.join(f'{v:g}' for v in lrs)}")

    print("\nOK" if not bad else f"\n{bad} issue(s) that would distort a comparison")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
