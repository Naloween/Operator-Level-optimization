"""Extend a learning-rate sweep until every optimum is interior to the grid.

    python -m olo.extend_lr "e02_relu_L*" --metric eval_val_accuracy
    python -m olo.extend_lr "e02_*" --floor 1e-8 --max-rounds 4 --dry-run

A method whose best learning rate is the smallest (or largest) one tried has its optimum
*outside* the grid, so its score is a lower bound rather than a tuned result. Reporting
such a cell as a method's performance at that depth is how a grid edge gets mistaken for
a depth wall -- and this is exactly the regime where that mistake is easy to make, since
`olo.theory.deep_linear` puts the usable step size at O(1/L): a range that brackets the
optimum at depth 2 can sit entirely above it at depth 256.

So the criterion is not "did this cell do badly" but "is its optimum inside the grid". The
loop extends by one decade in the offending direction, re-runs only the cells that need
it, and stops when no cell is at an edge, when a floor/ceiling is reached, or after
`max_rounds`. Cells that are already interior are never re-run.

Each new run is rebuilt from an existing run's own `config.yaml`, so the model,
initialization, task, budget and diagnostics are identical to the sweep it extends, and
only the learning rate differs.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from olo.viz.loading import Run, load_runs

FLOOR = 1e-8
CEILING = 1e2


def cell_key(run: Run) -> tuple:
    """What identifies a column of the grid: one model family at one depth."""
    model = run.config["model"]
    return (model["type"], model.get("init", "xavier"), run.depth)


def edge_cells(
    runs: list[Run], metric: str, mode: str, direction: str = "down"
) -> list[dict]:
    """Cells whose best learning rate sits at an edge of the range tried."""
    groups: dict[tuple, list[Run]] = {}
    for r in runs:
        groups.setdefault(cell_key(r) + (r.method,), []).append(r)

    out = []
    for key, group in sorted(groups.items()):
        lrs = sorted({r.lr for r in group})
        if len(lrs) < 2:
            continue
        pick = min if mode == "min" else max
        best = pick(group, key=lambda r: r.value(metric, mode))
        at_low, at_high = best.lr == lrs[0], best.lr == lrs[-1]
        if (direction in ("down", "both") and at_low) or \
           (direction in ("up", "both") and at_high):
            out.append({
                "model": key[0], "init": key[1], "depth": key[2], "method": key[3],
                "best_lr": best.lr, "lrs": lrs, "score": best.value(metric, mode),
                "low": at_low, "template": best,
            })
    return out


def next_lr(cell: dict) -> float | None:
    """One decade past the offending edge, or None once the floor/ceiling is reached."""
    if cell["low"]:
        nxt = cell["lrs"][0] / 10.0
        return nxt if nxt >= FLOOR else None
    nxt = cell["lrs"][-1] * 10.0
    return nxt if nxt <= CEILING else None


def launch(cell: dict, lr: float, out_dir: str, dry_run: bool) -> bool:
    """Re-run one cell at `lr`, reusing its own config so only the rate changes."""
    template: Run = cell["template"]
    # Derive the name from the template's own, replacing only its learning-rate suffix.
    # Rebuilding it from the base instead would drop every other axis the sweep varied --
    # depth above all -- so two different cells would be handed the same name and the
    # second would be refused as a config clash.
    stem = re.sub(r"_lr[^_/]*$", "", template.config["name"])
    if stem == template.config["name"]:                  # no lr in the name to replace
        stem = f"{stem}__depth{cell['depth']}_type{cell['method']}"
    name = f"{stem}_lr{_slug(lr)}"
    cmd = [
        sys.executable, "-m", "olo.run", str(template.path / "config.yaml"), "--quiet",
        "--set", f"name={name}",
        "--set", f"optim.type={cell['method']}",
        "--set", f"optim.lr={lr}",
        "--set", f"out_dir={out_dir}",
    ]
    print(f"  {cell['model']:10s} L={cell['depth']:<4d} {cell['method']:10s} "
          f"lr {cell['best_lr']:g} (edge) -> {lr:g}")
    if dry_run:
        return True
    return subprocess.run(cmd).returncode == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="olo.extend_lr", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pattern", help="glob over run directory names, e.g. 'e02_relu_L*'")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--metric", default="eval_val_accuracy")
    p.add_argument("--mode", default=None, choices=["min", "max"])
    p.add_argument("--direction", default="down", choices=["down", "up", "both"])
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    from olo.viz.plots import _infer_mode
    mode = args.mode or _infer_mode(args.metric)

    for rnd in range(1, args.max_rounds + 1):
        runs = load_runs(args.runs_dir, args.pattern)
        if not runs:
            print(f"no runs match {args.pattern!r} under {args.runs_dir}")
            return 1
        cells = edge_cells(runs, args.metric, mode, args.direction)
        pending = [(c, next_lr(c)) for c in cells]
        pending = [(c, lr) for c, lr in pending if lr is not None]

        exhausted = len(cells) - len(pending)
        print(f"\nround {rnd}: {len(runs)} runs, {len(cells)} cell(s) with an optimum at a "
              f"grid edge, {len(pending)} extendable"
              + (f", {exhausted} already at the floor/ceiling" if exhausted else ""))
        if not pending:
            print("every optimum is interior to its grid (or the range is exhausted)")
            return 0

        ok = all(launch(c, lr, args.runs_dir, args.dry_run) for c, lr in pending)
        if args.dry_run:
            return 0
        if not ok:
            print("some extensions failed", file=sys.stderr)
            return 1

    print(f"\nstopped after {args.max_rounds} rounds; re-run to continue extending")
    return 0


def _slug(v: float) -> str:
    return f"{v:g}".replace(".", "p").replace("-", "m").replace("+", "")


if __name__ == "__main__":
    raise SystemExit(main())
