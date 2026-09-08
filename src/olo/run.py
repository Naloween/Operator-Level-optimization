"""CLI entry point: run a config, or a sweep expanded from one.

    python -m olo.run configs/experiments/e01.yaml
    python -m olo.run configs/experiments/e01.yaml --set optim.lr=0.1
    python -m olo.run configs/experiments/e01.yaml --sweep optim.lr=1e-1,1e-2,1e-3 --seeds 0,1,2
    python -m olo.run configs/experiments/e01.yaml --sweep optim.type=als,adam,muon --dry-run

Sweeping is the intended way to compare methods, because it makes the comparison fair by
construction: one config fixes the model, task, budget and diagnostics, and only the swept
axis differs. A learning-rate grid applied to every method at once is a `--sweep`, not a
protocol anyone has to remember.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from olo.config import RunCfg, expand, load_yaml, parse_scalar, set_path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    with open(args.config) as fh:
        raw = load_yaml(fh)
    for item in args.set or []:
        key, _, val = item.partition("=")
        set_path(raw, key, parse_scalar(val))

    overrides = {}
    for item in args.sweep or []:
        key, _, vals = item.partition("=")
        overrides[key] = [parse_scalar(v) for v in vals.split(",")]
    if args.seeds:
        overrides["seed"] = [int(s) for s in args.seeds.split(",")]

    configs = [RunCfg.from_dict(d) for d in expand(raw, overrides)]
    print(f"{len(configs)} run(s) from {args.config}")
    for cfg in configs:
        print(f"  {cfg.name} seed={cfg.seed} -> {cfg.run_dir}")
    if args.dry_run:
        return 0

    from olo.runner import run                       # deferred: pulls in torch

    failures = []
    for i, cfg in enumerate(configs, 1):
        if _is_complete(cfg.run_dir) and not args.overwrite:
            clash = _config_clash(cfg)
            if clash:
                # Two different configurations mapping to one directory means the run name
                # does not capture everything being varied -- typically a value set with
                # --set (which does not enter the name) inside a loop. Skipping silently
                # would fill the sweep with whichever configuration ran first while
                # reporting success, so refuse instead.
                print(
                    f"[{i}/{len(configs)}] REFUSED {cfg.name}: an existing run at "
                    f"{cfg.run_dir} has a different config ({clash}). Give the runs "
                    f"distinct names (--set name=...), or pass --overwrite.",
                    file=sys.stderr)
                failures.append((cfg.name, ValueError(f"config clash: {clash}")))
                continue
            print(f"[{i}/{len(configs)}] skip {cfg.name} (exists; --overwrite to redo)")
            continue
        print(f"[{i}/{len(configs)}] {cfg.name} seed={cfg.seed}")
        try:
            run(cfg, progress=not args.quiet)
        except Exception as exc:
            # One diverged or misconfigured cell must not abandon the rest of a sweep --
            # a failed configuration is itself a result worth having the others beside.
            failures.append((cfg.name, exc))
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.traceback:
                traceback.print_exc()

    if failures:
        print(f"\n{len(failures)} run(s) failed:", file=sys.stderr)
        for name, exc in failures:
            print(f"  {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


def _is_complete(run_dir: Path) -> bool:
    """Whether a run finished, rather than merely having been started.

    `config.yaml` is written before training, so a run interrupted partway leaves a
    directory that looks done. Resuming would then skip it forever and the cell would be
    missing from the sweep -- indistinguishable, in any later analysis, from a
    configuration that was never requested. `metrics.jsonl` is written only after the
    training loop returns, so it is the marker of a finished run.
    """
    return (run_dir / "metrics.jsonl").exists()


def _config_clash(cfg: RunCfg) -> str | None:
    """Describe how an existing run at this path differs from `cfg`, or None if it matches.

    Only the fields that define the experiment are compared; `out_dir` and free-text notes
    are not part of a run's identity.
    """
    path = cfg.run_dir / "config.yaml"
    if not path.exists():
        return None
    try:
        old = RunCfg.load(path).to_dict()
    except Exception:
        return None                          # unreadable: let the normal skip apply
    new = cfg.to_dict()
    ignore = {"out_dir", "notes"}
    diffs = [k for k in set(old) | set(new)
             if k not in ignore and old.get(k) != new.get(k)]
    if not diffs:
        return None
    return ", ".join(
        f"{k}: {old.get(k)!r} != {new.get(k)!r}" for k in sorted(diffs)[:3]
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="olo.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override one config value (dotted path)")
    p.add_argument("--sweep", action="append", metavar="KEY=V1,V2",
                   help="expand a config axis into multiple runs")
    p.add_argument("--seeds", metavar="0,1,2", help="shorthand for --sweep seed=...")
    p.add_argument("--overwrite", action="store_true", help="redo runs whose output exists")
    p.add_argument("--dry-run", action="store_true", help="list the runs and exit")
    p.add_argument("--quiet", action="store_true", help="no per-step output")
    p.add_argument("--traceback", action="store_true", help="full traceback on failure")
    return p


if __name__ == "__main__":
    raise SystemExit(main())
