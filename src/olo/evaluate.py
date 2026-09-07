"""Score saved checkpoints on the test split, after training.

    python -m olo.evaluate runs/e02_depth_wall/seed0
    python -m olo.evaluate "runs/e02_*" --ckpt ckpt_final.pt --out test.json
    python -m olo.evaluate "runs/e02_*" --table

Kept out of the training loop on purpose. `runner.run` writes `test.json` once at the end,
and nothing during training ever reads the test split -- early stopping and learning-rate
selection both go through validation. This module exists so a test number can be recomputed
from a checkpoint (a different checkpoint, a changed metric, a task whose test split was
added after the run) without retraining and without giving the loop a way to see it.

A run is rebuilt from its own `config.yaml`, so the model, the initialization and the data
splits are exactly the ones it was trained with.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from olo.config import RunCfg


def rebuild(cfg: RunCfg, ckpt: str | Path | None = None):
    """(net, task) as the run had them, with weights loaded from `ckpt` if given."""
    from olo.registry import build
    from olo.runner import _build_model, _device, _dtype

    device, dtype = _device(cfg.device), _dtype(cfg.dtype)
    task = build("task", cfg.task).to(device, dtype)
    net = _build_model(cfg, task).to(device=device, dtype=dtype)
    if ckpt is not None:
        net.load_state_dict(torch.load(ckpt, map_location=device))
    return net, task


def evaluate_run(run_dir: str | Path, ckpt_name: str = "ckpt_final.pt") -> dict:
    """Validation and test metrics for one run directory."""
    run_dir = Path(run_dir)
    cfg = RunCfg.load(run_dir / "config.yaml")
    ckpt = run_dir / ckpt_name
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found; the run may not have finished")

    net, task = rebuild(cfg, ckpt)
    net.eval()
    with torch.no_grad():
        out = {f"val_{k}": v for k, v in task.evaluate(net).items()}
        out.update({f"test_{k}": v for k, v in task.test(net).items()})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="olo.evaluate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories, or globs over them")
    p.add_argument("--ckpt", default="ckpt_final.pt")
    p.add_argument("--out", default=None, help="write results into each run dir under this name")
    p.add_argument("--table", action="store_true", help="print one row per run")
    args = p.parse_args(argv)

    dirs: list[Path] = []
    for pattern in args.runs:
        path = Path(pattern)
        if (path / "config.yaml").exists():
            dirs.append(path)
        else:                                    # a glob over run names
            dirs += [c.parent for c in sorted(Path().glob(f"{pattern}/*/config.yaml"))]
            dirs += [c.parent for c in sorted(Path().glob(f"{pattern}/config.yaml"))]
    dirs = sorted(set(dirs))
    if not dirs:
        print("no runs matched")
        return 1

    rows = []
    for d in dirs:
        try:
            res = evaluate_run(d, args.ckpt)
        except Exception as exc:               # one broken run should not stop the rest
            print(f"{d}: FAILED {type(exc).__name__}: {exc}")
            continue
        rows.append((d, res))
        if args.out:
            (d / args.out).write_text(json.dumps(res, indent=2))
        if not args.table:
            print(f"{d}: " + "  ".join(f"{k}={v:.4g}" for k, v in res.items()
                                       if isinstance(v, float)))

    if args.table and rows:
        keys = [k for k in rows[0][1] if isinstance(rows[0][1][k], float)]
        print(f"{'run':52s} " + " ".join(f"{k:>18s}" for k in keys))
        for d, res in rows:
            print(f"{str(d)[-52:]:52s} " + " ".join(f"{res.get(k, float('nan')):18.6g}"
                                                    for k in keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
