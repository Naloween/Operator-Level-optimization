"""One config in, one run directory out.

The runner is deliberately method-agnostic: it builds whatever the config names, then
drives the identical loop for all of them -- same batches, same step budget, same
diagnostic schedule, same early-stopping rule. Fairness between an operator-space method
and a parameter-space one is then a property of the harness rather than something the
experimenter has to remember to arrange, which is exactly the objection the previous round
of this work could not answer.

Outputs, all under `runs/<name>/seed<k>/`:

    config.yaml     the fully resolved config, so a run reproduces itself
    meta.json       git commit, device, versions, resolved component descriptions
    metrics.jsonl   one row per logged step
    arrays.npz      spectra and per-layer profiles, keyed by step
    ckpt_*.pt       weights
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from olo.config import RunCfg, Spec
from olo.diagnostics.conditioning import context_conditioning
from olo.diagnostics.cost import StepTimer
from olo.diagnostics.linearization import linearization_validity
from olo.diagnostics.mismatch import gd_first_order_alignment, step_alignment
from olo.diagnostics.spectrum import operator_spectrum
from olo.registry import build

DTYPES = {"float32": torch.float32, "float64": torch.float64}

#: Breakdowns that are results, not defects: a Kronecker factor going singular, a
#: covariance filling with non-finite values after the weights have already diverged.
#: Caught per step so the cell is recorded as failed instead of disappearing.
NUMERICAL_FAILURES = (torch._C._LinAlgError,)


def run(cfg: RunCfg, progress: bool = True) -> Path:
    torch.manual_seed(cfg.seed)
    device = _device(cfg.device)
    dtype = _dtype(cfg.dtype)

    task = build("task", cfg.task).to(device, dtype)
    net = _build_model(cfg, task).to(device=device, dtype=dtype)
    opt = _build_optimizer(cfg, net, task)

    out = cfg.run_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.yaml")
    _write_meta(out, cfg, net, task, opt, device)

    timer = StepTimer(device)
    timer.reset_peak_memory()
    stopper = _Stopper(cfg)
    rows: list[dict] = []
    arrays: dict[str, list] = {}
    lam = float(getattr(opt, "lam", 0.0))       # the rho_k monitor is relative to it

    for step in range(cfg.train.steps):
        x, y = task.train_batch(step, cfg.train.batch_size)
        want_diag = _due(step, cfg.diagnostics.every, cfg.train.steps)

        # Diagnostics run on a subsample: the context stack they build is the largest
        # allocation in the whole loop and grows with depth (see DiagnosticsCfg).
        nd = min(cfg.diagnostics.batch_size, x.shape[0])
        xd, yd = (x[:nd], y[:nd]) if want_diag else (x, y)

        pre = _pre_step_snapshot(net, xd, yd, task, cfg, want_diag)
        try:
            with timer.time():
                metrics = opt.step(x, y, task)
        except NUMERICAL_FAILURES as exc:
            # A preconditioner whose factors have gone singular or non-finite has broken
            # down, which at depth is an experimental outcome rather than a bug. Letting
            # it propagate would abort the run and leave no directory at all, and a cell
            # missing from a sweep is indistinguishable from one never requested -- so the
            # comparison would quietly drop whichever method failed. Record and stop.
            rows.append({"step": step, "loss": float("nan"),
                         "stopped": "numerical_failure", "error": f"{type(exc).__name__}: {exc}"})
            if progress:
                print(f"  numerical failure at step {step}: {type(exc).__name__}", flush=True)
            break

        if want_diag or _due(step, cfg.train.eval_every, cfg.train.steps):
            row = {"step": step, **metrics}
            if want_diag:
                scal, arr = _diagnostics(net, xd, cfg, pre, opt, lam)
                row.update(scal)
                for k, v in arr.items():
                    arrays.setdefault(k, []).append((step, v))
                pre.clear()                  # release the context stack before the next step
            if _due(step, cfg.train.eval_every, cfg.train.steps):
                row.update({f"eval_{k}": v for k, v in task.evaluate(net).items()})
            if cfg.diagnostics.cost:
                row.update(timer.stats())
            rows.append(row)
            if progress:
                _print(row)

            stop = stopper(row.get("eval_primary"))
            if stop:
                rows[-1]["stopped"] = stop
                if progress:
                    print(f"  stopped at step {step}: {stop}", flush=True)
                break

        if cfg.train.ckpt_every and _due(step, cfg.train.ckpt_every, cfg.train.steps):
            torch.save(net.state_dict(), out / f"ckpt_{step:06d}.pt")

    torch.save(net.state_dict(), out / "ckpt_final.pt")
    _write_metrics(out, rows, arrays)

    # Test metrics, computed once, after training, and written to their own file. Nothing
    # in the loop above has seen this split, so it is a measurement rather than a
    # selection. `olo.evaluate` recomputes it from any checkpoint.
    test = task.test(net)
    if test:
        (out / "test.json").write_text(json.dumps(
            {"step": rows[-1]["step"] if rows else 0,
             "stopped": rows[-1].get("stopped") if rows else None,
             **{k: _jsonable(v) for k, v in test.items()}}, indent=2))
        if progress:
            print("  test: " + "  ".join(f"{k}={v:.4g}" for k, v in test.items()
                                          if isinstance(v, float)), flush=True)
    return out


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def _build_model(cfg: RunCfg, task):
    """Build the network, taking its input/output widths from the task by default."""
    params = dict(cfg.model.params)
    scheme = params.pop("init", "xavier")
    gain = params.pop("init_gain", "auto")
    params.setdefault("d_in", task.d_in)
    params.setdefault("d_out", task.d_out)

    net = build("model", Spec(cfg.model.type, params))
    net.initialize(scheme, cfg.seed, gain)
    net.init_scheme = scheme
    return net


def _build_optimizer(cfg: RunCfg, net, task):
    """Build the optimizer, folding in the block of params named after its type.

    A config carries per-method blocks, and each may set its own learning rate:

        optim:
          type: als
          als:  {lr: 1.0, lam: 1.0e-4, n_sweeps: 1}
          adam: {lr: 1.0e-3}
          kfac: {lr: 1.0e-3, damping: 1.0e-2}

    Learning rates are *not* comparable across these methods and should not share a
    default. ALS's `lr` scales the operator target `P - lr G`, so it is a step in operator
    space and is depth-independent by construction; Adam's is a step in parameter space,
    whose usable range shrinks with depth (`olo.theory.deep_linear`: the induced operator
    step goes as `L * eta`). Giving both the same number compares step-size conventions,
    not methods.

    Precedence: a block value is the method's *default*; a value written at the top level
    of `optim` overrides every block. That is what makes a tuning sweep work --
    `--sweep optim.lr=...` sets one explicit rate for whichever method is selected, so a
    method x lr grid reads each method at its own optimum, while a run with no sweep gets
    sensible per-method defaults.

    Blocks for methods that were not selected are dropped rather than forwarded, which is
    what lets `--sweep optim.type=als,adam,kfac` run from a single file at all.
    """
    from olo.registry import OPTIMS

    shared = {k: v for k, v in cfg.optim.params.items() if k not in OPTIMS}
    block = dict(cfg.optim.params.get(cfg.optim.type, {}))
    params = {**block, **shared}                     # explicit shared values win
    opt = build("optim", Spec(cfg.optim.type, params), net=net)
    # An operator-recovery target needs the task's P*, which the config cannot name.
    target = getattr(opt, "target", None)
    if target is not None and getattr(target, "P_star", "missing") is None:
        if task.P_star is None:
            raise ValueError(
                f"optim.target is 'fixed' but task {cfg.task.type!r} defines no target "
                "operator P*. Use a teacher-student or matrix-sensing task, or a "
                "'gradient'/'shadow' target."
            )
        target.set_target(task.P_star)
    return opt


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def _pre_step_snapshot(net, x, y, task, cfg: RunCfg, want_diag: bool) -> dict:
    """State captured before the step, so the step's effect can be attributed.

    Deliberately outside the timed region: the extra forward/backward this costs is the
    price of the measurement, not of the method, and charging it to the optimizer would
    make whichever method is diagnosed most heavily look slowest.
    """
    if not want_diag:
        return {}
    d = cfg.diagnostics
    gates = net.gates(x)
    G = task.operator_gradient(net, x, y) if d.mismatch else None
    with torch.no_grad():
        snap = {"gates": gates, "P": net.operator(x, gates).clone()}
        if d.mismatch:
            snap["contexts"] = net.all_contexts(x, gates)
            snap["weights"] = [W.detach().clone() for W in net.weights]
            snap["G"] = G
    return snap


def _diagnostics(net, x, cfg: RunCfg, pre: dict, opt, lam: float):
    d = cfg.diagnostics
    scalars: dict[str, float] = {}
    arrays: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        P_now = net.operator(x, pre["gates"])

        if d.mismatch and "contexts" in pre:
            G = pre.get("G")
            if G is not None:
                dW = [W - w0 for W, w0 in zip(net.weights, pre["weights"])]
                scalars.update(
                    step_alignment(pre["P"], P_now, G, opt.lr, pre["contexts"], dW)
                )
                scalars.update(gd_first_order_alignment(pre["contexts"], G))

        if d.conditioning and "contexts" in pre:
            s, a = context_conditioning(pre["contexts"], lam)
            scalars.update(s)
            arrays.update(a)

        if d.spectrum:
            s, a = operator_spectrum(net.operator(x), d.spectrum_samples)
            scalars.update(s)
            arrays.update(a)

        scalars.update(linearization_validity(net, x, pre["gates"]))
    return scalars, arrays


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def _write_metrics(out: Path, rows: list[dict], arrays: dict[str, list]) -> None:
    with open(out / "metrics.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps({k: _jsonable(v) for k, v in r.items()}) + "\n")
    if arrays:
        payload = {}
        for name, entries in arrays.items():
            payload[f"{name}_steps"] = np.array([s for s, _ in entries])
            payload[name] = np.stack([v.detach().cpu().numpy() for _, v in entries])
        np.savez_compressed(out / "arrays.npz", **payload)


def _write_meta(out: Path, cfg: RunCfg, net, task, opt, device) -> None:
    meta = {
        "git_commit": _git_commit(),
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "model": net.describe(),
        "init": getattr(net, "init_scheme", None),
        "task": task.describe(),
        "optim": opt.describe() if hasattr(opt, "describe") else {"type": cfg.optim.type},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _jsonable(v):
    if isinstance(v, torch.Tensor):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _due(step: int, every: int, total: int) -> bool:
    return bool(every) and (step % every == 0 or step == total - 1)


class _Stopper:
    """Decides when a run has stopped being worth compute, and says why.

    Three exits, each meaning something different in a sweep summary: `converged`,
    `diverged`, and `stalled`. Keeping them distinct matters -- a cell that stalled at its
    initialization loss and a cell that blew up are both "bad", but only the second is a
    step-size artifact, and collapsing them into one label would hide exactly the
    distinction the depth experiments are about.
    """

    def __init__(self, cfg: RunCfg) -> None:
        self.cfg = cfg.train
        self.best = np.inf
        self.since_improved = 0

    def __call__(self, primary) -> str | None:
        if primary is None:
            return None
        if not np.isfinite(primary):
            return "diverged"
        if self.cfg.stop_below is not None and primary < self.cfg.stop_below:
            return "converged"
        if self.cfg.stop_above is not None and primary > self.cfg.stop_above:
            return "diverged"

        if primary < self.best * (1.0 - self.cfg.stop_min_delta):
            self.best, self.since_improved = primary, 0
        else:
            self.best = min(self.best, primary)
            self.since_improved += 1
            if (self.cfg.stop_patience is not None
                    and self.since_improved >= self.cfg.stop_patience):
                return "stalled"
        return None


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {name!r}")
    return DTYPES[name]


def _print(row: dict) -> None:
    keys = ("step", "loss", "eval_primary", "cos_target", "cos_gd_first_order", "rho_min")
    parts = [f"{k}={row[k]:.4g}" if isinstance(row.get(k), float) else f"{k}={row[k]}"
             for k in keys if k in row]
    print("  " + "  ".join(parts), flush=True)
