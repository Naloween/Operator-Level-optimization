"""Run configuration: typed schema, YAML (de)serialization, override/sweep expansion.

A config is a single YAML file that fully describes one run. Sections that name a
pluggable component (`model`, `task`, `optim`, and `optim.target`) are `Spec`s: a `type`
string resolved through `olo.registry` plus arbitrary keyword parameters forwarded to the
constructor. That keeps this module ignorant of every component's signature -- adding a
model does not mean touching the config schema.

Everything the runner itself needs (seeds, device, dtype, budgets, diagnostics) is typed
here, because those fields must mean the same thing for every component: a fair
comparison between optimizers depends on them being shared, not re-declared per method.
"""
from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

import yaml


class Loader(yaml.SafeLoader):
    """SafeLoader that parses `1e-3` and `1.0e8` as numbers.

    YAML 1.1 -- which PyYAML implements -- requires a decimal point *and* a signed
    exponent, so `1e-3`, `1.0e8` and `1e6` all load as strings. In a config file full of
    learning rates and thresholds that is a trap with no error message: the run proceeds
    with a string where a float belongs and either crashes somewhere unrelated or, worse,
    compares as a string and silently does the wrong thing. This resolver adopts the
    YAML 1.2 / JSON reading instead.
    """


Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |\.[0-9][0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_yaml(source) -> Any:
    """Parse YAML with the numeric fix above. Use this everywhere, never `safe_load`."""
    return yaml.load(source, Loader=Loader)


# ---------------------------------------------------------------------------
# Component specs
# ---------------------------------------------------------------------------


@dataclass
class Spec:
    """A component: a registry `type` plus the kwargs its constructor takes."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any, *, where: str) -> "Spec":
        if isinstance(raw, str):                       # bare string shorthand
            return cls(type=raw, params={})
        if not isinstance(raw, dict):
            raise ConfigError(f"{where}: expected a mapping or a string, got {type(raw).__name__}")
        d = dict(raw)
        if "type" not in d:
            raise ConfigError(f"{where}: missing required key 'type'")
        t = d.pop("type")
        if not isinstance(t, str):
            raise ConfigError(f"{where}.type: expected a string, got {t!r}")
        return cls(type=t, params=d)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.params}


# ---------------------------------------------------------------------------
# Typed sections
# ---------------------------------------------------------------------------


@dataclass
class TrainCfg:
    """Budget and stopping rules -- shared by every method, which is the point.

    The three stopping rules cover the three ways a cell in a sweep stops being worth
    compute: it has converged (`stop_below`), it has blown up (`stop_above`, plus a
    non-finite check that is always on), or it has stalled (`stop_patience` evaluations
    without a relative improvement of `stop_min_delta`). The last one is what catches a
    collapsed network sitting at its initialization loss for the whole budget -- a real
    result, but one that does not need 400 steps to establish.
    """

    steps: int = 1000
    batch_size: int | None = None          # None = full batch
    eval_every: int = 10
    ckpt_every: int = 0                    # 0 = only final
    stop_below: float | None = None        # converged
    stop_above: float | None = None        # diverged
    stop_patience: int | None = None       # evaluations without improvement before stopping
    stop_min_delta: float = 1e-3           # relative improvement that counts as progress


@dataclass
class DiagnosticsCfg:
    """What to measure, how often, and on how much data.

    `batch_size` is the subsample the context-based diagnostics run on, and it is a memory
    bound, not a nicety. A layer's right context has shape (B, width, d_in), so the full
    stack costs B * d_in * width * L: on MNIST at batch 128 that is 3.4 GB at depth 64 and
    7.3 GB at depth 256, which is enough to take down the machine before it takes down the
    run. The quantities being measured are spectra and alignments of those contexts, which
    a handful of samples characterizes as well as a hundred.
    """

    every: int = 10                        # in steps; diagnostics are the expensive part
    batch_size: int = 8                    # subsample for context-based diagnostics
    mismatch: bool = True                  # cos(realized dP, -eta G)
    conditioning: bool = True              # per-layer rho_k collapse monitor
    spectrum: bool = True                  # singular values of P / J(x)
    cost: bool = True                      # wall-clock and peak memory
    spectrum_samples: int = 4              # per-sample operators to spectrally profile


@dataclass
class RunCfg:
    name: str
    model: Spec
    task: Spec
    optim: Spec
    seed: int = 0
    device: str = "auto"                   # auto | cpu | cuda
    dtype: str = "float32"                 # float32 | float64
    train: TrainCfg = field(default_factory=TrainCfg)
    diagnostics: DiagnosticsCfg = field(default_factory=DiagnosticsCfg)
    out_dir: str = "runs"
    notes: str = ""

    # -- (de)serialization --------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunCfg":
        d = copy.deepcopy(raw)
        missing = [k for k in ("name", "model", "task", "optim") if k not in d]
        if missing:
            raise ConfigError(f"config is missing required key(s): {', '.join(missing)}")

        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ConfigError(
                f"unknown top-level config key(s): {', '.join(sorted(unknown))}. "
                f"Known keys: {', '.join(sorted(known))}"
            )

        return cls(
            name=d["name"],
            model=Spec.parse(d["model"], where="model"),
            task=Spec.parse(d["task"], where="task"),
            optim=Spec.parse(d["optim"], where="optim"),
            seed=int(d.get("seed", 0)),
            device=d.get("device", "auto"),
            dtype=d.get("dtype", "float32"),
            train=_section(TrainCfg, d.get("train", {}), "train"),
            diagnostics=_section(DiagnosticsCfg, d.get("diagnostics", {}), "diagnostics"),
            out_dir=d.get("out_dir", "runs"),
            notes=d.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "device": self.device,
            "dtype": self.dtype,
            "model": self.model.to_dict(),
            "task": self.task.to_dict(),
            "optim": self.optim.to_dict(),
            "train": asdict(self.train),
            "diagnostics": asdict(self.diagnostics),
            "out_dir": self.out_dir,
            "notes": self.notes,
        }

    @classmethod
    def load(cls, path: str | Path) -> "RunCfg":
        with open(path) as fh:
            raw = load_yaml(fh)
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: top level must be a mapping")
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @property
    def run_dir(self) -> Path:
        """Where this run's artifacts go: `<out_dir>/<name>/seed<k>`."""
        return Path(self.out_dir) / self.name / f"seed{self.seed}"


class ConfigError(ValueError):
    """Raised for malformed configs, with the offending key named."""


def _section(cls_, raw: Any, where: str):
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    unknown = set(raw) - set(cls_.__dataclass_fields__)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(sorted(unknown))}. "
            f"Known: {', '.join(sorted(cls_.__dataclass_fields__))}"
        )
    return cls_(**raw)


# ---------------------------------------------------------------------------
# Dotted-path overrides and sweeps
# ---------------------------------------------------------------------------


def set_path(raw: dict[str, Any], path: str, value: Any) -> None:
    """Set `raw["a"]["b"] = value` from the dotted path "a.b", creating dicts as needed."""
    keys = path.split(".")
    node = raw
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = value


def parse_scalar(text: str) -> Any:
    """Parse a CLI value with YAML rules, so `true`, `null`, `[1,2]` all work.

    With one correction: YAML 1.1 does not recognize `1e-3` as a number (it wants
    `1.0e-3`), and would hand back the *string* "1e-3". A learning rate that is silently a
    string surfaces much later as a confusing type error, or worse, as a sweep axis whose
    values never took effect -- so numeric-looking strings are converted here.
    """
    value = load_yaml(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def expand(raw: dict[str, Any], overrides: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Cartesian product of `overrides` applied to `raw`.

    Every combination gets a name suffix naming the varied axes, so a sweep's runs land in
    distinct directories and stay identifiable without consulting the config.
    """
    if not overrides:
        yield copy.deepcopy(raw)
        return

    keys = list(overrides)
    for combo in itertools.product(*(overrides[k] for k in keys)):
        out = copy.deepcopy(raw)
        parts = []
        for k, v in zip(keys, combo):
            set_path(out, k, v)
            if len(overrides[k]) > 1:                # only varied axes enter the name
                parts.append(f"{k.split('.')[-1]}{_slug(v)}")
        if parts:
            out["name"] = f"{out.get('name', 'run')}__{'_'.join(parts)}"
        yield out


def _slug(v: Any) -> str:
    s = str(v)
    for a, b in ((".", "p"), ("-", "m"), ("+", ""), (" ", "")):
        s = s.replace(a, b)
    return s
