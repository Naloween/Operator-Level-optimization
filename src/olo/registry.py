"""Name -> constructor resolution for the pluggable config sections.

Entries are `"module:attribute"` strings resolved lazily, so importing `olo.registry`
(which the CLI does immediately, to validate names before doing any work) does not drag in
torch, torchvision, or dataset downloads.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

from olo.config import Spec

MODELS: dict[str, str] = {
    "deep_linear": "olo.models.deep_linear:DeepLinear",
    "fgln": "olo.models.fgln:FGLN",
    "relu_mlp": "olo.models.relu_mlp:ReLUMLP",
    "crelu_mlp": "olo.models.crelu_mlp:CReLUMLP",
}

OPTIMS: dict[str, str] = {
    # ours -- the only operator-space method
    "als": "olo.optim.als:OperatorALS",
    # parameter-space comparators
    "heavyball": "olo.optim.baselines.torch_optims:HeavyBall",
    "adam": "olo.optim.baselines.torch_optims:Adam",
    "muon": "olo.optim.baselines.muon:Muon",
    "kfac": "olo.optim.baselines.kfac:KFAC",
    "shampoo": "olo.optim.baselines.shampoo:Shampoo",
    "soap": "olo.optim.baselines.soap:SOAP",
}

TASKS: dict[str, str] = {
    "teacher_student": "olo.tasks.teacher_student:TeacherStudent",
    "matrix_sensing": "olo.tasks.matrix_sensing:MatrixSensing",
    "mnist": "olo.tasks.vision:MNIST",
    "cifar10": "olo.tasks.vision:CIFAR10",
}

TARGETS: dict[str, str] = {
    "gradient": "olo.optim.targets:GradientTarget",
    "shadow": "olo.optim.targets:ShadowTarget",
    "fixed": "olo.optim.targets:FixedTarget",
}

INITS: tuple[str, ...] = ("xavier", "haar", "looks_linear", "identity")

_TABLES: dict[str, dict[str, str]] = {
    "model": MODELS,
    "optim": OPTIMS,
    "task": TASKS,
    "target": TARGETS,
}


def resolve(kind: str, name: str) -> Callable[..., Any]:
    """Look up a constructor, naming the alternatives when the lookup fails."""
    table = _TABLES.get(kind)
    if table is None:
        raise KeyError(f"unknown component kind {kind!r}; known: {', '.join(sorted(_TABLES))}")
    path = table.get(name)
    if path is None:
        raise KeyError(
            f"unknown {kind} {name!r}. Available: {', '.join(sorted(table))}"
        )
    module_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def build(kind: str, spec: Spec, **extra: Any) -> Any:
    """Instantiate the component described by `spec`, forwarding `extra` runtime args."""
    ctor = resolve(kind, spec.type)
    try:
        return ctor(**spec.params, **extra)
    except TypeError as exc:
        raise TypeError(
            f"could not build {kind} {spec.type!r} with params "
            f"{sorted(spec.params)}: {exc}"
        ) from exc
