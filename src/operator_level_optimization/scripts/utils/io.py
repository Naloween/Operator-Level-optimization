from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def get_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_out_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any, *, indent: int = 2, sort_keys: bool = False) -> None:
    path.write_text(json.dumps(obj, indent=indent, sort_keys=sort_keys))


def jsonify(obj: Any) -> Any:
    """Recursively convert to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)
