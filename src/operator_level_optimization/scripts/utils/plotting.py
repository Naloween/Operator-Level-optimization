from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_fig(
    fig: Any,
    path: Path | str,
    *,
    dpi: int = 160,
    tight_layout: bool = True,
    **savefig_kwargs: Any,
) -> None:
    """Tight-layout, save, and close a figure in one call."""
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, **savefig_kwargs)
    plt.close(fig)


def clip_for_plot(values: list[float], y_max: float) -> tuple[np.ndarray, bool]:
    """Clip finite values at ``y_max`` for display; NaNs are left unchanged.

    Returns ``(y_display, clipped)`` where ``clipped`` is True if any value exceeded
    ``y_max``.  Set ``y_max <= 0`` to disable clipping.
    """
    a = np.asarray(values, dtype=float)
    if y_max <= 0 or a.size == 0:
        return a, False
    finite = np.isfinite(a)
    over = finite & (a > y_max)
    if not np.any(over):
        return a, False
    out = a.copy()
    out[over] = y_max
    return out, True


def annotate_clip(ax: Any, clipped: bool, y_max: float) -> None:
    if clipped and y_max > 0:
        ax.text(
            0.01,
            0.99,
            f"values > {y_max:g} clipped for display",
            transform=ax.transAxes,
            fontsize=7,
            color="0.35",
            va="top",
            ha="left",
        )
