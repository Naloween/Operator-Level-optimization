"""Figure style and the fixed method->color assignment.

Color follows the method, never its rank in a plot. A figure showing three optimizers and
one showing seven must give Adam the same color in both, or a reader comparing figures is
silently misled. So the mapping is a table, assigned once, and no plotting function is
allowed to cycle a color list.

The hues are the reference categorical palette in its documented order (slots 1-7), used
unchanged -- that order is validated for colorblind separation on the adjacent pairlist.
Every method also carries a distinct line style and marker: a paper figure gets printed in
grayscale and read by colorblind reviewers, so identity is never carried by color alone.
"""
from __future__ import annotations

METHOD_COLORS: dict[str, str] = {
    "als": "#2a78d6",         # slot 1 -- the method under test
    "adam": "#eb6834",        # slot 2
    "heavyball": "#1baf7a",   # slot 3
    "muon": "#eda100",        # slot 4
    "kfac": "#e87ba4",        # slot 5
    "shampoo": "#008300",     # slot 6
    "soap": "#4a3aa7",        # slot 7
}

METHOD_STYLES: dict[str, tuple[str, str]] = {
    "als": ("-", "o"),
    "adam": ("--", "s"),
    "heavyball": ("-.", "^"),
    "muon": (":", "v"),
    "kfac": ("--", "D"),
    "shampoo": ("-.", "P"),
    "soap": (":", "X"),
}

#: single-hue ramp for an ordered axis (depth, learning rate). Starts at step 250, the
#: lightest step that still clears 2:1 on a light surface.
SEQUENTIAL_BLUE = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]

MODEL_LABELS = {
    "deep_linear": "deep linear",
    "fgln": "FGLN",
    "relu_mlp": "ReLU MLP",
    "crelu_mlp": "CReLU MLP",
}

METHOD_LABELS = {
    "als": "ALS-exact", "adam": "Adam", "heavyball": "Heavy Ball", "muon": "Muon",
    "kfac": "K-FAC", "shampoo": "Shampoo", "soap": "SOAP",
}


def color(method: str) -> str:
    """Color for a method, by name. Unknown methods get a neutral gray, not a new hue."""
    return METHOD_COLORS.get(method, "#8a8a85")


def style(method: str) -> tuple[str, str]:
    return METHOD_STYLES.get(method, ("-", "."))


def label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def ramp(n: int) -> list[str]:
    """`n` steps of the sequential hue, for an ordered (not categorical) axis."""
    if n <= 1:
        return [SEQUENTIAL_BLUE[-3]]
    idx = [round(i * (len(SEQUENTIAL_BLUE) - 1) / (n - 1)) for i in range(n)]
    return [SEQUENTIAL_BLUE[i] for i in idx]


def apply_style(dark: bool = False) -> None:
    """Recessive grid and axes, thin marks, text in ink rather than in series color."""
    import matplotlib as mpl

    ink = "#ffffff" if dark else "#22221f"
    muted = "#8a8a85"
    surface = "#1a1a19" if dark else "#ffffff"
    mpl.rcParams.update({
        "figure.facecolor": surface,
        "axes.facecolor": surface,
        "savefig.facecolor": surface,
        "axes.edgecolor": muted,
        "axes.labelcolor": ink,
        "axes.titlecolor": ink,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": muted,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "text.color": ink,
        "xtick.color": muted,
        "ytick.color": muted,
        "xtick.labelcolor": ink,
        "ytick.labelcolor": ink,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
    })
