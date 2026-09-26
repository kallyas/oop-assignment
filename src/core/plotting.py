"""A small, shared Matplotlib style so every notebook's figures read as one set.

Categorical colours come from a colour-vision-deficiency-validated palette and
are always assigned in the same fixed order (never cycled or re-ranked), so a
given entity keeps its colour across figures.
"""

from __future__ import annotations

from typing import Final

import matplotlib as mpl
from cycler import cycler

SERIES: Final[tuple[str, ...]] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
INK: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#8a8984"
GRID: Final = "#e4e3df"
SURFACE: Final = "#fcfcfb"
NEUTRAL_FILL: Final = "#f0efec"

# Status colours are reserved for state (good / warning / critical), never series.
STATUS_GOOD: Final = "#0ca30c"
STATUS_WARNING: Final = "#fab219"
STATUS_CRITICAL: Final = "#d03b3b"

# Single-hue sequential ramp (light -> dark) for magnitude encodings.
SEQUENTIAL_BLUE: Final[tuple[str, ...]] = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)


def series_color(index: int) -> str:
    """Colour for the ``index``-th categorical series.

    Raises:
        IndexError: Beyond eight series; fold extras into "Other" or facet instead.
    """
    if not 0 <= index < len(SERIES):
        raise IndexError(f"only {len(SERIES)} categorical colours are defined")
    return SERIES[index]


def apply_style() -> None:
    """Install the house style into Matplotlib's ``rcParams``."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "axes.prop_cycle": cycler(color=list(SERIES)),
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "text.color": INK,
            "font.size": 9.5,
        }
    )
