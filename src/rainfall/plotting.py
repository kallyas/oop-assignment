"""Figures for the rainfall and crop-suitability analyser."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.core.plotting import INK, SURFACE, series_color
from src.rainfall.crops import CropRule, Suitability
from src.rainfall.region import MONTHS, Region
from src.rainfall.seasons import SeasonProfile

# Diverging encoding: drought (warm) <- good (neutral-green) -> waterlogging (cool).
_SUITABILITY_COLORS = {
    Suitability.DROUGHT: "#eb6834",
    Suitability.GOOD: "#b9e3c6",
    Suitability.WATERLOGGING: "#2a78d6",
}


def plot_rainfall(regions: Sequence[Region], profiles: Sequence[SeasonProfile]) -> Figure:
    """All regions on one line chart, with detected season peaks marked."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    x = np.arange(12)
    for i, (region, profile) in enumerate(zip(regions, profiles)):
        color = series_color(i)
        ax.plot(x, region.rainfall, color=color, marker="o", ms=4,
                label=f"{region.name} ({region.annual_total():,.0f} mm/yr, {profile.modality.value.lower()})")
        for m in profile.peak_months:
            ax.plot(m, region.rainfall[m], marker="v", ms=10, color=color, markeredgecolor=SURFACE, zorder=5)
    ax.set_xticks(x, MONTHS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean monthly rainfall (mm)")
    ax.set_ylim(bottom=0)
    ax.set_title("Monthly rainfall regimes (▼ = detected rainy-season peak)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_suitability_heatmap(regions: Sequence[Region], rules: Sequence[CropRule]) -> Figure:
    """One heatmap panel per crop: rows = regions, columns = months, cells = rainfall class."""
    order = [Suitability.DROUGHT, Suitability.GOOD, Suitability.WATERLOGGING]
    cmap = ListedColormap([_SUITABILITY_COLORS[s] for s in order])
    fig, axes = plt.subplots(len(rules), 1, figsize=(10, 0.75 * len(regions) * len(rules) + 1.6), sharex=True)
    axes_list = list(np.atleast_1d(axes))
    for ax, rule in zip(axes_list, rules):
        codes = np.vstack([rule.codes(r) for r in regions])
        ax.imshow(codes, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for (i, j), code in np.ndenumerate(codes):
            ink = INK if code == Suitability.GOOD.value else SURFACE  # keep text legible on saturated cells
            ax.text(j, i, f"{regions[i].rainfall[j]:.0f}", ha="center", va="center", fontsize=7.5, color=ink)
        ax.set_yticks(range(len(regions)), [r.name for r in regions])
        ax.set_title(f"{rule.crop}: good band {rule.min_mm:.0f}–{rule.max_mm:.0f} mm/month", loc="left", fontsize=10)
        ax.set_xticks(np.arange(-0.5, 12), minor=True)
        ax.set_yticks(np.arange(-0.5, len(regions)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
    axes_list[-1].set_xticks(range(12), MONTHS)
    handles = [Patch(color=_SUITABILITY_COLORS[s], label=s.label("crop").replace("crop", "the crop")) for s in order]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Crop suitability by month and region (cell text = rainfall, mm)", fontweight="semibold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
