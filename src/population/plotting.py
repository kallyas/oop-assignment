"""Figures for the population forecaster."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.core.forecasting import Forecaster
from src.core.plotting import INK, INK_MUTED, NEUTRAL_FILL, series_color
from src.population.district import DistrictPopulation
from src.population.evaluation import Forecast, ValidationReport


def plot_forecasts(
    reports: Sequence[ValidationReport],
    forecasts: Mapping[str, Forecast],
    full_models: Mapping[str, Forecaster],
    model_order: Sequence[str],
    *,
    ncols: int = 2,
) -> Figure:
    """One subplot per district: actual, hold-out fits, final forecast and interval.

    Args:
        reports: Validation reports (one per district), for the test-period fits.
        forecasts: Final forecast per district name.
        full_models: Selected model refitted on the full series, per district.
        model_order: All model names, fixing each model's colour across panels.
        ncols: Subplot columns.
    """
    colors = {name: series_color(i) for i, name in enumerate(model_order)}
    nrows = int(np.ceil(len(reports) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows), squeeze=False)

    for ax, report in zip(axes.flat, reports):
        district: DistrictPopulation = report.district
        fc = forecasts[district.name]
        split_year = float(report.train.years[-1]) + 0.5

        ax.axvspan(split_year, float(report.test.years[-1]) + 0.5, color=NEUTRAL_FILL, zorder=0)
        ax.axvline(split_year, color=INK_MUTED, lw=1, ls="--")
        ax.text(split_year, 1.0, " test →", transform=ax.get_xaxis_transform(),
                va="top", ha="left", fontsize=8, color=INK_MUTED)

        for score in report.scores:
            ax.plot(report.test.years, score.test_predictions, color=colors[score.model_name],
                    lw=1.3, ls=":", marker="o", ms=3.5, label=f"{score.model_name} (test fit)")

        model = full_models[district.name]
        fitted = model.fitted_values()
        ax.plot(district.years, fitted, color=colors[fc.model_name], lw=1.2, alpha=0.6,
                label=f"{fc.model_name} (in-sample fit)")

        if fc.lower is not None and fc.upper is not None and fc.level is not None:
            ax.fill_between(fc.years, fc.lower, fc.upper, color=colors[fc.model_name],
                            alpha=0.18, lw=0, label=f"{fc.level:.0%} bootstrap PI")
        ax.plot(np.r_[district.years[-1], fc.years], np.r_[district.populations[-1], fc.point],
                color=colors[fc.model_name], lw=2.2, label=f"Forecast: {fc.model_name}")
        ax.plot(district.years, district.populations, color=INK, lw=0, marker="o", ms=5,
                label="Actual", zorder=5)

        ax.set_title(f"{district.name} (best model: {fc.model_name})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Population (thousands)")
        ax.set_xticks(np.arange(district.years[0], fc.years[-1] + 1, 2))
        ax.annotate(f"{fc.final_value:,.0f}k", (float(fc.years[-1]), fc.final_value),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
        # Keep the explosive Fibonacci test fit from flattening the panel.
        top = max(float(np.max(district.populations)), float(np.max(fc.upper if fc.upper is not None else fc.point)))
        ax.set_ylim(bottom=float(np.min(district.populations)) * 0.9, top=top * 1.12)

    for ax in list(axes.flat)[len(reports):]:
        ax.set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    for ax in list(axes.flat)[1 : len(reports)]:
        h, lab = ax.get_legend_handles_labels()
        by_label.update(dict(zip(lab, h)))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("District population: validation fits (2022–24) and 2025–29 forecasts",
                 fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig
