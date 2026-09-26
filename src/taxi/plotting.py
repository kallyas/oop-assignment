"""Figures for the taxi route planner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.core.plotting import INK, INK_MUTED, series_color
from src.core.types import FloatArray
from src.taxi.backtest import RouteBacktest


def plot_backtests(
    backtests: Sequence[RouteBacktest],
    next_day_revenue: Mapping[str, float],
    model_order: Sequence[str],
) -> Figure:
    """Actual daily revenue, walk-forward forecasts per model, and the day-11 forecast."""
    colors = {name: series_color(i) for i, name in enumerate(model_order)}
    fig, axes = plt.subplots(1, len(backtests), figsize=(5.2 * len(backtests), 4.2), sharey=False)
    for ax, bt in zip(np.atleast_1d(axes), backtests):
        route = bt.route
        days = np.arange(1, len(route) + 1)
        ax.plot(days, route.daily_revenue() / 1_000, color=INK, marker="o", ms=5, lw=1.6, label="Actual", zorder=5)
        fc_days = days[bt.first_origin :]
        for name in model_order:
            lw = 2.0 if name == bt.best_model else 1.1
            ax.plot(fc_days, bt.forecasts[name] * route.fare / 1_000, color=colors[name], lw=lw,
                    ls="-" if name == bt.best_model else "--", label=name)
        nxt = next_day_revenue[route.name] / 1_000
        ax.plot([days[-1], days[-1] + 1], [route.daily_revenue()[-1] / 1_000, nxt],
                color=colors[bt.best_model], lw=2, ls=":")
        ax.plot(days[-1] + 1, nxt, marker="*", ms=14, color=colors[bt.best_model], zorder=6)
        ax.annotate(f"Day 11\nUGX {nxt:,.0f}k", (days[-1] + 1, nxt), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.axvline(bt.first_origin + 0.5, color=INK_MUTED, lw=1, ls=":")
        ax.set_xticks(np.arange(1, len(route) + 2))
        ax.set_xlabel("Day")
        ax.set_ylabel("Daily revenue (UGX thousands)")
        ax.set_title(f"{route.name} (best: {bt.best_model})")
        ax.margins(y=0.2)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Walk-forward one-step forecasts (days 4–10) and day-11 revenue forecast",
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def plot_seasonal_comparison(values: FloatArray, forecasts: Mapping[str, FloatArray], first_origin: int) -> Figure:
    """Simulated 60-day series with walk-forward forecasts from competing models."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    days = np.arange(1, values.size + 1)
    ax.plot(days, values, color=INK, lw=1.2, marker="o", ms=3, label="Simulated passengers")
    for i, (name, fc) in enumerate(forecasts.items()):
        ax.plot(days[first_origin:], fc, color=series_color(i), lw=1.6, label=name)
    for d in days[(days - 1) % 7 == 4]:
        ax.axvline(d, color=INK_MUTED, lw=0.6, alpha=0.4)
    ax.set_xlabel("Day (faint lines = Fridays)")
    ax.set_ylabel("Passengers per day")
    ax.set_title("Weekly pattern: seasonal-naïve vs moving-average forecasts")
    ax.legend(ncol=len(forecasts) + 1, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return fig
