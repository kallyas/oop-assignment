"""Figures for the fish-stock and export-risk model."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.core.plotting import INK, INK_MUTED, STATUS_CRITICAL, series_color
from src.fishery.cooperative import ScenarioSummary


def plot_stock_trajectories(summaries: Sequence[ScenarioSummary], carrying_capacity: float) -> Figure:
    """Biomass over time for each harvest rate, with K and the MSY stock level."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, s in enumerate(summaries):
        biomass = s.simulation.trajectory.biomass
        ax.plot(np.arange(biomass.size), biomass, color=series_color(i), label=f"h = {s.harvest_rate:.2f}")
        ax.annotate(f"h={s.harvest_rate:.2f}: {biomass[-1]:,.0f} t", (biomass.size - 1, biomass[-1]),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.axhline(carrying_capacity, color=INK_MUTED, ls="--", lw=1)
    ax.text(0, carrying_capacity, "carrying capacity K", va="bottom", fontsize=8, color=INK_MUTED)
    ax.axhline(carrying_capacity / 2, color=INK_MUTED, ls=":", lw=1)
    ax.text(26, carrying_capacity / 2 + 120, "K/2: stock that gives MSY", va="bottom", fontsize=8, color=INK_MUTED)
    ax.set_xlabel("Week")
    ax.set_ylabel("Fish stock (tonnes)")
    ax.set_ylim(0, carrying_capacity * 1.08)
    ax.set_xlim(0, summaries[0].simulation.trajectory.biomass.size + 12)
    ax.set_title("Lake Victoria stock under four harvest rates (logistic model, 52 weeks)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_revenue_distribution(summary: ScenarioSummary) -> Figure:
    """Histogram of simulated annual revenue with the 5 % VaR marked."""
    samples = summary.annual_revenue_samples / 1e9
    var = summary.var
    fig, ax = plt.subplots(figsize=(9, 4.2))
    counts, _, _ = ax.hist(samples, bins=45, color=series_color(0), alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, float(np.max(counts)) * 1.5)  # headroom so annotations never overlap bars
    ax.axvline(var.expected / 1e9, color=INK, lw=1.4)
    ax.axvline(var.quantile / 1e9, color=STATUS_CRITICAL, lw=1.6, ls="--")
    ax.annotate("", xy=(var.quantile / 1e9, 0.92), xytext=(var.expected / 1e9, 0.92),
                xycoords=("data", "axes fraction"),
                arrowprops={"arrowstyle": "<->", "color": STATUS_CRITICAL, "lw": 1.2})
    ax.text((var.quantile + var.expected) / 2e9, 0.94, f"5% VaR = UGX {var.var / 1e9:,.1f} bn",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(var.expected / 1e9, 0.8, f"  mean\n  {var.expected / 1e9:,.1f} bn",
            transform=ax.get_xaxis_transform(), va="top", fontsize=8, color=INK)
    ax.text(var.quantile / 1e9, 0.8, f"  5th percentile\n  {var.quantile / 1e9:,.1f} bn",
            transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, color=STATUS_CRITICAL)
    ax.set_xlabel("Annual revenue (UGX billions)")
    ax.set_ylabel(f"Number of simulated years (of {samples.size:,})")
    ax.set_title(f"Monte Carlo annual revenue at h = {summary.harvest_rate:.2f} "
                 f"({samples.size:,} price paths)")
    fig.tight_layout()
    return fig
