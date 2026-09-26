"""Figures for the micro-grid dispatch planner."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from src.core.plotting import INK, INK_SECONDARY, STATUS_CRITICAL, SURFACE, series_color
from src.microgrid.analysis import SensitivityResult
from src.microgrid.demand import DemandSchedule
from src.microgrid.feasibility import DispatchPlan


def plot_dispatch(plan: DispatchPlan, schedule: DemandSchedule) -> Figure:
    """Stacked daily energy by source, with daily cost on a secondary axis.

    The brief requires the second axis. To keep it readable the cost is drawn
    as a thin dark line with its own labelled axis in the same ink, and
    infeasible (repaired) days are marked above the bars.
    """
    days = np.arange(len(schedule))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    bottom = np.zeros(days.size)
    for i, source in enumerate(plan.grid.sources):
        ax.bar(days, plan.dispatch[i], bottom=bottom, width=0.78, color=series_color(i),
               edgecolor=SURFACE, linewidth=1.0, label=f"{source.capitalize()} (kWh)")
        bottom += plan.dispatch[i]
    for j in plan.infeasible_days:
        ax.annotate("✕", (j, bottom[j]), xytext=(0, 3), textcoords="offset points",
                    ha="center", color=STATUS_CRITICAL, fontsize=10, fontweight="bold")
    ax.set_ylabel("Energy dispatched (kWh)")
    ax.set_xticks(days, [f"{d:%a}\n{d.day}" for d in schedule.dates], fontsize=7.5)
    ax.set_xlim(-0.6, days.size - 0.4)
    ax.set_ylim(0, bottom.max() * 1.15)

    cost_ax = ax.twinx()
    cost_ax.plot(days, plan.daily_cost / 1_000, color=INK, lw=1.4, marker="o", ms=3.5,
                 label="Daily cost (UGX '000, right axis)")
    cost_ax.set_ylabel("Daily cost (UGX thousands)", color=INK)
    cost_ax.spines["right"].set_visible(True)
    cost_ax.grid(False)
    cost_ax.set_ylim(0, (plan.daily_cost / 1_000).max() * 1.15)
    cost_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    handles = ax.get_legend_handles_labels()[0] + cost_ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + cost_ax.get_legend_handles_labels()[1]
    handles.append(Line2D([], [], color=STATUS_CRITICAL, marker="$✕$", lw=0, ms=8))
    labels.append(f"Infeasible day, repaired by {plan.policy_name.upper()}")
    ax.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.14))
    ax.set_title(f"Kasese health-centre micro-grid: daily dispatch and cost "
                 f"(month total UGX {plan.total_cost:,.0f})")
    fig.tight_layout()
    return fig


def plot_sensitivity(result: SensitivityResult, sources: tuple[str, ...], cond: float) -> Figure:
    """Histograms of relative source changes and of the error amplification factor."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    rel = result.relative_source_change * 100
    for i, name in enumerate(sources):
        axes[0].hist(rel[:, i], bins=40, histtype="step", linewidth=1.8, color=series_color(i),
                     label=name.capitalize())
    axes[0].set_xlabel("Change in source output vs base day (%)")
    axes[0].set_ylabel("Number of draws")
    axes[0].set_title("Dispatch response to ±5 % demand noise")
    axes[0].legend()

    axes[1].hist(result.amplification, bins=40, color=series_color(0), alpha=0.8)
    axes[1].axvline(cond, color=INK, ls="--", lw=1.2)
    axes[1].text(cond, 0.95, f" cond(A) = {cond:.2f}\n (upper bound)", transform=axes[1].get_xaxis_transform(),
                 va="top", ha="right", fontsize=8, color=INK_SECONDARY)
    axes[1].set_xlabel("Relative error amplification  ‖Δs‖/‖s‖ ÷ ‖Δd‖/‖d‖")
    axes[1].set_ylabel("Number of draws")
    axes[1].set_title("Observed amplification never exceeds cond(A)")
    fig.tight_layout()
    return fig
