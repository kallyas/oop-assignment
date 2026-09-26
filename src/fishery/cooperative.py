"""Composes stock dynamics and prices into revenue, risk and scenario analyses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from src.core.types import FloatArray
from src.fishery.price import PriceModel
from src.fishery.risk import RiskAssessor, RiskLevel, ValueAtRisk
from src.fishery.stock import WEEKS_PER_YEAR, FishStock, StockTrajectory

KG_PER_TONNE = 1_000.0


@dataclass(frozen=True, slots=True)
class RevenueSimulation:
    """One deterministic stock trajectory priced along one price path."""

    trajectory: StockTrajectory
    prices: FloatArray
    weekly_revenue: FloatArray

    @property
    def total_revenue(self) -> float:
        return float(self.weekly_revenue.sum())


class CooperativeModel:
    """A fish-export cooperative: a :class:`FishStock` sold at :class:`PriceModel` prices.

    The stock is deterministic, so for a fixed harvest rate all price paths
    share the same catch; revenue uncertainty comes from price alone.
    """

    def __init__(self, stock: FishStock, prices: PriceModel) -> None:
        self.stock = stock
        self.prices = prices

    def __repr__(self) -> str:
        return f"CooperativeModel({self.stock!r}, {self.prices!r})"

    def simulate(self, weeks: int = WEEKS_PER_YEAR) -> RevenueSimulation:
        """Weekly revenue (UGX) = catch (kg) × price (UGX/kg) on the seeded price path."""
        trajectory = self.stock.simulate(weeks)
        prices = self.prices.simulate(weeks)[0]
        return RevenueSimulation(trajectory, prices, trajectory.harvest * KG_PER_TONNE * prices)

    def simulate_total_revenue(
        self, weeks: int = WEEKS_PER_YEAR, n_paths: int = 1_000, rng: np.random.Generator | None = None
    ) -> FloatArray:
        """Monte Carlo distribution of total revenue over ``weeks`` (one value per price path)."""
        if n_paths < 1:
            raise ValueError("n_paths must be positive")
        harvest_kg = self.stock.simulate(weeks).harvest * KG_PER_TONNE
        paths = self.prices.simulate(weeks, n_paths=n_paths, rng=rng)
        return np.asarray(paths @ harvest_kg, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ScenarioSummary:
    """Headline results for one harvest rate."""

    harvest_rate: float
    simulation: RevenueSimulation
    risk: RiskAssessor
    var: ValueAtRisk
    annual_revenue_samples: FloatArray
    equilibrium_stock: float
    equilibrium_yield: float
    msy: float

    @property
    def final_stock(self) -> float:
        return self.simulation.trajectory.final_stock

    @property
    def total_revenue(self) -> float:
        return self.simulation.total_revenue

    @property
    def mean_weekly_catch(self) -> float:
        return float(self.simulation.trajectory.harvest.mean())

    @property
    def annual_revenue_cv(self) -> float:
        """CV of annual revenue across price paths: pure price risk, no stock transient."""
        return float(np.std(self.annual_revenue_samples, ddof=1) / np.mean(self.annual_revenue_samples))

    @property
    def risk_level(self) -> RiskLevel:
        return self.risk.classify()

    def row(self) -> dict[str, float | str]:
        """Table row (monetary values in UGX billions)."""
        return {
            "h": self.harvest_rate,
            "final stock (t)": self.final_stock,
            "equilibrium stock (t)": self.equilibrium_stock,
            "mean weekly catch (t)": self.mean_weekly_catch,
            "equilibrium yield (t/wk)": self.equilibrium_yield,
            "yield / MSY": self.equilibrium_yield / self.msy,
            "total revenue (UGX bn)": self.total_revenue / 1e9,
            "CV weekly revenue": self.risk.cv,
            "risk": self.risk_level.value,
            "CV annual revenue (MC)": self.annual_revenue_cv,
            "5% VaR (UGX bn)": self.var.var / 1e9,
        }


def run_scenarios(
    harvest_rates: Sequence[float],
    *,
    stock_factory: Callable[[float], FishStock] = lambda h: FishStock(harvest_rate=h),
    prices: PriceModel | None = None,
    weeks: int = WEEKS_PER_YEAR,
    n_paths: int = 1_000,
    seed: int = 7,
) -> list[ScenarioSummary]:
    """Simulate every harvest rate with common random numbers.

    Every scenario is priced on the *same* seeded price paths, so differences
    between scenarios reflect the harvest policy rather than sampling noise.
    """
    price_model = prices or PriceModel()
    summaries = []
    for h in harvest_rates:
        model = CooperativeModel(stock_factory(h), price_model)
        sim = model.simulate(weeks)
        samples = model.simulate_total_revenue(weeks, n_paths, rng=np.random.default_rng(seed))
        summaries.append(
            ScenarioSummary(
                harvest_rate=h,
                simulation=sim,
                risk=RiskAssessor(sim.weekly_revenue),
                var=RiskAssessor.value_at_risk(samples),
                annual_revenue_samples=samples,
                equilibrium_stock=model.stock.equilibrium_stock,
                equilibrium_yield=model.stock.equilibrium_yield,
                msy=model.stock.msy,
            )
        )
    return summaries
