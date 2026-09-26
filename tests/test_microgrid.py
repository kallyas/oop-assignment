"""Tests for mini-project 2 (solar micro-grid dispatch)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from src.microgrid import (
    ClipPolicy,
    HybridMicroGrid,
    LeastCostPolicy,
    MicroGrid,
    NNLSPolicy,
    SingularSystemError,
    dispatch_with_policy,
    monte_carlo_sensitivity,
    prompt_non_negative_float,
    read_demand_csv,
    simulate_demand,
    write_demand_csv,
)


def test_solve_day_matches_hand_solution() -> None:
    # x = (2*D2 - D1)/5, y = (4*D1 - 3*D2)/5; D1=120, D2=105 -> x=18, y=33
    x, y = MicroGrid().solve_day(120.0, 105.0)
    assert x == pytest.approx(18.0)
    assert y == pytest.approx(33.0)


def test_conditioning_values() -> None:
    diag = MicroGrid().check_well_posed()
    assert diag.determinant == pytest.approx(-5.0)
    assert diag.condition_number == pytest.approx(np.linalg.cond([[3, 2], [4, 1]]))
    assert not diag.is_singular


def test_loop_and_vectorised_solutions_agree() -> None:
    grid, demands = MicroGrid(), simulate_demand(30).matrix
    np.testing.assert_allclose(grid.solve_days_loop(demands), grid.solve_days(demands))


def test_negative_or_misshaped_demand_rejected() -> None:
    with pytest.raises(ValueError):
        MicroGrid().solve_day(-1.0, 10.0)
    with pytest.raises(ValueError):
        MicroGrid().solve_day(1.0)


def test_dependent_constraint_is_singular() -> None:
    grid = HybridMicroGrid.with_dependent_constraint()
    assert grid.conditioning().is_singular
    with pytest.raises(SingularSystemError):
        grid.solve_day(10.0, 10.0, 20.0)


def test_infeasible_day_repaired_to_non_negative() -> None:
    grid = MicroGrid()
    demands = np.array([[120.0, 60.0], [105.0, 105.0]])  # day 2: D2/D1 > 4/3 -> y < 0
    for policy in (ClipPolicy(), NNLSPolicy()):
        plan = dispatch_with_policy(grid, demands, policy)
        assert plan.infeasible_days == [1]
        assert np.all(plan.dispatch >= 0)
        np.testing.assert_allclose(plan.dispatch[:, 0], [18.0, 33.0])
    nnls = dispatch_with_policy(grid, demands, NNLSPolicy())
    clip = dispatch_with_policy(grid, demands, ClipPolicy())
    assert np.linalg.norm(nnls.residual[:, 1]) <= np.linalg.norm(clip.residual[:, 1]) + 1e-9


def test_least_cost_policy_never_leaves_demand_unmet() -> None:
    grid = MicroGrid()
    demands = np.array([[78.6], [109.2]])  # a real infeasible Saturday
    plan = dispatch_with_policy(grid, demands, LeastCostPolicy())
    assert np.all(plan.unmet <= 1e-9)
    # Hand solution: battery off, solar sized for the critical load, x = D2 / 4.
    np.testing.assert_allclose(plan.dispatch[:, 0], [109.2 / 4, 0.0], atol=1e-9)
    # NNLS, by contrast, under-supplies the critical constraint on this day.
    assert dispatch_with_policy(grid, demands, NNLSPolicy()).unmet[1, 0] > 0


def test_daily_cost() -> None:
    assert MicroGrid().daily_cost([18.0, 33.0])[0] == pytest.approx(18 * 150 + 33 * 450)


def test_csv_round_trip(tmp_path: Path) -> None:
    schedule = simulate_demand(30, seed=1)
    path = tmp_path / "demand.csv"
    write_demand_csv(schedule, path)
    loaded = read_demand_csv(path)
    assert loaded.dates == schedule.dates
    np.testing.assert_allclose(loaded.matrix, schedule.matrix)


def test_csv_rejects_negative(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("date,weekday,d1_kwh,d2_kwh\n2026-10-01,Thu,-5,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        read_demand_csv(path)


def test_interactive_prompt_reprompts_until_valid() -> None:
    answers: Iterator[str] = iter(["", "abc", "-3", "nan", "42.5"])
    messages: list[str] = []
    value = prompt_non_negative_float("D1: ", input_fn=lambda _: next(answers), output_fn=messages.append)
    assert value == 42.5
    assert len(messages) == 4


def test_interactive_prompt_gives_up() -> None:
    with pytest.raises(RuntimeError):
        prompt_non_negative_float("D1: ", input_fn=lambda _: "x", output_fn=lambda _: None, max_attempts=3)


def test_sensitivity_amplification_bounded_by_condition_number() -> None:
    grid = MicroGrid()
    result = monte_carlo_sensitivity(grid, [120.0, 105.0], n_draws=500, rng=np.random.default_rng(3))
    assert result.amplification.max() <= grid.conditioning().condition_number + 1e-9
