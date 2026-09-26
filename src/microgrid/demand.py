"""Daily demand input: synthetic generation, CSV persistence and interactive entry."""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.types import FloatArray

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]

CSV_HEADER: tuple[str, ...] = ("date", "weekday", "d1_kwh", "d2_kwh")


@dataclass(frozen=True, slots=True)
class DemandSchedule:
    """Daily demands for the two base constraints.

    Attributes:
        dates: Calendar date of each day.
        d1: Daytime load (kWh) per day.
        d2: Critical-equipment load (kWh) per day.
    """

    dates: tuple[dt.date, ...]
    d1: FloatArray
    d2: FloatArray

    def __post_init__(self) -> None:
        if not (len(self.dates) == self.d1.size == self.d2.size):
            raise ValueError("dates, d1 and d2 must have equal lengths")
        if len(self.dates) == 0:
            raise ValueError("a schedule needs at least one day")
        if np.any(self.d1 < 0) or np.any(self.d2 < 0):
            raise ValueError("demands cannot be negative")

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def matrix(self) -> FloatArray:
        """Right-hand-side matrix of shape ``(2, n_days)``."""
        return np.vstack([self.d1, self.d2])


def simulate_demand(
    days: int = 30,
    *,
    seed: int = 2026,
    start: dt.date = dt.date(2026, 10, 1),
    weekday_d1: float = 120.0,
    weekend_d1: tuple[float, float] = (90.0, 74.0),
    d2_mean: float = 105.0,
    noise_sd: tuple[float, float] = (6.0, 5.0),
) -> DemandSchedule:
    """Generate realistic synthetic demand with a weekly pattern plus noise.

    Daytime load (D1) follows the outpatient clinic's week: high Monday-Friday,
    lower on Saturday and lowest on Sunday. Critical-equipment load (D2:
    fridges for vaccines, oxygen concentrators, theatre) is roughly constant.
    Gaussian noise is added and negatives are floored at zero.

    Raises:
        ValueError: If ``days`` is not positive.
    """
    if days <= 0:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    dates = tuple(start + dt.timedelta(days=i) for i in range(days))
    weekday = np.array([d.weekday() for d in dates])
    base_d1 = np.where(weekday == 5, weekend_d1[0], np.where(weekday == 6, weekend_d1[1], weekday_d1))
    d1 = np.maximum(0.0, base_d1 + rng.normal(0.0, noise_sd[0], days)).round(1)
    d2 = np.maximum(0.0, d2_mean + rng.normal(0.0, noise_sd[1], days)).round(1)
    return DemandSchedule(dates, d1, d2)


def write_demand_csv(schedule: DemandSchedule, path: Path) -> None:
    """Persist ``schedule`` as CSV with columns ``date, weekday, d1_kwh, d2_kwh``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for date, a, b in zip(schedule.dates, schedule.d1, schedule.d2):
            writer.writerow([date.isoformat(), date.strftime("%a"), f"{a:.1f}", f"{b:.1f}"])


def read_demand_csv(path: Path) -> DemandSchedule:
    """Load and validate a demand CSV written by :func:`write_demand_csv`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On a wrong header or an invalid row (the line number is reported).
    """
    dates: list[dt.date] = []
    d1: list[float] = []
    d2: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != CSV_HEADER:
            raise ValueError(f"expected header {CSV_HEADER}, got {reader.fieldnames}")
        for line_no, row in enumerate(reader, start=2):
            try:
                dates.append(dt.date.fromisoformat(row["date"]))
                a, b = float(row["d1_kwh"]), float(row["d2_kwh"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid row {row}") from exc
            if a < 0 or b < 0 or not np.isfinite([a, b]).all():
                raise ValueError(f"{path.name}:{line_no}: demand must be a non-negative number")
            d1.append(a)
            d2.append(b)
    return DemandSchedule(tuple(dates), np.array(d1), np.array(d2))


def prompt_non_negative_float(
    prompt: str,
    *,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
    max_attempts: int | None = None,
) -> float:
    """Ask until the user enters a finite, non-negative number.

    Empty, non-numeric, negative, ``nan`` and ``inf`` entries are rejected with a
    message and the question is asked again. ``input_fn``/``output_fn`` are
    injectable so the loop can be tested and demonstrated non-interactively.

    Raises:
        RuntimeError: If ``max_attempts`` invalid entries are made.
    """
    attempts = 0
    while max_attempts is None or attempts < max_attempts:
        attempts += 1
        text = input_fn(prompt).strip()
        if not text:
            output_fn("  ✗ Please enter a value (input was empty).")
            continue
        try:
            value = float(text)
        except ValueError:
            output_fn(f"  ✗ '{text}' is not a number.")
            continue
        if not np.isfinite(value):
            output_fn("  ✗ The value must be a finite number.")
        elif value < 0:
            output_fn("  ✗ Demand cannot be negative.")
        else:
            return value
    raise RuntimeError(f"no valid value after {max_attempts} attempts")


def read_day_interactively(
    *, input_fn: InputFn = input, output_fn: OutputFn = print
) -> tuple[float, float]:
    """Prompt for one day's ``(D1, D2)``, re-asking until each value is valid."""
    d1 = prompt_non_negative_float("Daytime load D1 (kWh): ", input_fn=input_fn, output_fn=output_fn)
    d2 = prompt_non_negative_float(
        "Critical-equipment load D2 (kWh): ", input_fn=input_fn, output_fn=output_fn
    )
    return d1, d2
