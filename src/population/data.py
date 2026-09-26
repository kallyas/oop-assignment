"""Illustrative district population data (thousands, 2015-2024).

Kampala, Wakiso and Gulu are the figures given in the assignment brief.
Mbarara and Arua are *illustrative* additions chosen to widen the range of
growth regimes: a secondary city adding a roughly constant number of people
each year (Mbarara, i.e. linear rather than exponential growth) and a district
whose growth accelerated with the 2016-2018 refugee influx from South Sudan
(Arua). They are not official UBOS statistics.
"""

from __future__ import annotations

from src.population.district import DistrictPopulation

YEARS: tuple[int, ...] = tuple(range(2015, 2025))

RAW_SERIES: dict[str, tuple[float, ...]] = {
    "Kampala": (1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800),
    "Wakiso": (950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670),
    "Gulu": (320, 330, 345, 360, 375, 390, 410, 430, 455, 480),
    "Mbarara": (470, 486, 499, 517, 531, 548, 562, 579, 594, 610),
    "Arua": (780, 805, 860, 930, 975, 1005, 1030, 1055, 1080, 1105),
}


def load_districts() -> list[DistrictPopulation]:
    """Return one :class:`DistrictPopulation` per district in ``RAW_SERIES``."""
    return [DistrictPopulation(name, YEARS, pops) for name, pops in RAW_SERIES.items()]
