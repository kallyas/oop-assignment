"""Daily passenger counts (10 days) from the assignment brief."""

from __future__ import annotations

from src.taxi.route import Route

ROUTE_DATA: dict[str, tuple[tuple[int, ...], float]] = {
    "Kampala-Ntinda": ((35, 40, 42, 50, 55, 60, 48, 52, 47, 45), 2_000.0),
    "Kampala-Entebbe": ((60, 58, 65, 70, 72, 80, 75, 68, 66, 64), 5_000.0),
    "Kampala-Mukono": ((45, 47, 50, 49, 55, 62, 58, 53, 51, 50), 3_000.0),
}


def load_routes() -> list[Route]:
    return [Route(name, counts, fare) for name, (counts, fare) in ROUTE_DATA.items()]
