"""Fleet sizing from a passenger forecast."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FleetPlan:
    """Vehicles needed to carry a forecast number of passengers.

    Attributes:
        route: Route name.
        forecast_passengers: Expected passengers for the day.
        seats: Seats per vehicle (14 for a matatu).
        trips_per_vehicle: One-way trips each vehicle makes per day.
        buffer: Safety margin as a fraction (0.15 = 15 %).
        min_vehicles: Floor so every operating route has at least one vehicle.
    """

    route: str
    forecast_passengers: float
    seats: int = 14
    trips_per_vehicle: int = 8
    buffer: float = 0.15
    min_vehicles: int = 1

    def __post_init__(self) -> None:
        if self.forecast_passengers < 0:
            raise ValueError("forecast passengers cannot be negative")
        if self.seats <= 0 or self.trips_per_vehicle <= 0:
            raise ValueError("seats and trips per vehicle must be positive")
        if self.buffer < 0:
            raise ValueError("buffer cannot be negative")

    @property
    def capacity_per_vehicle(self) -> int:
        """Passengers one vehicle can carry per day (seats × trips)."""
        return self.seats * self.trips_per_vehicle

    @property
    def planned_passengers(self) -> float:
        return self.forecast_passengers * (1.0 + self.buffer)

    @property
    def exact_vehicles(self) -> float:
        return self.planned_passengers / self.capacity_per_vehicle

    @property
    def vehicles(self) -> int:
        """Rounded *up*: rounding down would strand passengers the buffer was meant for."""
        return max(self.min_vehicles, math.ceil(self.exact_vehicles - 1e-9))

    @property
    def utilisation(self) -> float:
        """Expected load factor of the deployed fleet (unbuffered forecast / capacity)."""
        return self.forecast_passengers / (self.vehicles * self.capacity_per_vehicle)
