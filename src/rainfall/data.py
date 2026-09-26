"""Illustrative mean monthly rainfall (mm, Jan-Dec) from the assignment brief."""

from __future__ import annotations

from src.rainfall.region import Region

RAW_RAINFALL: dict[str, tuple[float, ...]] = {
    "Kampala": (120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130),
    "Gulu": (8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15),
    "Mbarara": (70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90),
}

#: Climate regime expected from the literature (e.g. Basalirwa, 1995, Int. J.
#: Climatology 15:1161-1177; UNMA seasonal outlooks): the Lake Victoria basin and
#: south-west have two rainy seasons (MAM and SON); the north has a single long
#: season (Apr-Oct) that is often interrupted by a short June dip.
EXPECTED_MODALITY: dict[str, str] = {
    "Kampala": "Bimodal",
    "Gulu": "Unimodal",
    "Mbarara": "Bimodal",
}


def load_regions() -> list[Region]:
    return [Region(name, rain) for name, rain in RAW_RAINFALL.items()]
