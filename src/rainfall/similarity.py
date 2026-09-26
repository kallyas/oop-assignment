"""Vector similarity and distance measures between rainfall regimes.

Last year's brief used ``math.cos`` to compare regions. ``math.cos(x)`` is
the trigonometric cosine of an *angle* ``x``; it says nothing about two
vectors. Cosine *similarity* is the cosine of the angle *between* two vectors,
``a·b / (‖a‖ ‖b‖)``, which is what is implemented here.

References:
    Manning, C. D., Raghavan, P. & Schütze, H. (2008). *Introduction to
        Information Retrieval*. Cambridge University Press, sec. 6.3.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import distance

from src.core.types import FloatArray, as_float_array
from src.rainfall.region import Region

PairwiseMeasure = Callable[[FloatArray, FloatArray], float]


def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Cosine of the angle between two vectors, in ``[-1, 1]``.

    Raises:
        ValueError: If the vectors differ in length or either has zero norm.
    """
    u, v = as_float_array(a, name="a"), as_float_array(b, name="b")
    if u.shape != v.shape:
        raise ValueError(f"vectors differ in length ({u.size} vs {v.size})")
    norms = float(np.linalg.norm(u) * np.linalg.norm(v))
    if norms == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(u, v) / norms)


def scipy_cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Reference implementation: ``1 - scipy.spatial.distance.cosine``."""
    return 1.0 - float(distance.cosine(as_float_array(a), as_float_array(b)))


def pearson_correlation(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Pearson correlation: cosine similarity of the *mean-centred* vectors."""
    u, v = as_float_array(a, name="a"), as_float_array(b, name="b")
    return cosine_similarity(u - u.mean(), v - v.mean())


def euclidean_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Straight-line distance in mm; sensitive to both shape and total amount."""
    u, v = as_float_array(a, name="a"), as_float_array(b, name="b")
    if u.shape != v.shape:
        raise ValueError(f"vectors differ in length ({u.size} vs {v.size})")
    return float(np.linalg.norm(u - v))


@dataclass(frozen=True, slots=True)
class SimilarityMatrix:
    """Symmetric pairwise matrix of one measure over a set of regions."""

    measure: str
    labels: tuple[str, ...]
    values: FloatArray

    def __getitem__(self, pair: tuple[str, str]) -> float:
        i, j = (self.labels.index(p) for p in pair)
        return float(self.values[i, j])

    def as_rows(self) -> list[dict[str, float | str]]:
        """Rows keyed by region name (convenient for ``pandas.DataFrame``)."""
        return [
            {"region": r, **{c: float(self.values[i, j]) for j, c in enumerate(self.labels)}}
            for i, r in enumerate(self.labels)
        ]


def pairwise_matrix(regions: Sequence[Region], measure: PairwiseMeasure, name: str) -> SimilarityMatrix:
    """Evaluate ``measure`` on every ordered pair of regions."""
    n = len(regions)
    values = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            values[i, j] = measure(regions[i].rainfall, regions[j].rainfall)
    return SimilarityMatrix(name, tuple(r.name for r in regions), values)


def all_matrices(regions: Sequence[Region]) -> dict[str, SimilarityMatrix]:
    """Cosine similarity, Pearson correlation and Euclidean distance matrices."""
    measures: dict[str, PairwiseMeasure] = {
        "Cosine similarity": cosine_similarity,
        "Pearson correlation": pearson_correlation,
        "Euclidean distance (mm)": euclidean_distance,
    }
    return {name: pairwise_matrix(regions, fn, name) for name, fn in measures.items()}
