"""Linear-system model of a health-centre micro-grid.

Each day the energy drawn from ``n`` sources must satisfy ``n`` linear demand
constraints ``A @ s = d``. For the base Kasese system (solar ``x``, battery
``y``)::

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

References:
    Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM,
        lecture 12 (conditioning and condition numbers).
    Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.).
        Johns Hopkins University Press, sec. 2.6 and 3.1 (LU solves).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import scipy.linalg

from src.core.types import FloatArray


class SingularSystemError(np.linalg.LinAlgError):
    """The coefficient matrix is (numerically) singular: no unique dispatch exists."""


@dataclass(frozen=True, slots=True)
class Conditioning:
    """Well-posedness diagnostics for a square coefficient matrix.

    Attributes:
        determinant: ``det(A)``; zero means no unique solution.
        condition_number: 2-norm condition number ``||A|| * ||A^-1||``.
    """

    determinant: float
    condition_number: float

    #: Beyond this condition number we treat the matrix as singular (~1/eps).
    SINGULAR_THRESHOLD: ClassVar[float] = 1e12

    @property
    def is_singular(self) -> bool:
        return not np.isfinite(self.condition_number) or self.condition_number > self.SINGULAR_THRESHOLD

    @property
    def digits_at_risk(self) -> float:
        """Rough number of significant digits that input errors can corrupt (log10 cond)."""
        return float(np.log10(self.condition_number)) if np.isfinite(self.condition_number) else float("inf")

    def describe(self) -> str:
        """One-paragraph interpretation for reports."""
        if self.is_singular:
            return (
                f"det(A) = {self.determinant:.3g} and cond(A) = {self.condition_number:.3g}: the "
                "system is singular, so the constraints are linearly dependent and there is either "
                "no dispatch or infinitely many."
            )
        quality = "well-conditioned" if self.condition_number < 100 else "ill-conditioned"
        return (
            f"det(A) = {self.determinant:.3g} ≠ 0, so a unique dispatch exists for any demand. "
            f"cond(A) = {self.condition_number:.3g} ({quality}): a 1 % error in demand can move the "
            f"solution by at most ~{self.condition_number:.1f} % (about {self.digits_at_risk:.1f} "
            "significant digits at risk)."
        )


class MicroGrid:
    """A micro-grid whose daily dispatch solves a square linear system.

    Args:
        coefficients: Square matrix ``A`` (rows = constraints, columns = sources).
        sources: Name of each source, in column order.
        tariffs: Cost per kWh (UGX) for each source.

    Raises:
        ValueError: If shapes are inconsistent or a tariff is missing/negative.
    """

    DEFAULT_COEFFICIENTS: ClassVar[tuple[tuple[float, ...], ...]] = ((3.0, 2.0), (4.0, 1.0))
    DEFAULT_SOURCES: ClassVar[tuple[str, ...]] = ("solar", "battery")
    DEFAULT_TARIFFS: ClassVar[Mapping[str, float]] = {"solar": 150.0, "battery": 450.0}

    def __init__(
        self,
        coefficients: npt.ArrayLike | None = None,
        sources: Sequence[str] | None = None,
        tariffs: Mapping[str, float] | None = None,
    ) -> None:
        a = np.asarray(
            self.DEFAULT_COEFFICIENTS if coefficients is None else coefficients, dtype=np.float64
        )
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"coefficient matrix must be square, got shape {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError("coefficient matrix must be finite")
        names = tuple(self.DEFAULT_SOURCES if sources is None else sources)
        if len(names) != a.shape[1]:
            raise ValueError(f"expected {a.shape[1]} source names, got {len(names)}")
        prices = dict(self.DEFAULT_TARIFFS if tariffs is None else tariffs)
        missing = set(names) - prices.keys()
        if missing:
            raise ValueError(f"missing tariff for {sorted(missing)}")
        if any(prices[n] < 0 for n in names):
            raise ValueError("tariffs cannot be negative")
        a.setflags(write=False)
        self._a = a
        self._sources = names
        self._tariffs = np.array([prices[n] for n in names])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sources={self._sources}, A={self._a.tolist()})"

    # ---------------------------------------------------------- properties
    @property
    def coefficients(self) -> FloatArray:
        return self._a

    @property
    def sources(self) -> tuple[str, ...]:
        return self._sources

    @property
    def tariffs(self) -> FloatArray:
        """UGX per kWh for each source, in column order."""
        return self._tariffs

    @property
    def n_constraints(self) -> int:
        return int(self._a.shape[0])

    # -------------------------------------------------------- diagnostics
    def conditioning(self) -> Conditioning:
        """Determinant and condition number of ``A`` (never raises)."""
        return Conditioning(
            determinant=float(np.linalg.det(self._a)),
            condition_number=float(np.linalg.cond(self._a)),
        )

    def check_well_posed(self) -> Conditioning:
        """Return the diagnostics, raising if the system has no unique solution.

        Raises:
            SingularSystemError: If ``A`` is numerically singular.
        """
        diag = self.conditioning()
        if diag.is_singular:
            raise SingularSystemError(diag.describe())
        return diag

    # ------------------------------------------------------------ solving
    def solve_day(self, *demands: float) -> FloatArray:
        """Solve ``A @ s = d`` for one day's demands (one value per constraint).

        Raises:
            ValueError: On the wrong number of demands or a negative/non-finite value.
            SingularSystemError: If the system is singular.
        """
        d = self._validate_demands(np.asarray(demands, dtype=np.float64).reshape(-1, 1))
        return self._solve(d)[:, 0]

    def solve_days_loop(self, demands: npt.ArrayLike) -> FloatArray:
        """Solve each column of an ``(n_constraints, n_days)`` demand matrix in a Python loop."""
        d = self._validate_demands(np.asarray(demands, dtype=np.float64))
        self.check_well_posed()
        out = np.empty_like(d)
        for j in range(d.shape[1]):
            out[:, j] = scipy.linalg.solve(self._a, d[:, j])
        return out

    def solve_days(self, demands: npt.ArrayLike) -> FloatArray:
        """Solve all days in one vectorised call with a multi-column right-hand side."""
        d = self._validate_demands(np.asarray(demands, dtype=np.float64))
        return self._solve(d)

    def daily_cost(self, dispatch: npt.ArrayLike) -> FloatArray:
        """Cost in UGX per day for an ``(n_sources, n_days)`` dispatch matrix."""
        s = np.asarray(dispatch, dtype=np.float64)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        if s.shape[0] != len(self._sources):
            raise ValueError(f"dispatch must have {len(self._sources)} rows")
        return self._tariffs @ s

    # ------------------------------------------------------------ helpers
    def _solve(self, d: FloatArray) -> FloatArray:
        self.check_well_posed()
        return np.asarray(scipy.linalg.solve(self._a, d), dtype=np.float64)

    def _validate_demands(self, d: FloatArray) -> FloatArray:
        if d.ndim != 2 or d.shape[0] != self.n_constraints:
            raise ValueError(
                f"demands must have shape ({self.n_constraints}, n_days), got {d.shape}"
            )
        if d.shape[1] == 0:
            raise ValueError("no demands supplied")
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("demands must be finite and non-negative")
        return d


class HybridMicroGrid(MicroGrid):
    """Solar + battery + diesel generator with a third (night-time) constraint.

    The base two constraints gain a diesel column, and a night-time load row is
    added where solar cannot contribute (illustrative coefficients)::

        3x + 2y + 1z = D1   (daytime load)
        4x + 1y + 2z = D2   (critical-equipment load)
        0x + 3y + 2z = D3   (night-time load)

    Diesel is priced at an illustrative UGX 1,200/kWh.
    """

    DEFAULT_COEFFICIENTS: ClassVar[tuple[tuple[float, ...], ...]] = (
        (3.0, 2.0, 1.0),
        (4.0, 1.0, 2.0),
        (0.0, 3.0, 2.0),
    )
    DEFAULT_SOURCES: ClassVar[tuple[str, ...]] = ("solar", "battery", "diesel")
    DEFAULT_TARIFFS: ClassVar[Mapping[str, float]] = {
        "solar": 150.0,
        "battery": 450.0,
        "diesel": 1200.0,
    }

    @classmethod
    def with_dependent_constraint(cls) -> HybridMicroGrid:
        """A deliberately broken grid whose third row is the sum of the first two."""
        a = np.array(cls.DEFAULT_COEFFICIENTS[:2], dtype=np.float64)
        return cls(np.vstack([a, a[0] + a[1]]))
