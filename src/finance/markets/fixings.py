"""Historical index fixings represented as a flat date/value line."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.math.interpolation import Flat
from common.math.line import Extrapolation, Line1d

DateArray = np.ndarray
FloatArray = np.ndarray


@dataclass(frozen=True)
class HistoricalFixings:
    """Immutable historical rate observations, held flat between prints.

    A fixing holds until the next one: Friday's print covers Saturday and Sunday, Monday's
    takes over on Monday.  Queries before the first print return it; queries after the last
    return the last.

    The object deliberately carries no valuation-date or projection behavior. ``YieldCurve``
    owns the history and ``RateGenerator`` decides whether a requested fixing is historical
    by comparing its date with the associated ``ZeroCurve.origin``.
    """

    dates: DateArray
    values: FloatArray
    _line: Line1d = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        dates = np.asarray(self.dates).astype("datetime64[D]").copy()
        values = np.asarray(self.values, dtype=np.float64).copy()
        if dates.ndim != 1 or dates.size == 0:
            raise ValueError("historical fixing dates must be a non-empty 1-D array.")
        if values.shape != dates.shape:
            raise ValueError("historical fixing dates and values must have the same shape.")
        if np.isnat(dates).any():
            raise ValueError("historical fixing dates cannot contain NaT.")
        if np.any(dates[1:] <= dates[:-1]):
            raise ValueError("historical fixing dates must be strictly increasing.")
        if not np.isfinite(values).all():
            raise ValueError("historical fixing values must be finite.")

        dates.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "_line",
            Line1d(
                x=dates.view(np.int64).astype(np.float64),
                y=values,
                interpolator=Flat(),
                left=Extrapolation.Flat,
                right=Extrapolation.Flat,
            ),
        )

    def get_value(self, dates: DateArray) -> FloatArray:
        """Return the fixing in force on each query date (previous print held flat)."""
        query = np.asarray(dates).astype("datetime64[D]")
        if query.ndim != 1:
            raise ValueError("historical fixing queries require a 1-D date array.")
        if np.isnat(query).any():
            raise ValueError("historical fixing queries cannot contain NaT.")
        return self._line(query.view(np.int64).astype(np.float64))


__all__ = ["HistoricalFixings"]
