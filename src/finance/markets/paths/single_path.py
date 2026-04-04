"""Path data, useful for historic resets or pre-computed forward rates"""
import numpy as np
from dataclasses import dataclass, field
from finance.dates import Date
from common.math.line import (
    Line1d,
    InterpolationType,
    ExtrapolationType,
)


@dataclass
class SinglePath:
    """Class representing a single path of market data for a financial instrument."""
    name: str
    as_of_date: Date
    dates: np.ndarray
    values: np.ndarray
    interp: InterpolationType = InterpolationType.Flat
    extrap: ExtrapolationType = ExtrapolationType.Flat
    curve: Line1d = field(init=False)
    _as_of_date_int: int = field(init=False)

    def __post_init__(self):
        self._as_of_date_int = self.as_of_date.toordinal()
        self.curve = Line1d(
            x=self.dates.astype(np.int64) - self._as_of_date_int,
            y=self.values,
            interp=self.interp,
            extrap=self.extrap,
        )

    def get_value(self, dates: np.ndarray) -> np.ndarray:
        """Get interpolated/extrapolated values for given dates."""
        date_ints = dates.astype(np.int64) - self._as_of_date_int
        return self.curve.get_value(date_ints)

@dataclass
class FlatPath:
    """Class representing a flat path of market data for a financial instrument."""
    name: str
    as_of_date: Date
    value: float

    def get_value(self, dates: np.ndarray) -> np.ndarray:
        """Get flat value for given dates."""
        return np.full_like(dates, self.value, dtype=np.float64)


@dataclass
class InvertedPath(SinglePath):
    """Invert the single-path of values"""

    def __post_init__(self):
        self.values[np.isclose(self.values, 0.0)] = 1.0 # avoid division by zero
        super().__post_init__()
