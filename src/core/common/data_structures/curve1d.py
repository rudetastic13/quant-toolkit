"""Types and functions to support a 1-dimensional curve structure"""
from enum import IntEnum, auto
import numpy as np
from numpy.typing import NDArray
from pandas.io.formats.format import return_docstring

IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]

class InterpolationType(IntEnum):
    """Enumeration of interpolation types for Curve1D"""
    Linear = auto()
    Flat = auto()

class ExtrapolationType:
    NotAllowed = auto()
    Flat = auto()

def _interp1dflat(
    x: IntArray,
    xs: IntArray,
    ys: FloatArray,
    extrap: ExtrapolationType) -> FloatArray:
    """1D flat interpolation implementation"""
    idx = np.clip(np.searchsorted(xs, x), 0, len(xs) - 1)
    if extrap == ExtrapolationType.NotAllowed:
        low, high = np.min(xs), np.max(xs)
        mask = (x < low) | (x > high)
        result = ys[idx]
        result[mask] = np.nan
        return result
    return ys[idx]

def _interp1dlinear(
    x: IntArray,
    xs: IntArray,
    ys: FloatArray,
    extrap: ExtrapolationType) -> FloatArray:
    """1D linear interpolation implementation"""
    if extrap == ExtrapolationType.NotAllowed:
        return np.interp(x, xp=xs, fp=ys, left=np.nan, right=np.nan)
    return np.interp(x, xp=xs, fp=ys, left=ys[0], right=ys[-1])

class Curve1d:

    def __init__(
        self,
        x: IntArray,
        y: FloatArray,
        interp: InterpolationType = InterpolationType.Linear,
        extrap: ExtrapolationType = ExtrapolationType.NotAllowed,
    ):
        self.x = x
        self.y = y
        self.interp = interp
        self.extrap = extrap
        self._curve = None

    def _get(self):
        if not self._curve:
            if self.interp == InterpolationType.Flat:
                self._curve = lambda x: _interp1dflat(x, self.x, self.y, self.extrap)
            else:
                self._curve = lambda x: _interp1dlinear(x, self.x, self.y, self.extrap)
        return self._curve

    def get_value(self, x: IntArray) -> FloatArray:
        """Get interpolated value at given x positions"""
        result = self._get()(x)
        if self.extrap == ExtrapolationType.NotAllowed:
            if np.isnan(result).any():
                raise ValueError("Extrapolation not allowed, some x values are out of bounds")
        return result
