"""One-dimensional interpolated line: nodes + an interpolator + per-side extrapolation.

``Line1d`` is the foundation for every 1-D curve-like object: historical fixings, rate paths,
and (in log-DF or zero-rate space) the discount curve.  It owns node validation, splits a
query into left / interior / right, and rebinds new node values cheaply via :meth:`with_y`.
Interpolation math lives in :mod:`common.math.interpolation`.

``x`` and ``y`` are 1-D float64 arrays of equal length; ``x`` strictly increasing.  A single
node is allowed (a one-print fixings history) but only :class:`Flat` can interpolate it.
Callers convert dates to floats before they get here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto

import numpy as np
from numpy.typing import NDArray

from common.math.interpolation import BoundInterpolator, Interpolator, Linear

FloatArray = NDArray[np.float64]


class Extrapolation(IntEnum):
    """Behaviour beyond an end node.  Applied independently on the left and on the right."""

    NotAllowed = auto()  # raise on any query beyond the end node
    Flat = auto()  # hold the end node's value
    Linear = auto()  # continue the fitted interpolant's end slope


def _check_values(y: FloatArray, x: FloatArray) -> None:
    if y.shape != x.shape:
        raise ValueError(f"x and y must have the same shape; got {x.shape} and {y.shape}.")
    if not np.isfinite(y).all():
        raise ValueError("y must be finite.")


def _check_nodes(x: FloatArray, y: FloatArray) -> None:
    if x.ndim != 1 or x.size < 1:
        raise ValueError("x must be a non-empty 1-D array.")
    if not np.isfinite(x).all():
        raise ValueError("x must be finite.")
    if np.any(x[1:] <= x[:-1]):
        raise ValueError("x must be strictly increasing with no duplicates.")
    _check_values(y, x)


@dataclass(frozen=True, eq=False)
class Line1d:
    x: FloatArray
    y: FloatArray
    interpolator: Interpolator = Linear()
    left: Extrapolation = Extrapolation.NotAllowed
    right: Extrapolation = Extrapolation.NotAllowed
    _bound: BoundInterpolator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _check_nodes(self.x, self.y)
        object.__setattr__(self, "_bound", self.interpolator.fit(self.x, self.y))

    # -- construction -------------------------------------------------------

    def with_y(self, y: FloatArray) -> Line1d:
        """Same nodes, interpolator and extrapolation with new node values.

        Skips the ``x`` checks (they cannot have changed) and only refits.  This is the path
        calibration and bump-and-reprice risk take on every iteration.
        """
        _check_values(y, self.x)
        line = object.__new__(Line1d)
        object.__setattr__(line, "x", self.x)
        object.__setattr__(line, "y", y)
        object.__setattr__(line, "interpolator", self.interpolator)
        object.__setattr__(line, "left", self.left)
        object.__setattr__(line, "right", self.right)
        object.__setattr__(line, "_bound", self.interpolator.fit(self.x, y))
        return line

    # -- queries --------------------------------------------------------------

    def __call__(self, xq: FloatArray) -> FloatArray:
        """Interpolated values at ``xq``; the ends are handled per ``left`` / ``right``."""
        lo, hi, inside = self._split(xq)
        out = self._bound(inside)
        if lo is not None:
            out[lo] = self._extrapolate_value(self.left, xq[lo], 0)
        if hi is not None:
            out[hi] = self._extrapolate_value(self.right, xq[hi], -1)
        return out

    get_value = __call__

    def derivative(self, xq: FloatArray) -> FloatArray:
        """First derivative dy/dx at ``xq``; zero under flat extrapolation, the end slope under linear."""
        lo, hi, inside = self._split(xq)
        out = self._bound.derivative(inside)
        if lo is not None:
            out[lo] = self._extrapolate_slope(self.left, xq[lo], 0)
        if hi is not None:
            out[hi] = self._extrapolate_slope(self.right, xq[hi], -1)
        return out

    @property
    def coefficients(self) -> FloatArray:
        """Piecewise-polynomial coefficients of the interior interpolant, for engine kernels."""
        return self._bound.coefficients

    # -- internals ----------------------------------------------------------

    def _split(self, xq: FloatArray) -> tuple[FloatArray | None, FloatArray | None, FloatArray]:
        """Masks of out-of-range points (``None`` when there are none) and the clipped interior query."""
        lo = xq < self.x[0]
        hi = xq > self.x[-1]
        any_lo = bool(lo.any())
        any_hi = bool(hi.any())
        if not (any_lo or any_hi):
            return None, None, xq
        return (lo if any_lo else None), (hi if any_hi else None), np.clip(xq, self.x[0], self.x[-1])

    def _end_slope(self, end: int) -> float:
        return float(self._bound.derivative(self.x[end : end + 1] if end == 0 else self.x[-1:])[0])

    def _reject(self, xq: FloatArray, end: int) -> None:
        if end == 0:
            side, furthest = "before the first", xq.min()
        else:
            side, furthest = "beyond the last", xq.max()
        raise ValueError(
            f"{xq.size} query point(s) {side} node x={self.x[end]} (furthest: {furthest}); extrapolation is not allowed."
        )

    def _extrapolate_value(self, mode: Extrapolation, xq: FloatArray, end: int) -> FloatArray:
        if mode is Extrapolation.Flat:
            return np.full(xq.shape, self.y[end], dtype=np.float64)
        if mode is Extrapolation.Linear:
            return self.y[end] + self._end_slope(end) * (xq - self.x[end])
        self._reject(xq, end)

    def _extrapolate_slope(self, mode: Extrapolation, xq: FloatArray, end: int) -> FloatArray:
        if mode is Extrapolation.Flat:
            return np.zeros(xq.shape, dtype=np.float64)
        if mode is Extrapolation.Linear:
            return np.full(xq.shape, self._end_slope(end), dtype=np.float64)
        self._reject(xq, end)

    def __repr__(self) -> str:
        return (
            f"Line1d(nodes={self.x.size}, x=[{self.x[0]}, {self.x[-1]}], "
            f"interpolator={self.interpolator!r}, left={self.left.name}, right={self.right.name})"
        )


__all__ = ["Extrapolation", "FloatArray", "Line1d"]
