"""Pure discount curve on day offsets: a :class:`Line1d` in log-DF or zero-rate space plus the
discount-factor invariants.

``ZeroCurve`` knows nothing about dates, indices, fixings or coupons.  Its ``x`` axis is
float days from the origin (``x[0] == 0``, ``dfs[0] == 1``); :class:`YieldCurve` owns the
origin date and converts dates to ``x``.  Queries before the origin are rejected.  Beyond the
last pillar the terminal instantaneous forward is continued (or the query is rejected).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from common.math.interpolation import Interpolator, Linear
from common.math.line import Extrapolation, Line1d
from finance.markets.curves.types import CurveSpace, RateExtrapolator

FloatArray = NDArray[np.float64]

DAYS_PER_YEAR = 365.0


def _to_space(space: CurveSpace, x: FloatArray, dfs: FloatArray) -> FloatArray:
    """Node values in the interpolation space."""
    if space is CurveSpace.LogDF:
        return np.log(dfs)
    rates = np.empty_like(dfs)
    rates[1:] = -np.log(dfs[1:]) / (x[1:] / DAYS_PER_YEAR)
    # r(0) is the limit of -ln DF / t, i.e. the instantaneous forward at the origin: take it
    # by linear extrapolation from the first two pillars (or the first pillar when there is one).
    if x.size >= 3:
        rates[0] = rates[1] - x[1] * (rates[2] - rates[1]) / (x[2] - x[1])
    else:
        rates[0] = rates[1]
    return rates


def _check_dfs(x: FloatArray, dfs: FloatArray) -> None:
    if dfs.shape != x.shape:
        raise ValueError(f"x and dfs must have the same shape; got {x.shape} and {dfs.shape}.")
    if not np.isfinite(dfs).all() or np.any(dfs <= 0.0):
        raise ValueError("dfs must be finite, strictly positive discount factors.")
    if abs(dfs[0] - 1.0) > 1e-12:
        raise ValueError("the first node (the origin) must have DF = 1.0.")


class ZeroCurve:
    """Discount factors on a day-offset axis, interpolated in a declared space.

    Parameters
    ----------
    x : float64 array, strictly increasing, ``x[0] == 0.0``
        Day offsets of the pillars from the origin.
    dfs : float64 array
        Discount factors at the pillars; ``dfs[0]`` must be 1.
    space : CurveSpace
        Interpolate ``ln DF`` (default) or the Act/365 continuously compounded zero rate.
    interpolator : Interpolator
        Any :mod:`common.math.interpolation` scheme; parameters travel with it.
    extrapolation : RateExtrapolator
        ``FlatForward`` continues the terminal instantaneous forward; ``NotAllowed`` rejects.
    """

    __slots__ = ("_x", "_dfs", "_space", "_interpolator", "_extrapolation", "_line", "_terminal_forward")

    def __init__(
        self,
        x: FloatArray,
        dfs: FloatArray,
        *,
        space: CurveSpace = CurveSpace.LogDF,
        interpolator: Interpolator = Linear(),
        extrapolation: RateExtrapolator = RateExtrapolator.FlatForward,
    ) -> None:
        if x.ndim != 1 or x.size < 2:
            raise ValueError("x must be a 1-D array with at least 2 nodes (origin + one pillar).")
        if x[0] != 0.0:
            raise ValueError(f"x[0] must be 0.0 (the origin); got {x[0]}.")
        _check_dfs(x, dfs)
        space = CurveSpace(space)
        extrapolation = RateExtrapolator(extrapolation)
        if extrapolation is RateExtrapolator.NotAllowed:
            right = Extrapolation.NotAllowed
        elif space is CurveSpace.LogDF:
            right = Extrapolation.Linear  # linear ln DF == constant forward
        else:
            right = Extrapolation.Flat  # placeholder: flat-forward is applied in the DF transform
        line = Line1d(x, _to_space(space, x, dfs), interpolator, left=Extrapolation.NotAllowed, right=right)
        self._init(x, dfs, space, interpolator, extrapolation, line)

    def _init(self, x, dfs, space, interpolator, extrapolation, line) -> None:
        self._x = x
        self._dfs = dfs
        self._space = space
        self._interpolator = interpolator
        self._extrapolation = extrapolation
        self._line = line
        self._terminal_forward = float(self._forward_interior(x[-1:])[0])

    # -- state ------------------------------------------------------------------------

    @property
    def x(self) -> FloatArray:
        """Pillar day offsets, origin-inclusive (``x[0] == 0``)."""
        return self._x

    @property
    def dfs(self) -> FloatArray:
        return self._dfs

    @property
    def max_x(self) -> float:
        return float(self._x[-1])

    @property
    def space(self) -> CurveSpace:
        return self._space

    @property
    def interpolator(self) -> Interpolator:
        return self._interpolator

    @property
    def extrapolation(self) -> RateExtrapolator:
        return self._extrapolation

    @property
    def line(self) -> Line1d:
        """The interpolated line in ``space``; ``line.coefficients`` is the engine hand-off."""
        return self._line

    @property
    def node_zero_rates(self) -> FloatArray:
        """Act/365 continuously compounded zero rates at the non-origin pillars (the calibration parameters)."""
        return -np.log(self._dfs[1:]) / (self._x[1:] / DAYS_PER_YEAR)

    @property
    def is_log_linear(self) -> bool:
        """Log-linear DF: the one scheme the Numba/JAX engines model analytically today."""
        return self._space is CurveSpace.LogDF and self._interpolator == Linear()

    def with_dfs(self, dfs: FloatArray) -> ZeroCurve:
        """Same pillars, space, interpolator and extrapolation with new discount factors."""
        _check_dfs(self._x, dfs)
        curve = object.__new__(ZeroCurve)
        curve._init(
            self._x,
            dfs,
            self._space,
            self._interpolator,
            self._extrapolation,
            self._line.with_y(_to_space(self._space, self._x, dfs)),
        )
        return curve

    # -- queries (xq: float64 day offsets) ---------------------------------------------

    def log_discount_factor(self, xq: FloatArray) -> FloatArray:
        y = self._line(xq)
        if self._space is CurveSpace.LogDF:
            return y
        log_df = -y * xq / DAYS_PER_YEAR
        beyond = xq > self._x[-1]
        if beyond.any():  # the line held r flat; replace with the flat-forward continuation
            log_df[beyond] = (
                self._log_df_terminal() - self._terminal_forward * (xq[beyond] - self._x[-1]) / DAYS_PER_YEAR
            )
        return log_df

    def discount_factor(self, xq: FloatArray) -> FloatArray:
        return np.exp(self.log_discount_factor(xq))

    def zero_rate(self, xq: FloatArray) -> FloatArray:
        """Act/365 continuously compounded zero rate; at the origin, the instantaneous forward."""
        out = np.empty(xq.shape, dtype=np.float64)
        at_origin = xq <= 0.0
        if at_origin.any():
            out[at_origin] = self.instantaneous_forward(xq[at_origin])
        inside = ~at_origin
        if inside.any():
            out[inside] = -self.log_discount_factor(xq[inside]) / (xq[inside] / DAYS_PER_YEAR)
        return out

    def instantaneous_forward(self, xq: FloatArray) -> FloatArray:
        """Annualised instantaneous forward ``-d ln DF / dt`` (Act/365)."""
        f = self._forward_interior(xq)
        if self._space is CurveSpace.ZeroRate:
            beyond = xq > self._x[-1]
            if beyond.any():
                f[beyond] = self._terminal_forward
        return f

    # -- internals --------------------------------------------------------------------

    def _forward_interior(self, xq: FloatArray) -> FloatArray:
        if self._space is CurveSpace.LogDF:
            return -self._line.derivative(xq) * DAYS_PER_YEAR
        # ln DF = -r(x) x / 365  ->  f = r + x r'
        return self._line(xq) + xq * self._line.derivative(xq)

    def _log_df_terminal(self) -> float:
        return float(np.log(self._dfs[-1]))

    def __repr__(self) -> str:
        return (
            f"ZeroCurve(nodes={self._x.size}, max_x={self.max_x:g}, space={self._space.name}, "
            f"interpolator={self._interpolator!r}, extrapolation={self._extrapolation.name})"
        )


__all__ = ["DAYS_PER_YEAR", "ZeroCurve"]
