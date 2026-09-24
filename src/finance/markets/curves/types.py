"""Curve-level enums: the coordinate space a curve interpolates in, its extrapolation, and
the named ``(space, scheme)`` pairs used by config / Excel."""

from __future__ import annotations

from enum import Enum, IntEnum, auto

from common.math.interpolation import Cubic, Interpolator, Linear, Quadratic


class CurveSpace(Enum):
    """Coordinate the interpolator works in.  The curve owns the transform to and from DF."""

    LogDF = auto()  # y = ln DF(x)
    ZeroRate = auto()  # y = r(x), DF = exp(-r * x / 365): continuously compounded, Act/365


class RateExtrapolator(IntEnum):
    """Behaviour beyond the last pillar.  Queries before the origin are always rejected."""

    FlatForward = 1  # continue the terminal instantaneous forward
    NotAllowed = 2
    Flat = 1  # alias for FlatForward (historical name)


class CurveInterpolator(IntEnum):
    """Named ``(space, scheme)`` pairs: a parse table for config and Excel.

    ``resolve()`` returns the objects the curve is actually built from; code that constructs
    curves directly should pass a :class:`CurveSpace` and an interpolator instance instead.
    """

    LogLinearDF = 1
    LogCubicDF = 2
    RateLinear = 3
    RateQuadratic = 4
    RateCubic = 5

    def resolve(self) -> tuple[CurveSpace, Interpolator]:
        return _SCHEMES[self]


_SCHEMES: dict[CurveInterpolator, tuple[CurveSpace, Interpolator]] = {
    CurveInterpolator.LogLinearDF: (CurveSpace.LogDF, Linear()),
    CurveInterpolator.LogCubicDF: (CurveSpace.LogDF, Cubic()),
    CurveInterpolator.RateLinear: (CurveSpace.ZeroRate, Linear()),
    CurveInterpolator.RateQuadratic: (CurveSpace.ZeroRate, Quadratic()),
    CurveInterpolator.RateCubic: (CurveSpace.ZeroRate, Cubic()),
}

__all__ = ["CurveInterpolator", "CurveSpace", "RateExtrapolator"]
