"""Curves: pure ``ZeroCurve`` math, the convention-aware ``YieldCurve``, and their registry.

Interpolators are re-exported from :mod:`common.math.interpolation` for convenience.
"""

from common.math.interpolation import Cubic, Flat, Interpolator, Linear, Mixed, Quadratic

from ._curve_impl.zero_curve import ZeroCurve
from .namespace import CurveNamespace
from .types import CurveInterpolator, CurveSpace, RateExtrapolator
from .yield_curve import YieldCurve, dates_to_x

__all__ = [
    "ZeroCurve",
    "YieldCurve",
    "dates_to_x",
    "CurveInterpolator",
    "CurveSpace",
    "RateExtrapolator",
    "CurveNamespace",
    "Interpolator",
    "Flat",
    "Linear",
    "Cubic",
    "Quadratic",
    "Mixed",
]
