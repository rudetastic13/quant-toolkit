"""Curves: the ``DiscountCurve`` protocol and its implementations (``ZeroCurve`` atom,
``CompositeCurve`` weighted sums in log-DF space, also reachable as ``a + b``, ``a - 5 * bp``),
and the convention-aware ``YieldCurve``.

Interpolators are re-exported from :mod:`common.math.interpolation` for convenience.
"""

from common.math.interpolation import Cubic, Flat, Interpolator, Linear, Mixed, Quadratic

from ._curve_impl.composite_curve import CompositeCurve
from ._curve_impl.discount_curve import DiscountCurve, Leaf
from ._curve_impl.zero_curve import ZeroCurve
from .types import CurveInterpolator, CurveSpace, RateExtrapolator
from .units import bp, pct
from .yield_curve import YieldCurve, dates_to_x

__all__ = [
    "DiscountCurve",
    "Leaf",
    "ZeroCurve",
    "CompositeCurve",
    "bp",
    "pct",
    "YieldCurve",
    "dates_to_x",
    "CurveInterpolator",
    "CurveSpace",
    "RateExtrapolator",
    "Interpolator",
    "Flat",
    "Linear",
    "Cubic",
    "Quadratic",
    "Mixed",
]
