"""Operator support for discount curves: the weighted-sum algebra, spelled as arithmetic.

A spread curve *is* arithmetic on a base curve, so the arithmetic is the API::

    termsofr_3m = sofr + s3m          # CompositeCurve({"base": sofr, "spread": s3m})
    fedfund     = sofr - 5 * bp       # base plus ZeroCurve.constant(-5bp): Fed Funds is SOFR minus 5
    basis       = fedfund - sofr      # CompositeCurve((fedfund, sofr), (1, -1))

Every result is a :class:`~finance.markets.curves.CompositeCurve`, i.e. a weighted sum in log-DF,
so the operators cannot build anything outside the engine's affine contract.  Only ``+`` and
``-`` are defined; any other weight is spelled ``CompositeCurve(components, weights)``.

Operators *define* curves; they never *move* them.  ``sofr + 10 * bp`` is a new curve with one
more parameter (a 10 bp spread), not SOFR shifted by 10 bp (the same parameters at new values).
Both price identically today and report different risk; shifting is ``with_params`` (or the
market's ``bump``).

The mixin carries no state (``__slots__ = ()``) and imports the concrete classes lazily to avoid
an import cycle with the modules that inherit it.
"""

from __future__ import annotations

from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finance.markets.curves._curve_impl.composite_curve import CompositeCurve
    from finance.markets.curves._curve_impl.discount_curve import DiscountCurve


def _is_curve(obj: object) -> bool:
    from finance.markets.curves._curve_impl.discount_curve import DiscountCurve

    return isinstance(obj, DiscountCurve)


def _is_scalar(obj: object) -> bool:
    return isinstance(obj, Real) and not isinstance(obj, bool)


def _constant_like(curve: DiscountCurve, rate: float):
    from finance.markets.curves._curve_impl.zero_curve import ZeroCurve

    return ZeroCurve.constant(float(rate), curve.max_x)


class CurveArithmetic:
    """Mixin giving a :class:`DiscountCurve` the ``+`` and ``-`` operators.

    Notes
    -----
    ``a + b`` labels the result ``{"base": a, "spread": b}`` (the left operand is the base).
    ``a - b`` is the unlabelled ``(a, b)`` with weights ``(1, -1)``: a basis has no base.
    A scalar operand ``s`` stands for ``ZeroCurve.constant(s, a.max_x)``, a flat spread.
    """

    __slots__ = ()

    def __add__(self, other: DiscountCurve | float) -> CompositeCurve:
        from finance.markets.curves._curve_impl.composite_curve import CompositeCurve

        if _is_curve(other):
            return CompositeCurve({"base": self, "spread": other})
        if _is_scalar(other):
            return CompositeCurve({"base": self, "spread": _constant_like(self, other)})
        return NotImplemented

    def __radd__(self, other: float) -> CompositeCurve:
        return self.__add__(other)

    def __sub__(self, other: DiscountCurve | float) -> CompositeCurve:
        from finance.markets.curves._curve_impl.composite_curve import CompositeCurve

        if _is_curve(other):
            return CompositeCurve((self, other), (1.0, -1.0))
        if _is_scalar(other):
            return CompositeCurve({"base": self, "spread": _constant_like(self, -other)})
        return NotImplemented
