"""A base discount curve plus a spread curve: ``DF = DF_base · DF_spread``.

The two-component :class:`~finance.markets.curves.CompositeCurve` with weights ``(1, 1)``,
given a name and the accessors a desk actually uses: ``base``, ``spread``, ``spread_rate(x)``,
and the parameter blocks of each so risk splits into *base* (index/funding) and *spread*
(basis) ladders.  :meth:`SpreadCurve.flat` builds the constant-spread case (an index defined as
another index plus a fixed margin, a flat funding spread, a z-spread) from a single number.

Deciding between a ``SpreadCurve`` and a spread inside the rate generator: when the spread is
a *risk factor* the desk wants to see and bump, it belongs here, as a curve with parameters.
When it is a *contractual* redefinition of an index with no risk of its own, it is a
convention on the leg or index, not a curve.
"""

from __future__ import annotations

from typing import Hashable

import numpy as np

from finance.markets.curves._curve_impl.composite_curve import CompositeCurve
from finance.markets.curves._curve_impl.discount_curve import DiscountCurve, FloatArray
from finance.markets.curves._curve_impl.zero_curve import DAYS_PER_YEAR, ZeroCurve


class SpreadCurve:
    """Base curve plus spread curve in log-DF space: ``ln DF = ln DF_base + ln DF_spread``.

    Parameters
    ----------
    base : DiscountCurve
        The curve being spread over (an index atom, or another composite).
    spread : DiscountCurve
        The spread, itself a discount curve: its zero rate at ``x`` is the continuously
        compounded Act/365 spread to ``x``.

    Notes
    -----
    Parameters are ``base.params`` followed by ``spread.params``; :attr:`base_slice` and
    :attr:`spread_slice` index them.  Composition, not inheritance: the arithmetic lives in
    the wrapped :attr:`composite`, this class only names the two roles.

    See Also
    --------
    SpreadCurve.flat : constant spread from one number.
    CompositeCurve : the general weighted sum.
    """

    __slots__ = ("_composite",)

    def __init__(self, base: DiscountCurve, spread: DiscountCurve) -> None:
        self._composite = CompositeCurve((base, spread))

    @classmethod
    def _from_composite(cls, composite: CompositeCurve) -> SpreadCurve:
        curve = object.__new__(cls)
        curve._composite = composite
        return curve

    @classmethod
    def flat(cls, base: DiscountCurve, spread: float, *, horizon: float | None = None) -> SpreadCurve:
        """Constant continuously compounded Act/365 ``spread`` over ``base``.

        The spread curve is a two-node log-linear ``ZeroCurve`` on ``[0, horizon]`` whose single
        parameter *is* the spread; flat-forward extrapolation continues it unchanged for every
        ``x``, so ``horizon`` only names the pillar.

        Parameters
        ----------
        base : DiscountCurve
        spread : float
            Decimal rate, e.g. ``0.00085`` for 8.5 bp.
        horizon : float, optional
            Day offset of the spread pillar.  Default ``base.max_x``.

        Returns
        -------
        SpreadCurve
            ``n_params == base.n_params + 1``; the last parameter equals ``spread``.

        Raises
        ------
        ValueError
            If ``horizon`` is not positive.
        """
        horizon = float(base.max_x if horizon is None else horizon)
        if not horizon > 0.0:
            raise ValueError(f"horizon must be positive; got {horizon}.")
        x = np.array([0.0, horizon])
        dfs = np.array([1.0, np.exp(-float(spread) * horizon / DAYS_PER_YEAR)])
        return cls(base, ZeroCurve(x, dfs))

    # -- state ----------------------------------------------------------------------------

    @property
    def base(self) -> DiscountCurve:
        """The base component."""
        return self._composite.components[0]

    @property
    def spread(self) -> DiscountCurve:
        """The spread component."""
        return self._composite.components[1]

    @property
    def composite(self) -> CompositeCurve:
        """The underlying two-component composite with weights ``(1, 1)``."""
        return self._composite

    @property
    def x(self) -> FloatArray:
        """Union of base and spread pillars, origin-inclusive."""
        return self._composite.x

    @property
    def dfs(self) -> FloatArray:
        """Spread-adjusted discount factors at :attr:`x`."""
        return self._composite.dfs

    @property
    def max_x(self) -> float:
        """Largest pillar offset of base or spread."""
        return self._composite.max_x

    @property
    def scheme(self) -> Hashable:
        """``("spread", composite.scheme)``."""
        return ("spread", self._composite.scheme)

    @property
    def is_log_linear(self) -> bool:
        """Always ``False``."""
        return False

    # -- parameters -----------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """``base.n_params + spread.n_params``."""
        return self._composite.n_params

    @property
    def params(self) -> FloatArray:
        """``base.params`` followed by ``spread.params``."""
        return self._composite.params

    @property
    def base_slice(self) -> slice:
        """Slice of :attr:`params` (and Jacobian columns) owned by the base."""
        return self._composite.component_slice(0)

    @property
    def spread_slice(self) -> slice:
        """Slice of :attr:`params` (and Jacobian columns) owned by the spread."""
        return self._composite.component_slice(1)

    def with_params(self, params: FloatArray) -> SpreadCurve:
        """Rebind base and spread parameters from one flat vector.

        Parameters
        ----------
        params : array_like of float64, shape (n_params,)

        Returns
        -------
        SpreadCurve
        """
        return self._from_composite(self._composite.with_params(params))

    def with_base(self, base: DiscountCurve) -> SpreadCurve:
        """Same spread over a different base (any geometry)."""
        return SpreadCurve(base, self.spread)

    def with_spread(self, spread: DiscountCurve) -> SpreadCurve:
        """Same base under a different spread curve (any geometry)."""
        return SpreadCurve(self.base, spread)

    # -- queries --------------------------------------------------------------------------

    def log_discount_factor(self, xq: FloatArray) -> FloatArray:
        """``ln DF_base(xq) + ln DF_spread(xq)``."""
        return self._composite.log_discount_factor(xq)

    def discount_factor(self, xq: FloatArray) -> FloatArray:
        """``DF_base(xq) · DF_spread(xq)``."""
        return self._composite.discount_factor(xq)

    def zero_rate(self, xq: FloatArray) -> FloatArray:
        """``z_base(xq) + spread_rate(xq)``."""
        return self._composite.zero_rate(xq)

    def instantaneous_forward(self, xq: FloatArray) -> FloatArray:
        """``f_base(xq) + f_spread(xq)``."""
        return self._composite.instantaneous_forward(xq)

    def spread_rate(self, xq: FloatArray) -> FloatArray:
        """The spread component's zero rate: the Act/365 continuous spread over ``base`` to ``xq``.

        Parameters
        ----------
        xq : ndarray of float64

        Returns
        -------
        ndarray of float64, same shape as ``xq``
        """
        return self.spread.zero_rate(xq)

    def log_df_affine(self, xq: FloatArray) -> tuple[FloatArray, FloatArray]:
        """The composite's affine map; columns ``base_slice`` then ``spread_slice``."""
        return self._composite.log_df_affine(xq)

    def __repr__(self) -> str:
        return f"SpreadCurve(base={self.base!r}, spread={self.spread!r})"
