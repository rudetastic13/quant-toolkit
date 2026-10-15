"""The discount-curve contract every curve object in a market satisfies.

:class:`~finance.markets.curves.ZeroCurve` is the atom: one ``Line1d`` in a declared space.
:class:`~finance.markets.curves.CompositeCurve` is a weighted sum of other discount curves in
log-DF space (spreads, bases, turns).  ``YieldCurve``, ``MarketContext`` and the pricing engines
consume only this protocol, so a composite prices and risks through exactly the same path as an
atom.

Parameters
    Every discount curve exposes a flat float64 vector ``params`` that the engines
    differentiate with respect to, and ``with_params`` rebinds it.  For a ``ZeroCurve`` the
    parameters are its non-origin pillar zero rates (Act/365, continuously compounded); for a
    composite they are its components' parameters concatenated in component order.

Affine contract
    ``ln DF(x)`` must be *exactly* affine in ``params`` for every query ``x``, and
    ``log_df_affine`` returns that map as ``(offset, jacobian)``.  Every interpolation scheme
    in :mod:`common.math.interpolation` is affine in its node values, node values are linear
    in the zero rates in either space, and a weighted sum of affine maps is affine, so the
    contract holds for every curve built from these parts.  It is what lets the Numba kernel
    evaluate ``exp(offset + jacobian @ params)`` and scatter adjoints without knowing the
    scheme, and it is why ``ln DF`` has no second-order sensitivity to the parameters: all PV
    curvature lives downstream of this map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Leaf:
    """One atom reached from a curve, with how it enters that curve's parameters.

    Parameters
    ----------
    curve : DiscountCurve
        The atom (a ``ZeroCurve``).
    weight : float
        Product of the composite weights along the path from the root to this atom.
    columns : slice
        The atom's columns in the root's ``params`` and ``log_df_affine`` Jacobian.
    """

    curve: DiscountCurve
    weight: float
    columns: slice


@runtime_checkable
class DiscountCurve(Protocol):
    """Discount factors on a float day-offset axis, differentiable in a flat parameter vector.

    Implementations: :class:`~finance.markets.curves.ZeroCurve` (atom) and
    :class:`~finance.markets.curves.CompositeCurve`.
    All queries take and return float64 arrays of day offsets from the origin; ``x[0] == 0``
    is the origin itself and queries before it are rejected.
    """

    # -- geometry -----------------------------------------------------------------------

    @property
    def x(self) -> FloatArray:
        """Pillar day offsets, origin-inclusive.

        Returns
        -------
        ndarray of float64, shape (n_nodes,)
            Strictly increasing, ``x[0] == 0``.  A composite reports the union of its
            components' pillars.
        """
        ...

    @property
    def dfs(self) -> FloatArray:
        """Discount factors at :attr:`x`.

        Returns
        -------
        ndarray of float64, shape (n_nodes,)
            ``dfs[0] == 1``.
        """
        ...

    @property
    def max_x(self) -> float:
        """Last pillar offset in days; beyond it the curve extrapolates or rejects per its rule."""
        ...

    @property
    def scheme(self) -> Hashable:
        """Everything but the parameter values that pins :meth:`log_df_affine`.

        Two curves with equal ``x`` and equal ``scheme`` have identical affine maps, so a
        prepared pricing program can reuse its query table across them.  This is the
        compatibility key of the reprice / risk hot path.
        """
        ...

    @property
    def is_log_linear(self) -> bool:
        """Whether the curve is log-linear in DF, the one scheme the JAX oracle models."""
        ...

    # -- parameters ---------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Number of differentiation parameters."""
        ...

    @property
    def params(self) -> FloatArray:
        """The differentiation parameters.

        Returns
        -------
        ndarray of float64, shape (n_params,)
        """
        ...

    def with_params(self, params: FloatArray) -> DiscountCurve:
        """Same geometry and scheme with replacement parameters.

        Parameters
        ----------
        params : array_like of float64, shape (n_params,)

        Returns
        -------
        DiscountCurve
            A new curve of the same concrete type; ``self`` is unchanged.

        Raises
        ------
        ValueError
            If ``params`` has the wrong shape or is not finite.
        """
        ...

    # -- queries (xq: float64 day offsets) ----------------------------------------------

    def log_discount_factor(self, xq: FloatArray) -> FloatArray:
        """``ln DF`` at ``xq``.

        Parameters
        ----------
        xq : ndarray of float64
            Day offsets from the origin, ``>= 0``.

        Returns
        -------
        ndarray of float64, same shape as ``xq``
        """
        ...

    def discount_factor(self, xq: FloatArray) -> FloatArray:
        """``DF = exp(ln DF)`` at ``xq``; see :meth:`log_discount_factor`."""
        ...

    def zero_rate(self, xq: FloatArray) -> FloatArray:
        """Act/365 continuously compounded zero rate ``-ln DF / (x / 365)`` at ``xq``.

        At the origin the limit is returned: the instantaneous forward at ``x = 0``.
        """
        ...

    def instantaneous_forward(self, xq: FloatArray) -> FloatArray:
        """Annualised instantaneous forward ``-d ln DF / dt`` (Act/365) at ``xq``."""
        ...

    # -- structure ----------------------------------------------------------------------

    def leaves(self) -> tuple[Leaf, ...]:
        """The atoms this curve is built from, in parameter order.

        Returns
        -------
        tuple of Leaf
            A ``ZeroCurve`` returns itself with weight ``1`` and all its columns; a composite
            returns its components' leaves with accumulated weights and shifted columns.  The
            same atom object may appear more than once when composites share it.
        """
        ...

    def replace(self, old: DiscountCurve, new: DiscountCurve) -> DiscountCurve:
        """This curve with ``new`` substituted wherever ``old`` appears (by identity).

        Returns ``self`` when ``old`` does not appear.  This is how a market propagates a
        replaced curve through every composite built on it.
        """
        ...

    # -- engine hand-off ----------------------------------------------------------------

    def log_df_affine(self, xq: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``ln DF(xq)`` as an exact affine map of :attr:`params`.

        Parameters
        ----------
        xq : ndarray of float64, shape (n_queries,)
            Day offsets from the origin, ``>= 0``.

        Returns
        -------
        offset : ndarray of float64, shape (n_queries,)
        jacobian : ndarray of float64, shape (n_queries, n_params)
            ``ln DF(xq) == offset + jacobian @ params`` for every query.
        """
        ...
