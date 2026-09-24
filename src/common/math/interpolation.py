"""Interpolation schemes for :class:`common.math.line.Line1d`.

Two phases.  An :class:`Interpolator` is a small frozen configuration object: the scheme and
its parameters.  ``fit(x, y)`` binds it to node data and returns a :class:`BoundInterpolator`
that evaluates, differentiates, and exports its piecewise-polynomial coefficients.

Every scheme except :class:`Flat` is a scipy ``PPoly`` underneath, so ``coefficients`` share
one layout: shape ``(degree + 1, n_intervals)``, highest power first, each piece expressed in
``x - x[i]``.  An engine that wants to fuse evaluation into its own kernel does
``searchsorted`` + Horner on that array.  Fitting happens here, once per construction, and
nothing in this module needs to be jit-compatible.

Queries handed to a bound interpolator are assumed to lie inside ``[x[0], x[-1]]``;
extrapolation is :class:`~common.math.line.Line1d`'s job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, PPoly

FloatArray = NDArray[np.float64]


# region, protocols
@runtime_checkable
class BoundInterpolator(Protocol):
    """A scheme bound to node data."""

    @property
    def x(self) -> FloatArray: ...

    def __call__(self, xq: FloatArray) -> FloatArray: ...

    def derivative(self, xq: FloatArray) -> FloatArray: ...

    @property
    def coefficients(self) -> FloatArray:
        """``(degree + 1, n_intervals)`` PPoly coefficients, highest power first."""
        ...


@runtime_checkable
class Interpolator(Protocol):
    """A scheme and its parameters; ``fit`` binds it to nodes."""

    def fit(self, x: FloatArray, y: FloatArray) -> BoundInterpolator: ...


# endregion


# region, bound implementations
class BoundPPoly:
    """Any piecewise polynomial: evaluation and derivative come from scipy."""

    __slots__ = ("_pp",)

    def __init__(self, pp: PPoly) -> None:
        self._pp = pp

    @property
    def x(self) -> FloatArray:
        return self._pp.x

    def __call__(self, xq: FloatArray) -> FloatArray:
        return self._pp(xq)

    def derivative(self, xq: FloatArray) -> FloatArray:
        return self._pp(xq, nu=1)

    @property
    def coefficients(self) -> FloatArray:
        return self._pp.c


class BoundFlat:
    """Previous-value step.  Not a PPoly: a degree-0 PPoly would return ``y[-2]`` at ``x[-1]``."""

    __slots__ = ("_x", "_y")

    def __init__(self, x: FloatArray, y: FloatArray) -> None:
        self._x = x
        self._y = y

    @property
    def x(self) -> FloatArray:
        return self._x

    def __call__(self, xq: FloatArray) -> FloatArray:
        idx = np.searchsorted(self._x, xq, side="right") - 1
        np.clip(idx, 0, self._x.size - 1, out=idx)
        return self._y[idx]

    def derivative(self, xq: FloatArray) -> FloatArray:
        return np.zeros(xq.shape, dtype=np.float64)

    @property
    def coefficients(self) -> FloatArray:
        return self._y[np.newaxis, :-1]


# endregion


# region, schemes
def _require_two_nodes(x: FloatArray, scheme: str) -> None:
    if x.size < 2:
        raise ValueError(f"{scheme} interpolation needs at least 2 nodes; got {x.size}.")


@dataclass(frozen=True)
class Flat:
    """Hold the previous node's value: ``y[i]`` on ``[x[i], x[i+1])``, ``y[-1]`` at ``x[-1]``.

    A Friday fixing covers Saturday and Sunday; Monday's fixing takes over on Monday.
    """

    def fit(self, x: FloatArray, y: FloatArray) -> BoundInterpolator:
        return BoundFlat(x, y)


@dataclass(frozen=True)
class Linear:
    """Piecewise linear in ``y``.  Local: a node moves only its two adjacent intervals."""

    def fit(self, x: FloatArray, y: FloatArray) -> BoundInterpolator:
        _require_two_nodes(x, "Linear")
        slopes = np.diff(y) / np.diff(x)
        return BoundPPoly(PPoly(np.vstack([slopes, y[:-1]]), x))


@dataclass(frozen=True)
class Cubic:
    """Cubic spline in ``y``; ``bc_type`` is passed straight to :class:`scipy.interpolate.CubicSpline`.

    ``"natural"`` (default) prescribes zero second derivative at both ends.  ``"clamped"``
    prescribes zero first derivative.  ``((2, a), (2, b))`` prescribes second derivatives
    ``a`` and ``b``; ``((1, a), (1, b))`` first derivatives.  ``"not-a-knot"`` is also valid.
    Global: every node affects the whole curve.
    """

    bc_type: str | tuple = "natural"

    def fit(self, x: FloatArray, y: FloatArray) -> BoundInterpolator:
        _require_two_nodes(x, "Cubic")
        return BoundPPoly(CubicSpline(x, y, bc_type=self.bc_type))


@dataclass(frozen=True)
class Quadratic:
    """Piecewise quadratic in ``y`` with knot slopes from blended forward/backward sweeps.

    The forward sweep starts from ``left_slope`` and the backward sweep from ``right_slope``;
    each knot's slope is a position-weighted blend of the two, which damps the oscillation a
    single sweep would produce.  Continuous in value everywhere; C1 only where the two sweeps
    agree.
    """

    left_slope: float = 0.0
    right_slope: float = 0.0

    def fit(self, x: FloatArray, y: FloatArray) -> BoundInterpolator:
        _require_two_nodes(x, "Quadratic")
        h = np.diff(x)
        d = np.diff(y) / h
        m = _blended_slopes(d, self.left_slope, self.right_slope)
        return BoundPPoly(PPoly(np.vstack([(d - m[:-1]) / h, m[:-1], y[:-1]]), x))


@dataclass(frozen=True)
class Mixed:
    """One scheme up to ``x[switch_node]`` and another after it; both share that node.

    Segments are fit independently, so the interpolant is continuous in value at the switch
    but not in slope.  Typical use: ``Linear`` over the meeting-dated front end and ``Cubic``
    beyond it.
    """

    short: Interpolator
    long: Interpolator
    switch_node: int

    def fit(self, x: FloatArray, y: FloatArray) -> BoundInterpolator:
        k = self.switch_node
        n = x.size
        if not 1 <= k <= n - 2:
            raise ValueError(f"switch_node must leave at least 2 nodes in each segment; got {k} of {n} nodes.")
        short = self.short.fit(x[: k + 1], y[: k + 1]).coefficients
        long = self.long.fit(x[k:], y[k:]).coefficients
        order = max(short.shape[0], long.shape[0])
        c = np.zeros((order, n - 1), dtype=np.float64)
        c[order - short.shape[0] :, :k] = short
        c[order - long.shape[0] :, k:] = long
        return BoundPPoly(PPoly(c, x))


# endregion


def _blended_slopes(d: FloatArray, left: float, right: float) -> FloatArray:
    """Knot slopes for :class:`Quadratic`: blend of the two alternating-sign sweeps.

    Forward sweep ``m[k+1] = 2 d[k] - m[k]`` from ``m[0] = left``; backward sweep
    ``m[k] = 2 d[k] - m[k+1]`` from ``m[-1] = right``.  Multiplying by ``(-1)^k`` turns each
    into a plain running sum, so both are cumsums rather than loops.
    """
    n = d.size + 1
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    signed = sign[:-1] * d  # (-1)^j d[j]

    forward = np.empty(n, dtype=np.float64)
    forward[0] = left
    forward[1:] = left - 2.0 * np.cumsum(signed)  # u[k] = left + 2 sum_{j<k} (-1)^{j+1} d[j]
    m_fwd = sign * forward

    backward = np.empty(n, dtype=np.float64)
    backward[-1] = sign[-1] * right
    backward[:-1] = backward[-1] + 2.0 * np.cumsum(signed[::-1])[::-1]  # v[k] = v[-1] + 2 sum_{j>=k} (-1)^j d[j]
    m_bwd = sign * backward

    w = np.linspace(0.0, 1.0, n)
    m = (1.0 - w) * m_fwd + w * m_bwd
    m[0] = left
    m[-1] = right
    return m


__all__ = [
    "BoundFlat",
    "BoundInterpolator",
    "BoundPPoly",
    "Cubic",
    "Flat",
    "FloatArray",
    "Interpolator",
    "Linear",
    "Mixed",
    "Quadratic",
]
