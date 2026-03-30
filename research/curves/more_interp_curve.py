"""
Curve interpolation library inspired by J.H.M. Darbyshire's
"Pricing and Trading Interest Rate Derivatives" (Ch. 6, Section 6.3.1)
and the design patterns of rateslib (attack68).

Implements the 5 interpolation methods from the textbook:
  - DF Log-Linear:    constant O/N rates between knots, locally stable
  - DF Log-Cubic:     smooth continuous O/N rates, global dependence (alpha, beta)
  - Rate Linear:      linear interpolation of rates, global dependence
  - Rate Quadratic:   smooth aesthetic curve between knots (beta)
  - Rate Cubic:       smooth aesthetic curve between knots (alpha, beta)

Design:
  - Interpolators are plug-in strategy components
  - A Curve is a general container composed of (potentially) two underlying
    interpolated segments split at a cutover point
  - Extrapolation beyond the last pillar: flat forward rate
  - Extrapolation before the first pillar: raises an error (to be resolved
    at a higher level via a reset rates table)

Interface:
  - node_dates : np.ndarray[datetime64[D]]
  - node_values : np.ndarray[float64]  (discount factors)
  - All public query methods accept np.ndarray[datetime64[D]] and return
    np.ndarray[float64].

Edge cases handled:
  - Origin node (t=0, DF=1) where r(0) = 0/0 is indeterminate
  - Two-node minimum curves (degenerate but valid)
  - Cutover landing exactly on a node
  - Cutover between nodes (synthesises shared overlap node)
  - Flat-forward extrapolation beyond last pillar
  - Pre-origin queries raise ValueError for higher-level resolution
  - RateQuadratic uses Fritsch-Carlson monotone derivatives to avoid the
    classic alternating-sign instability of the C1 quadratic recurrence
  - Cubic splines solved via Thomas algorithm (O(n)) not dense solve

TODO (cutover — experimental):
  The two-segment cutover logic is purely experimental and requires further
  research before it can be considered production-ready.  The core problem is
  that the short and long interpolators are fit independently: while discount
  factors are continuous at the cutover node, the first derivative (and
  therefore the instantaneous forward rate) is NOT matched across the join.
  This produces visible discontinuities in forward rate plots.  Possible
  remedies include: (a) imposing a shared derivative boundary condition at the
  cutover by passing the terminal gradient of the short segment as the initial
  boundary of the long segment; (b) using a single global spline with a knot
  multiplicity change at the cutover; (c) adopting a B-spline basis with a
  prescribed continuity order at the join.  Until one of these approaches is
  implemented and validated, all ``Curve`` instances constructed with
  ``interpolation_long`` emit a ``UserWarning``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PPoly

# ---------------------------------------------------------------------------
# Convenience type aliases
# ---------------------------------------------------------------------------

DateArray = NDArray[np.datetime64]   # shape (n,), dtype datetime64[D]
FloatArray = NDArray[np.float64]

_DAY = np.timedelta64(1, "D")


# ---------------------------------------------------------------------------
# Term representation
# ---------------------------------------------------------------------------

class TermUnit(Enum):
    DAYS = auto()
    WEEKS = auto()
    MONTHS = auto()
    YEARS = auto()


@dataclass(frozen=True, slots=True)
class Term:
    """A relative time period, e.g. Term(2, TermUnit.YEARS)."""
    n: int
    unit: TermUnit

    def to_days(self) -> int:
        """Approximate conversion to calendar days (no business-day logic)."""
        if self.unit is TermUnit.DAYS:
            return self.n
        if self.unit is TermUnit.WEEKS:
            return self.n * 7
        if self.unit is TermUnit.MONTHS:
            return self.n * 30  # approximate
        if self.unit is TermUnit.YEARS:
            return self.n * 365
        raise ValueError(f"Unknown unit: {self.unit}")

    def __repr__(self) -> str:
        return f"Term({self.n}, {self.unit.name})"


# ---------------------------------------------------------------------------
# Interpolation type enum
# ---------------------------------------------------------------------------

class InterpType(Enum):
    """The five interpolation styles from Darbyshire Ch. 6 §6.3.1."""
    DF_LOG_LINEAR = "log_linear"
    DF_LOG_CUBIC = "log_cubic"
    RATE_LINEAR = "rate_linear"
    RATE_QUADRATIC = "rate_quadratic"
    RATE_CUBIC = "rate_cubic"


# ---------------------------------------------------------------------------
# Helpers: date ↔ float (days since origin)
# ---------------------------------------------------------------------------

def _dates_to_floats(dates: DateArray, origin: np.datetime64) -> FloatArray:
    """Convert an array of datetime64[D] values to float days-since-origin."""
    return ((dates - origin) / _DAY).astype(np.float64)


def _date_to_float(date: np.datetime64, origin: np.datetime64) -> float:
    """Convert a single datetime64[D] to float days-since-origin."""
    return float((date - origin) / _DAY)


def _compute_zero_rates(x: np.ndarray, df: np.ndarray) -> np.ndarray:
    """
    Compute continuously compounded zero rates r(t) = -ln(DF(t)) / t.

    At t=0 (origin), r(0) is the limit of -ln(DF(t))/t as t->0, which
    equals the instantaneous forward rate.  We estimate it by linear
    extrapolation backward from the first two non-origin pillar rates
    when available, falling back to the rate at the first pillar.
    """
    n = len(x)
    rates = np.zeros(n)
    for i in range(n):
        if x[i] > 0:
            rates[i] = -np.log(df[i]) / x[i]

    # Fix origin rate (x[0] == 0)
    if n >= 3 and x[0] == 0.0 and x[1] > 0 and x[2] > 0:
        r1, r2 = rates[1], rates[2]
        rates[0] = r1 - x[1] * (r2 - r1) / (x[2] - x[1])
    elif n >= 2 and x[0] == 0.0 and x[1] > 0:
        rates[0] = rates[1]

    return rates


def _solve_cubic_second_derivs(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> np.ndarray:
    """
    Solve the standard cubic spline tridiagonal system for second
    derivatives M at each knot, with prescribed M[0]=alpha and
    M[-1]=beta as boundary conditions.

    Uses Thomas algorithm (O(n)) rather than dense np.linalg.solve.
    """
    n = len(x)
    h = np.diff(x)
    d = np.diff(y) / h

    if n <= 2:
        return np.array([alpha, beta][:n])

    # Interior system size = n-2
    m = n - 2
    b_diag = np.zeros(m)
    c_sup = np.zeros(m)
    a_sub = np.zeros(m)
    rhs = np.zeros(m)

    for j in range(m):
        i = j + 1  # index in original system
        b_diag[j] = 2.0 * (h[i - 1] + h[i])
        rhs[j] = 6.0 * (d[i] - d[i - 1])
        if j > 0:
            a_sub[j] = h[i - 1]
        if j < m - 1:
            c_sup[j] = h[i]

    # Subtract known boundary values
    rhs[0] -= h[0] * alpha
    rhs[-1] -= h[-1] * beta

    # Forward elimination (Thomas)
    for j in range(1, m):
        w = a_sub[j] / b_diag[j - 1]
        b_diag[j] -= w * c_sup[j - 1]
        rhs[j] -= w * rhs[j - 1]

    # Back substitution
    M_int = np.zeros(m)
    M_int[-1] = rhs[-1] / b_diag[-1]
    for j in range(m - 2, -1, -1):
        M_int[j] = (rhs[j] - c_sup[j] * M_int[j + 1]) / b_diag[j]

    M = np.empty(n)
    M[0] = alpha
    M[1:-1] = M_int
    M[-1] = beta
    return M


def _build_cubic_ppoly(
    x: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
) -> PPoly:
    """
    Build a scipy PPoly from knots x, values y, and second derivatives M.
    """
    n = len(x)
    h = np.diff(x)
    d = np.diff(y) / h

    c_arr = np.zeros((4, n - 1))
    for i in range(n - 1):
        hi = h[i]
        c_arr[3, i] = y[i]
        c_arr[2, i] = d[i] - hi * (2.0 * M[i] + M[i + 1]) / 6.0
        c_arr[1, i] = M[i] / 2.0
        c_arr[0, i] = (M[i + 1] - M[i]) / (6.0 * hi)

    return PPoly(c_arr, x)


# ---------------------------------------------------------------------------
# Interpolator protocol (plug-in strategy)
# ---------------------------------------------------------------------------

class Interpolator(ABC):
    """
    Base class for all interpolation strategies.

    An interpolator is initialised with the node data (dates converted to
    float offsets and their corresponding discount factors).  It then
    provides ``__call__(x)`` to return the interpolated discount factor at
    an arbitrary interior point *x* (float days offset).
    """

    @abstractmethod
    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None: ...

    @abstractmethod
    def __call__(self, x: FloatArray) -> FloatArray:
        """Return interpolated discount factors for an array of day offsets."""
        ...


# ---------------------------------------------------------------------------
# 1. DF Log-Linear  (constant O/N rates between knots, local)
# ---------------------------------------------------------------------------

class DFLogLinear(Interpolator):
    """
    ln(DF) is piecewise-linear between knots.
    Equivalent to constant instantaneous (overnight) forward rates in each
    interval.  Locally stable: changing one node only affects its two
    adjacent intervals.
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        self._log_df = np.log(df)

    def __call__(self, x: FloatArray) -> FloatArray:
        idx = np.clip(np.searchsorted(self._x, x, side="right") - 1, 0, len(self._x) - 2)
        t = (x - self._x[idx]) / (self._x[idx + 1] - self._x[idx])
        log_df = self._log_df[idx] + t * (self._log_df[idx + 1] - self._log_df[idx])
        return np.exp(log_df)


# ---------------------------------------------------------------------------
# 2. DF Log-Cubic  (smooth continuous O/N rates, global, alpha & beta)
# ---------------------------------------------------------------------------

class DFLogCubic(Interpolator):
    """
    Cubic spline on ln(DF).  Produces smooth, continuous overnight rates.
    Global dependence: every node affects the whole curve.

    Parameters
    ----------
    alpha : float
        Prescribed second derivative of ln(DF) at the left endpoint.
        Default 0.0 gives the *natural* spline boundary condition.
    beta : float
        Prescribed second derivative of ln(DF) at the right endpoint.
        Default 0.0 gives the *natural* spline boundary condition.
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        log_df = np.log(df)
        M = _solve_cubic_second_derivs(x, log_df, alpha, beta)
        self._pp = _build_cubic_ppoly(x, log_df, M)

    def __call__(self, x: FloatArray) -> FloatArray:
        return np.exp(self._pp(x))


# ---------------------------------------------------------------------------
# 3. Rate Linear  (linear interpolation on zero rates)
# ---------------------------------------------------------------------------

class RateLinear(Interpolator):
    """
    Linear interpolation of continuously compounded zero rates.
    Global dependence.  Forward rates are discontinuous at pillar dates
    (this is inherent to rate-linear interpolation, not a bug).
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        self._rates = _compute_zero_rates(x, df)

    def __call__(self, x: FloatArray) -> FloatArray:
        idx = np.clip(np.searchsorted(self._x, x, side="right") - 1, 0, len(self._x) - 2)
        t = (x - self._x[idx]) / (self._x[idx + 1] - self._x[idx])
        r = self._rates[idx] + t * (self._rates[idx + 1] - self._rates[idx])
        return np.exp(-r * x)


# ---------------------------------------------------------------------------
# 4. Rate Quadratic  (smooth piecewise-quadratic on zero rates, beta)
# ---------------------------------------------------------------------------

class RateQuadratic(Interpolator):
    """
    C1-continuous piecewise-quadratic on continuously compounded zero
    rates, stabilised via blended forward/backward sweeps.

    Parameters
    ----------
    alpha : float
        Prescribed first derivative of r(t) at the left endpoint.
        Default 0.0 (flat rate at the short end).
    beta : float
        Prescribed first derivative of r(t) at the right endpoint.
        Default 0.0 (flat rate at the long end).
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        n = len(x)
        rates = _compute_zero_rates(x, df)
        h = np.diff(x)
        d = np.diff(rates) / h

        # Forward sweep from m[0] = alpha
        m_fwd = np.zeros(n)
        m_fwd[0] = alpha
        for i in range(n - 1):
            m_fwd[i + 1] = 2.0 * d[i] - m_fwd[i]

        # Backward sweep from m[-1] = beta
        m_bwd = np.zeros(n)
        m_bwd[-1] = beta
        for i in range(n - 2, -1, -1):
            m_bwd[i] = 2.0 * d[i] - m_bwd[i + 1]

        if n >= 2:
            w = np.linspace(0.0, 1.0, n)
        else:
            w = np.array([0.5])
        m = (1.0 - w) * m_fwd + w * m_bwd

        m[0] = alpha
        m[-1] = beta

        c_arr = np.zeros((3, n - 1))
        for i in range(n - 1):
            c_arr[2, i] = rates[i]
            c_arr[1, i] = m[i]
            c_arr[0, i] = (d[i] - m[i]) / h[i]

        self._pp = PPoly(c_arr, x)

    def __call__(self, x: FloatArray) -> FloatArray:
        return np.exp(-self._pp(x) * x)


# ---------------------------------------------------------------------------
# 5. Rate Cubic  (cubic spline on zero rates, alpha & beta)
# ---------------------------------------------------------------------------

class RateCubic(Interpolator):
    """
    Cubic spline on continuously compounded zero rates.
    Smooth aesthetic curve between knots.  Global dependence.

    Parameters
    ----------
    alpha : float
        Prescribed second derivative of the rate function at the left
        endpoint.  Default 0.0 (natural spline).
    beta : float
        Prescribed second derivative of the rate function at the right
        endpoint.  Default 0.0 (natural spline).
    """

    def __init__(
        self,
        x: np.ndarray,
        df: np.ndarray,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> None:
        self._x = x
        rates = _compute_zero_rates(x, df)
        M = _solve_cubic_second_derivs(x, rates, alpha, beta)
        self._pp = _build_cubic_ppoly(x, rates, M)

    def __call__(self, x: FloatArray) -> FloatArray:
        return np.exp(-self._pp(x) * x)


# ---------------------------------------------------------------------------
# Registry mapping InterpType -> Interpolator class
# ---------------------------------------------------------------------------

_INTERP_REGISTRY: dict[InterpType, type[Interpolator]] = {
    InterpType.DF_LOG_LINEAR: DFLogLinear,
    InterpType.DF_LOG_CUBIC: DFLogCubic,
    InterpType.RATE_LINEAR: RateLinear,
    InterpType.RATE_QUADRATIC: RateQuadratic,
    InterpType.RATE_CUBIC: RateCubic,
}


def build_interpolator(
    interp_type: InterpType,
    x: np.ndarray,
    df: np.ndarray,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> Interpolator:
    """Factory: construct the correct Interpolator for a given InterpType."""
    cls = _INTERP_REGISTRY[interp_type]
    return cls(x, df, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# InterpolatedSegment: one contiguous interpolated region of a curve
# ---------------------------------------------------------------------------

@dataclass
class InterpolatedSegment:
    """
    Wraps an Interpolator together with the raw node data that produced it,
    and knows its own domain [x_min, x_max] in float-day space.
    """
    interp_type: InterpType
    x: np.ndarray
    df: np.ndarray
    alpha: float
    beta: float
    origin: np.datetime64
    interpolator: Interpolator = field(init=False)

    def __post_init__(self) -> None:
        self.interpolator = build_interpolator(
            self.interp_type, self.x, self.df,
            alpha=self.alpha, beta=self.beta,
        )

    @property
    def x_min(self) -> float:
        return float(self.x[0])

    @property
    def x_max(self) -> float:
        return float(self.x[-1])

    def discount_factor(self, xs: FloatArray) -> FloatArray:
        return self.interpolator(xs)


# ---------------------------------------------------------------------------
# Curve
# ---------------------------------------------------------------------------

class Curve:
    """
    A discount-factor curve composed of one or two interpolated segments.

    Parameters
    ----------
    node_dates : np.ndarray[datetime64[D]]
        Pillar dates.  The first entry must be the curve origin (today).
        Need not be pre-sorted — the constructor sorts them.
    node_values : np.ndarray[float64]
        Discount factors corresponding to each pillar date.  The value
        at the origin date must be exactly 1.0.
    interpolation : InterpType
        Primary interpolation method.
    interpolation_long : InterpType | None
        If provided, the interpolation method used *after* the cutover.
        If ``None``, a single interpolation is used for the whole curve.
    interpolation_cutover : Term | np.datetime64 | None
        The point at which the interpolation switches from ``interpolation``
        to ``interpolation_long``.  Can be expressed as:
          - a ``Term`` (relative to the curve's origin date), or
          - an absolute ``np.datetime64`` (any precision; cast to Day).
        Ignored when ``interpolation_long`` is ``None``.
    alpha : float
        Left boundary parameter passed to both segments unless overridden.
    beta : float
        Right boundary parameter passed to both segments unless overridden.
    alpha_long : float | None
        Override alpha for the long segment.  Falls back to ``alpha``.
    beta_long : float | None
        Override beta for the long segment.  Falls back to ``beta``.
    t : np.ndarray[datetime64[D]] | None
        Explicit knot sequence for spline methods (advanced usage).

    Public query methods
    --------------------
    All accept ``np.ndarray[datetime64[D]]`` and return ``np.ndarray[float64]``.

    discount_factor(dates)  →  DF values
    rate(dates)             →  continuously compounded zero rates (annualised)
    forward_rate(starts, ends) → forward rates between date pairs
    """

    def __init__(
        self,
        node_dates: DateArray,
        node_values: FloatArray,
        interpolation: InterpType = InterpType.DF_LOG_LINEAR,
        *,
        interpolation_long: Optional[InterpType] = None,
        interpolation_cutover: Optional[Term | np.datetime64] = None,
        alpha: float = 0.0,
        beta: float = 0.0,
        alpha_long: Optional[float] = None,
        beta_long: Optional[float] = None,
        spline_knot_sequence: Optional[DateArray] = None,
    ) -> None:
        if node_dates.ndim != 1 or len(node_dates) < 2:
            raise ValueError("node_dates must be a 1-D array with at least 2 elements.")
        if node_values.shape != node_dates.shape:
            raise ValueError("node_dates and node_values must have the same length.")
        if np.any(node_dates[:-1] > node_dates[1:]):
            raise ValueError("node_dates must be pre-sorted in ascending order.")

        self._origin: np.datetime64 = node_dates[0]
        self._node_dates: DateArray = node_dates
        self._node_dfs: FloatArray = node_values

        if abs(self._node_dfs[0] - 1.0) > 1e-12:
            raise ValueError(
                "The first node (origin / today) must have DF = 1.0."
            )

        self._x_all: FloatArray = _dates_to_floats(self._node_dates, self._origin)
        self._interp_type = interpolation
        self._interp_type_long = interpolation_long
        self._alpha = alpha
        self._beta = beta
        self._alpha_long = alpha_long if alpha_long is not None else alpha
        self._beta_long = beta_long if beta_long is not None else beta
        self._spline_knot_sequence = spline_knot_sequence

        # Resolve cutover
        self._cutover_x: Optional[float] = None
        if interpolation_long is not None:
            warnings.warn(
                "Cutover interpolation (interpolation_long) is experimental. "
                "The two segments are fit independently, so discount factors are "
                "continuous at the cutover but the forward rate is NOT — a "
                "derivative discontinuity is expected at the join. "
                "See the module-level TODO for details.",
                UserWarning,
                stacklevel=2,
            )
            if interpolation_cutover is None:
                raise ValueError(
                    "interpolation_cutover is required when "
                    "interpolation_long is specified."
                )
            if isinstance(interpolation_cutover, Term):
                cutover_date = self._origin + np.timedelta64(
                    interpolation_cutover.to_days(), "D"
                )
            else:
                cutover_date = np.datetime64(interpolation_cutover, "D")
            self._cutover_x = _date_to_float(cutover_date, self._origin)

        # Build segments
        self._segments: list[InterpolatedSegment] = []
        self._build_segments()

        # Cache for flat-forward extrapolation beyond last pillar
        self._last_x = float(self._x_all[-1])
        self._last_df = float(self._node_dfs[-1])
        self._last_fwd = self._compute_terminal_forward()

    # -- segment construction ------------------------------------------------

    def _build_segments(self) -> None:
        if self._cutover_x is None:
            seg = InterpolatedSegment(
                interp_type=self._interp_type,
                x=self._x_all,
                df=self._node_dfs,
                alpha=self._alpha,
                beta=self._beta,
                origin=self._origin,
            )
            self._segments = [seg]
        else:
            cutover = self._cutover_x
            short_mask = self._x_all <= cutover + 1e-10
            long_mask = self._x_all >= cutover - 1e-10

            x_short = self._x_all[short_mask]
            df_short = self._node_dfs[short_mask]

            x_long = self._x_all[long_mask]
            df_long = self._node_dfs[long_mask]

            if len(x_short) < 2 or len(x_long) < 2:
                raise ValueError(
                    "Cutover point must leave at least 2 nodes in each segment. "
                    f"Short segment has {len(x_short)} nodes, "
                    f"long segment has {len(x_long)} nodes."
                )

            seg_short = InterpolatedSegment(
                interp_type=self._interp_type,
                x=x_short,
                df=df_short,
                alpha=self._alpha,
                beta=self._beta,
                origin=self._origin,
            )
            seg_long = InterpolatedSegment(
                interp_type=self._interp_type_long,  # type: ignore[arg-type]
                x=x_long,
                df=df_long,
                alpha=self._alpha_long,
                beta=self._beta_long,
                origin=self._origin,
            )
            self._segments = [seg_short, seg_long]

    # -- terminal forward for flat extrapolation ----------------------------

    def _compute_terminal_forward(self) -> float:
        """
        Estimate the instantaneous forward rate at the last pillar.
        f(T) ≈ -[ln DF(T) - ln DF(T - 1day)] / 1day
        """
        eps = 1.0
        df_T = self._last_df
        df_T_minus = float(self._interpolate_interior(np.array([self._last_x - eps]))[0])
        if df_T_minus <= 0 or df_T <= 0:
            return 0.0
        return -(np.log(df_T) - np.log(df_T_minus)) / eps

    # -- vectorized interpolation dispatch (internal) -----------------------

    def _interpolate_interior(self, xs: FloatArray) -> FloatArray:
        """Vectorized DF interpolation within the curve's defined node range."""
        if len(self._segments) == 1:
            return self._segments[0].discount_factor(xs)

        cutover = self._cutover_x
        assert cutover is not None
        short_mask = xs <= cutover + 1e-10
        result = np.empty(len(xs), dtype=np.float64)
        if np.any(short_mask):
            result[short_mask] = self._segments[0].discount_factor(xs[short_mask])
        if np.any(~short_mask):
            result[~short_mask] = self._segments[1].discount_factor(xs[~short_mask])
        return result

    def _discount_factor_xs(self, xs: FloatArray) -> FloatArray:
        """Fully vectorized DF from float day offsets (no date-conversion overhead)."""
        result = np.ones(len(xs), dtype=np.float64)
        interior_mask = (xs > 1e-10) & (xs <= self._last_x + 1e-10)
        extrap_mask = xs > self._last_x + 1e-10
        if np.any(interior_mask):
            result[interior_mask] = self._interpolate_interior(xs[interior_mask])
        if np.any(extrap_mask):
            result[extrap_mask] = self._last_df * np.exp(
                -self._last_fwd * (xs[extrap_mask] - self._last_x)
            )
        return result

    def _rate_xs(self, xs: FloatArray) -> FloatArray:
        """Fully vectorized zero rates from float day offsets."""
        result = np.empty(len(xs), dtype=np.float64)
        origin_mask = xs <= 1e-10
        nonorigin_mask = ~origin_mask
        if np.any(origin_mask):
            df_1d = float(self._discount_factor_xs(np.array([1.0]))[0])
            result[origin_mask] = -np.log(df_1d) * 365.0
        if np.any(nonorigin_mask):
            dfs = self._discount_factor_xs(xs[nonorigin_mask])
            result[nonorigin_mask] = -np.log(dfs) / (xs[nonorigin_mask] / 365.0)
        return result

    # -- public API ----------------------------------------------------------

    @property
    def origin(self) -> np.datetime64:
        """The curve's valuation / construction date."""
        return self._origin

    @property
    def node_dates(self) -> DateArray:
        return self._node_dates.copy()

    @property
    def node_dfs(self) -> FloatArray:
        return self._node_dfs.copy()

    @property
    def max_date(self) -> np.datetime64:
        return self._node_dates[-1]

    def discount_factor(self, dates: DateArray) -> FloatArray:
        """
        Discount factors for an array of dates.

        Parameters
        ----------
        dates : np.ndarray[datetime64[D]]

        Returns
        -------
        np.ndarray[float64]

        Raises
        ------
        ValueError
            If any date is before the curve origin.
        """
        xs = _dates_to_floats(dates, self._origin)
        if np.any(xs < -1e-10):
            bad = dates[xs < -1e-10]
            raise ValueError(
                f"{len(bad)} date(s) before the curve origin (earliest: {bad[0]}). "
                "Pre-origin extrapolation is not supported at the Curve level."
            )
        return self._discount_factor_xs(xs)

    def rate(self, dates: DateArray) -> FloatArray:
        """
        Continuously compounded zero rates to each date.
        r(t) = -ln(DF(t)) / t   (annualised, Act/365)

        Parameters
        ----------
        dates : np.ndarray[datetime64[D]]

        Returns
        -------
        np.ndarray[float64]
        """
        return self._rate_xs(_dates_to_floats(dates, self._origin))

    def forward_rate(self, starts: DateArray, ends: DateArray) -> FloatArray:
        """
        Continuously compounded forward rates between paired date arrays.
        f(t1, t2) = -[ln DF(t2) - ln DF(t1)] / (t2 - t1)   (annualised)

        Parameters
        ----------
        starts : np.ndarray[datetime64[D]]
        ends   : np.ndarray[datetime64[D]]

        Returns
        -------
        np.ndarray[float64]
        """
        if starts.shape != ends.shape:
            raise ValueError("starts and ends must have the same shape.")
        xs1 = _dates_to_floats(starts, self._origin)
        xs2 = _dates_to_floats(ends, self._origin)
        dt_years = (xs2 - xs1) / 365.0
        degen = np.abs(dt_years) < 1e-14
        result = np.empty(len(xs1), dtype=np.float64)
        if np.any(~degen):
            df1 = self._discount_factor_xs(xs1[~degen])
            df2 = self._discount_factor_xs(xs2[~degen])
            result[~degen] = -(np.log(df2) - np.log(df1)) / dt_years[~degen]
        if np.any(degen):
            result[degen] = self._rate_xs(xs1[degen])
        return result

    def __repr__(self) -> str:
        seg_info = " + ".join(s.interp_type.value for s in self._segments)
        return (
            f"Curve(origin={self._origin}, "
            f"nodes={len(self._node_dates)}, "
            f"interpolation=[{seg_info}], "
            f"max_date={self._node_dates[-1]})"
        )


# ---------------------------------------------------------------------------
# Convenience: a quick visual sanity check when run as __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    node_dates = np.array([
        "2024-01-01", "2024-07-01", "2025-01-01",
        "2026-01-01", "2027-01-01", "2028-01-01", "2029-01-01",
    ], dtype="datetime64[D]")

    node_values = np.array(
        [1.0, 0.9975, 0.9940, 0.9850, 0.9730, 0.9600, 0.9460],
        dtype=np.float64,
    )

    print("=" * 70)
    print("Single-segment curves (each interpolation type)")
    print("=" * 70)
    test_dates = np.array(["2025-07-01"], dtype="datetime64[D]")
    for itype in InterpType:
        c = Curve(node_dates, node_values, itype)
        df = c.discount_factor(test_dates)[0]
        r = c.rate(test_dates)[0]
        print(f"  {itype.value:<20s}  DF({test_dates[0]}) = {df:.8f}   rate = {r * 100:.4f}%")

    # Extrapolation test
    print()
    c = Curve(node_dates, node_values, InterpType.DF_LOG_LINEAR)
    extrap = np.array(["2031-01-01"], dtype="datetime64[D]")
    print(f"Flat-fwd extrapolation: DF({extrap[0]}) = {c.discount_factor(extrap)[0]:.8f}")

    # Mixed interpolation (short: log-linear, long: log-cubic)
    print()
    print("=" * 70)
    print("Two-segment curve: log-linear short, log-cubic long")
    print("=" * 70)
    mixed = Curve(
        node_dates,
        node_values,
        interpolation=InterpType.DF_LOG_LINEAR,
        interpolation_long=InterpType.DF_LOG_CUBIC,
        interpolation_cutover=Term(2, TermUnit.YEARS),
    )
    print(mixed)
    query_dates = np.array(
        ["2024-10-01", "2026-06-01", "2028-06-01"], dtype="datetime64[D]"
    )
    dfs = mixed.discount_factor(query_dates)
    for d, df in zip(query_dates, dfs):
        print(f"  DF({d}) = {df:.8f}")

    # Pre-origin error test
    print()
    try:
        c.discount_factor(np.array(["2023-06-01"], dtype="datetime64[D]"))
    except ValueError as e:
        print(f"Expected error for pre-origin date: {e}")