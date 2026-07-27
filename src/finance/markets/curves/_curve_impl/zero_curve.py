"""Pure discount-function curve with support for dual-segment interpolation.

``ZeroCurve`` deliberately has no index, fixing, coupon, or market-forward semantics.  It
owns only mathematical curve state and exposes discount factors, log discount factors, and
continuously-compounded zero rates.  Market conventions are composed by ``YieldCurve``;
rate generation lives in ``RateGenerator``.
"""
from __future__ import annotations

import warnings
import numpy as np
from finance.dates import Term
from finance.markets.curves._curve_impl.interpolators import (
    _dates_to_floats,
    DateArray,
    FloatArray,
    InterpolatedSegment,
)
from finance.markets.curves.types import CurveInterpolator, RateExtrapolator


class ZeroCurve:
    """
    A discount-factor curve composed of one or two interpolated segments.

    Parameters
    ----------
    node_dates : np.ndarray[datetime64[D]]
        Strictly increasing pillar dates.  The first entry is the curve origin.
    node_values : np.ndarray[float64]
        Discount factors corresponding to each pillar date.  The value
        at the origin date must be exactly 1.0.
    interpolation : CurveInterpolator
        Primary interpolation method.
    interpolation_long : CurveInterpolator | None
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
    extrapolation : RateExtrapolator
        Behaviour after the final pillar. ``Flat`` continues the terminal instantaneous
        forward; ``NotAllowed`` rejects the query. Pre-origin queries are always rejected.

    Public query methods
    --------------------
    All accept ``np.ndarray[datetime64[D]]`` and return ``np.ndarray[float64]``.

    discount_factor(dates)  →  DF values
    log_discount_factor(dates) → natural logarithm of DF values
    zero_rate(dates)        → continuously compounded zero rates (Act/365 annualised)
    """

    def __init__(
        self,
        node_dates: DateArray,
        node_values: FloatArray,
        interpolation: CurveInterpolator = CurveInterpolator.LogLinearDF,
        *,
        interpolation_long: CurveInterpolator | None = None,
        interpolation_cutover: Term | np.datetime64 | None = None,
        alpha: float = 0.0,
        beta: float = 0.0,
        alpha_long: float | None = None,
        beta_long: float | None = None,
        extrapolation: RateExtrapolator = RateExtrapolator.Flat,
    ) -> None:
        dates = np.asarray(node_dates).astype("datetime64[D]").copy()
        dfs = np.asarray(node_values, dtype=np.float64).copy()
        if dates.ndim != 1 or len(dates) < 2:
            raise ValueError("node_dates must be a 1-D array with at least 2 elements.")
        if dfs.shape != dates.shape:
            raise ValueError("node_dates and node_values must have the same length.")
        if np.isnat(dates).any():
            raise ValueError("node_dates cannot contain NaT.")
        if np.any(dates[:-1] >= dates[1:]):
            raise ValueError("node_dates must be strictly increasing with no duplicates.")
        if not np.isfinite(dfs).all() or np.any(dfs <= 0.0):
            raise ValueError("node_values must be finite, strictly positive discount factors.")
        if not np.isfinite([alpha, beta]).all():
            raise ValueError("alpha and beta must be finite.")
        if alpha_long is not None and not np.isfinite(alpha_long):
            raise ValueError("alpha_long must be finite when supplied.")
        if beta_long is not None and not np.isfinite(beta_long):
            raise ValueError("beta_long must be finite when supplied.")

        self._origin: np.datetime64 = dates[0]
        self._origin_ord = self._origin.astype(np.int64)
        self._node_dates: DateArray = dates
        self._node_dfs: FloatArray = dfs

        if abs(self._node_dfs[0] - 1.0) > 1e-12:
            raise ValueError(
                "The first node (origin / today) must have DF = 1.0."
            )

        self._x_all: FloatArray = _dates_to_floats(self._node_dates, self._origin_ord)
        self._interp_type = CurveInterpolator(interpolation)
        self._interp_type_long = (
            None if interpolation_long is None else CurveInterpolator(interpolation_long)
        )
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._alpha_long = float(alpha_long) if alpha_long is not None else self._alpha
        self._beta_long = float(beta_long) if beta_long is not None else self._beta
        self._extrapolation = RateExtrapolator(extrapolation)

        # Resolve cutover
        self._cutover_x: float | None = None
        if interpolation_long is not None:
            warnings.warn(
                "Cutover interpolation (interpolation_long) is experimental. "
                "The two segments are fit independently, so discount factors are "
                "continuous at the cutover but the forward rate is NOT — a "
                "derivative discontinuity is expected at the join.",
                UserWarning,
                stacklevel=2,
            )
            if interpolation_cutover is None:
                raise ValueError(
                    "interpolation_cutover is required when "
                    "interpolation_long is specified."
                )
            if isinstance(interpolation_cutover, Term):
                cutover_date = self._origin + interpolation_cutover
            else:
                cutover_date = interpolation_cutover
            cutover_date = np.datetime64(cutover_date, "D")
            if cutover_date not in self._node_dates:
                raise ValueError(
                    "interpolation_cutover must coincide with a curve node so both "
                    "segments share the same discount factor."
                )
            self._cutover_x = float(cutover_date.astype(np.int64) - self._origin_ord)
        elif interpolation_cutover is not None:
            raise ValueError("interpolation_cutover requires interpolation_long.")

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
            if self._extrapolation is RateExtrapolator.NotAllowed:
                raise ValueError("query after the final curve pillar is not allowed.")
            result[extrap_mask] = self._last_df * np.exp(
                -self._last_fwd * (xs[extrap_mask] - self._last_x)
            )
        return result

    def _zero_rate_xs(self, xs: FloatArray) -> FloatArray:
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

    @property
    def interpolation(self) -> CurveInterpolator:
        """The curve's (short-segment) interpolation scheme."""
        return self._interp_type

    @property
    def extrapolation(self) -> RateExtrapolator:
        return self._extrapolation

    def _query_xs(self, dates: DateArray) -> FloatArray:
        dates = np.asarray(dates).astype("datetime64[D]")
        if dates.ndim != 1:
            raise ValueError("curve queries require a 1-D date array.")
        if np.isnat(dates).any():
            raise ValueError("curve query dates cannot contain NaT.")
        xs = _dates_to_floats(dates, self._origin_ord)
        if np.any(xs < -1e-10):
            bad = dates[xs < -1e-10]
            raise ValueError(
                f"{len(bad)} date(s) before the curve origin (earliest: {bad[0]}). "
                "Pre-origin extrapolation is not supported at the Curve level."
            )
        return xs

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
        return self._discount_factor_xs(self._query_xs(dates))

    def log_discount_factor(self, dates: DateArray) -> FloatArray:
        """Natural logarithm of the discount factor at each date."""
        return np.log(self._discount_factor_xs(self._query_xs(dates)))

    def zero_rate(self, dates: DateArray) -> FloatArray:
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
        return self._zero_rate_xs(self._query_xs(dates))

    def with_node_dfs(self, node_dfs: FloatArray) -> ZeroCurve:
        """Return the same curve geometry/configuration with new discount-factor state."""
        cutover = None
        if self._cutover_x is not None:
            cutover = self._origin + np.timedelta64(int(self._cutover_x), "D")
        return ZeroCurve(
            self._node_dates,
            node_dfs,
            interpolation=self._interp_type,
            interpolation_long=self._interp_type_long,
            interpolation_cutover=cutover,
            alpha=self._alpha,
            beta=self._beta,
            alpha_long=self._alpha_long,
            beta_long=self._beta_long,
            extrapolation=self._extrapolation,
        )

    def __repr__(self) -> str:
        seg_info = " + ".join(s.interp_type.name for s in self._segments)
        return (
            f"Curve(origin={self._origin}, "
            f"nodes={len(self._node_dates)}, "
            f"interpolation=[{seg_info}], "
            f"max_date={self._node_dates[-1]})"
        )
