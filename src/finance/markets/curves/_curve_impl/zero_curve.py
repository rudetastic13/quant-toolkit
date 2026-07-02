"""ZeroCurve implementation with support for dual-segment interpolation."""
import warnings
import numpy as np
from finance.dates import Term
from finance.markets.curves._curve_impl.interpolators import (
    _dates_to_floats,
    DateArray,
    FloatArray,
    InterpolatedSegment,
    CurveInterpolator,
)


class ZeroCurve:
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
        interpolation: CurveInterpolator = CurveInterpolator.LogLinearDF,
        *,
        interpolation_long: CurveInterpolator | None = None,
        interpolation_cutover: Term | np.datetime64 | None = None,
        alpha: float = 0.0,
        beta: float = 0.0,
        alpha_long: float | None = None,
        beta_long: float | None = None,
        spline_knot_sequence: DateArray | None = None,
    ) -> None:
        if node_dates.ndim != 1 or len(node_dates) < 2:
            raise ValueError("node_dates must be a 1-D array with at least 2 elements.")
        if node_values.shape != node_dates.shape:
            raise ValueError("node_dates and node_values must have the same length.")
        if np.any(node_dates[:-1] > node_dates[1:]):
            raise ValueError("node_dates must be pre-sorted in ascending order.")

        self._origin: np.datetime64 = node_dates[0]
        self._origin_ord = self._origin.astype(np.int64)
        self._node_dates: DateArray = node_dates
        self._node_dfs: FloatArray = node_values

        if abs(self._node_dfs[0] - 1.0) > 1e-12:
            raise ValueError(
                "The first node (origin / today) must have DF = 1.0."
            )

        self._x_all: FloatArray = _dates_to_floats(self._node_dates, self._origin_ord)
        self._interp_type = interpolation
        self._interp_type_long = interpolation_long
        self._alpha = alpha
        self._beta = beta
        self._alpha_long = alpha_long if alpha_long is not None else alpha
        self._beta_long = beta_long if beta_long is not None else beta
        self._spline_knot_sequence = spline_knot_sequence

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
            self._cutover_x = cutover_date.astype(np.int64) - self._origin_ord

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

    @property
    def interpolation(self) -> CurveInterpolator:
        """The curve's (short-segment) interpolation scheme."""
        return self._interp_type

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
        xs = _dates_to_floats(dates, self._origin_ord)
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
        return self._rate_xs(_dates_to_floats(dates, self._origin_ord))

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
        xs1 = _dates_to_floats(starts, self._origin_ord)
        xs2 = _dates_to_floats(ends, self._origin_ord)
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
        seg_info = " + ".join(s.interp_type.name for s in self._segments)
        return (
            f"Curve(origin={self._origin}, "
            f"nodes={len(self._node_dates)}, "
            f"interpolation=[{seg_info}], "
            f"max_date={self._node_dates[-1]})"
        )
