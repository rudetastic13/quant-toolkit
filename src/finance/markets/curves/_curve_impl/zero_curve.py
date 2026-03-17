"""
ZeroCurve implementation using log discount factors as the internal representation.

Design decisions:
- Internally stores log discount factors: -ln(df_t) for numerical stability and natural interpolation
- Supports construction from discount factors, log discount factors, or zero rates
- Uses Curve1d under the hood for interpolation/extrapolation
- Accepts datetime64[D] arrays and converts to integer offsets from curve_date
- Extensible for future composite/spread curve support
"""
from __future__ import annotations
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray

from src.finance.dates.date import Date
from common.containers.curve1d import (
    Curve1d,
    InterpolationType,
    ExtrapolationType,
    IntArray,
    FloatArray,
)
from src.finance.markets.curves.types import (
    ValueType,
    CurveInterpolator,
    CurveExtrapolator,
)

# Type aliases
DateArray = NDArray[np.datetime64]

# Constants
_DAYS_PER_YEAR = 365  # ACT/365.25 for now; day count convention can be abstracted later

def _log_to_zero(offsets: IntArray, values: FloatArray) -> FloatArray:
    """
    Convert log discount factors to zero rates: r_t.

    Parameters
    ----------
    offsets : IntArray
        Day offsets from curve date (t=0 at curve date)
    values : FloatArray
        Log discount factors: -ln(df_t)
    Returns
        Zero rates: r_t
    """
    result = np.divide(values *_DAYS_PER_YEAR, offsets, where=offsets != 0)
    result[offsets] = 0.0  # Define r(0) = 0
    return result

def _log_to_df(_: Any, values: FloatArray) -> FloatArray:
    """
    Convert log discount factors to discount factors: df_t.

    Parameters
    ----------
    _ : Any
        Day offsets from curve date (t=0 at curve date)
    values : FloatArray
        Log discount factors: -ln(df_t)

    Returns
    -------
    FloatArray
        Discount factors: df_t
    """
    return np.exp(-values)

def _log_to_log(_: Any, values: FloatArray) -> FloatArray: return values

def _df_to_log(_: Any, values: FloatArray) -> FloatArray:
    return -np.log(values)

def _df_to_zero(offsets: IntArray, values: FloatArray) -> FloatArray:
    log_dfs = _df_to_log(None, values)
    return _log_to_zero(offsets, log_dfs)

def _df_to_df(_: Any, values: FloatArray) -> FloatArray: return values

def _zero_to_log(offsets: IntArray, values: FloatArray) -> FloatArray:
    t = offsets.astype(np.float64) / _DAYS_PER_YEAR
    return values * t

def _zero_to_df(offsets: IntArray, values: FloatArray) -> FloatArray:
    log_dfs = _zero_to_log(offsets, values)
    return _log_to_df(None, log_dfs)

def _zero_to_zero(_: Any, values: FloatArray) -> FloatArray: return values

_value_transformer: dict[tuple[ValueType, ValueType], Callable] =  {
    (ValueType.LogDiscountFactor, ValueType.LogDiscountFactor): _log_to_log,
    (ValueType.LogDiscountFactor, ValueType.DiscountFactor): _log_to_df,
    (ValueType.LogDiscountFactor, ValueType.ZeroRate): _log_to_zero,
    (ValueType.ZeroRate, ValueType.DiscountFactor): _zero_to_df,
    (ValueType.ZeroRate, ValueType.LogDiscountFactor): _zero_to_log,
    (ValueType.ZeroRate, ValueType.ZeroRate): _zero_to_zero,
    (ValueType.DiscountFactor, ValueType.DiscountFactor): _df_to_df,
    (ValueType.DiscountFactor, ValueType.LogDiscountFactor): _df_to_log,
    (ValueType.DiscountFactor, ValueType.ZeroRate): _df_to_zero,
}

def _map_interpolator(interp: CurveInterpolator) -> InterpolationType:
    """Map high-level curve interpolator to Curve1d interpolation type."""
    # For log discount factors, linear interpolation in log space
    # is equivalent to log-linear interpolation in DF space
    if interp in (CurveInterpolator.Linear, CurveInterpolator.LogLinear):
        return InterpolationType.Linear
    elif interp in (CurveInterpolator.FlatForward, CurveInterpolator.FlatZero):
        return InterpolationType.Flat
    else:
        # Default to linear; cubic spline would need additional implementation
        return InterpolationType.Linear


def _map_extrapolator(extrap: CurveExtrapolator) -> ExtrapolationType:
    """Map high-level curve extrapolator to Curve1d extrapolation type."""
    if extrap == CurveExtrapolator.NotAllowed:
        return ExtrapolationType.NotAllowed
    elif extrap == CurveExtrapolator.FlatForward:
        # Flat extrapolation in log DF space = flat forward rate
        return ExtrapolationType.Flat
    else:
        return ExtrapolationType.NotAllowed

def _map_transform(interp: CurveInterpolator, value_type: ValueType) -> tuple[ValueType, ValueType]:
    if interp in {CurveInterpolator.LinearZero, CurveInterpolator.FlatZero}:
        return value_type, ValueType.ZeroRate
    elif interp in {CurveInterpolator.LinearLogDF, CurveInterpolator.FlatZero}:
        return value_type, ValueType.LogDiscountFactor


class ZeroCurve:
    """
    A zero curve that internally stores log discount factors for interpolation.

    The curve stores -ln(df_t) and provides methods to retrieve:
    - Discount factors: df(t)
    - Zero rates: r(t)
    - Log discount factors: -ln(df(t))

    Parameters
    ----------
    curve_date : Date
        The anchor date for the curve (t=0)
    offsets : np.ndarray[np.int64]
        Day offsets from curve_date for each curve point
    values : np.ndarray[np.float64]
        Values at each offset (interpretation depends on value_type)
    value_type : ValueType
        How to interpret the input values (DiscountFactor, LogDiscountFactor, ZeroRate)
    interp : CurveInterpolator
        Interpolation method for the curve
    extrap : CurveExtrapolator
        Extrapolation method for the curve

    Notes
    -----
    - Internally stores log discount factors: -ln(df_t)
    - Linear interpolation in log DF space = log-linear interpolation in DF space
    - This is equivalent to piecewise constant forward rates between nodes
    - Supports datetime64[D] arrays which are converted to offsets from curve_date
    """

    def __init__(
        self,
        curve_date: Date,
        dates: DateArray,
        values: FloatArray,
        value_type: ValueType,
        interp: CurveInterpolator = CurveInterpolator.LinearLogDF,
        extrap: CurveExtrapolator = CurveExtrapolator.FlatForward,
    ):
        if dates.dtype != "np.datetime64[D]":
            dates = dates.astype("datetime64[D]")
        self.dates = dates
        self.values = values
        self._curve_date = curve_date.to_numpy()
        self._offsets = np.asarray(offsets, dtype=np.int64)
        self.value_type = value_type
        self.interp = interp
        self.extrap = extrap

        # Convert input values to log discount factors
        values = np.asarray(values, dtype=np.float64)
        self._log_dfs = _to_log_discount_factors(self._offsets, values, value_type)

        # Build the underlying Curve1d with log discount factors
        self._curve = Curve1d(
            x=self._offsets,
            y=self._log_dfs,
            interp=_map_interpolator(interp),
            extrap=_map_extrapolator(extrap),
        )

    # -------------------------------------------------------------------------
    # Node conversions
    # -------------------------------------------------------------------------
    def _convert_to_nodes(self):
        node_values = self.values.copy()
        node_offsets = self.offsets.copy()
        source_type, target_type = _map_transform(self.interp, self._value_type)
        func = _value_transformer[(source_type, target_type)]
        node_values = func(node_values)
        self._node_values = node_values
        self._node_value_type = target_type

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def curve_date(self) -> Date:
        """The anchor date for the curve."""
        return self._curve_date

    @property
    def offsets(self) -> NDArray[np.int64]:
        """Day offsets from curve_date for each curve point."""
        return self._offsets.copy()

    @property
    def log_discount_factors_at_nodes(self) -> NDArray[np.float64]:
        """Log discount factors at the curve nodes."""
        return self._log_dfs.copy()  # type: ignore[return-value]

    # -------------------------------------------------------------------------
    # Date conversion helpers
    # -------------------------------------------------------------------------

    def _dates_to_offsets(self, dates: DateArray | NDArray[np.int64]) -> IntArray:
        """
        Convert datetime64[D] array or integer offsets to integer offsets from curve_date.

        Parameters
        ----------
        dates : DateArray or IntArray
            Either datetime64[D] values or integer offsets

        Returns
        -------
        IntArray
            Integer offsets from curve_date
        """
        dates = np.asarray(dates)

        if np.issubdtype(dates.dtype, np.datetime64):
            # Convert datetime64[D] to integer offsets
            # datetime64[D] stores days since 1970-01-01 (same as our ordinal epoch)
            days_since_epoch = dates.astype('datetime64[D]').astype(np.int64)
            return (days_since_epoch - self._curve_date_ordinal).astype(np.int64)
        else:
            # Assume already integer offsets
            return dates.astype(np.int64)

    # -------------------------------------------------------------------------
    # Curve value accessors
    # -------------------------------------------------------------------------

    def get_log_discount_factors(
        self,
        dates: DateArray | NDArray[np.int64],
    ) -> FloatArray:
        """
        Get log discount factors: -ln(df(t)) at the given dates.

        Parameters
        ----------
        dates : DateArray or IntArray
            Either datetime64[D] values or integer offsets from curve_date

        Returns
        -------
        FloatArray
            Log discount factors at the requested dates
        """
        offsets = self._dates_to_offsets(dates)
        return self._curve.get_value(offsets)

    def get_discount_factors(
        self,
        dates: DateArray | NDArray[np.int64],
    ) -> FloatArray:
        """
        Get discount factors: df(t) at the given dates.

        Parameters
        ----------
        dates : DateArray or IntArray
            Either datetime64[D] values or integer offsets from curve_date

        Returns
        -------
        FloatArray
            Discount factors at the requested dates
        """
        log_dfs = self.get_log_discount_factors(dates)
        return np.exp(-log_dfs)

    def get_zero_rates(
        self,
        dates: DateArray | NDArray[np.int64],
    ) -> FloatArray:
        """
        Get continuously compounded zero rates: r(t) at the given dates.

        Zero rate is defined as: r(t) = -ln(df(t)) / t

        Parameters
        ----------
        dates : DateArray or IntArray
            Either datetime64[D] values or integer offsets from curve_date

        Returns
        -------
        FloatArray
            Zero rates at the requested dates

        Notes
        -----
        Returns 0.0 for t=0 (curve_date) to avoid division by zero.
        Day count convention is ACT/365.25 for now.
        """
        offsets = self._dates_to_offsets(dates)
        log_dfs = self._curve.get_value(offsets)

        # Convert offsets to year fractions
        t = offsets.astype(np.float64) / _DAYS_PER_YEAR

        # Avoid division by zero at t=0
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.where(t != 0, log_dfs / t, 0.0)

        return rates

    def get_forward_rate(
        self,
        start_dates: DateArray | NDArray[np.int64],
        end_dates: DateArray | NDArray[np.int64],
    ) -> FloatArray:
        """
        Get forward rates between start and end dates.

        Forward rate is: f(t1, t2) = (ln(df(t1)) - ln(df(t2))) / (t2 - t1)
                                   = (log_df(t2) - log_df(t1)) / (t2 - t1)

        Parameters
        ----------
        start_dates : DateArray or IntArray
            Start dates (or offsets) for the forward period
        end_dates : DateArray or IntArray
            End dates (or offsets) for the forward period

        Returns
        -------
        FloatArray
            Forward rates for each (start, end) pair
        """
        start_offsets = self._dates_to_offsets(start_dates)
        end_offsets = self._dates_to_offsets(end_dates)

        log_df_start = self._curve.get_value(start_offsets)
        log_df_end = self._curve.get_value(end_offsets)

        # Year fractions
        dt = (end_offsets - start_offsets).astype(np.float64) / _DAYS_PER_YEAR

        with np.errstate(divide='ignore', invalid='ignore'):
            fwd = np.where(dt != 0, (log_df_end - log_df_start) / dt, 0.0)

        return fwd

    # -------------------------------------------------------------------------
    # Convenience scalar accessors
    # -------------------------------------------------------------------------

    def discount_factor(self, date: NDArray[np.datetime64]) -> FloatArray:
        """Get discount factors at the given dates.

        Parameters
        ----------
        date : NDArray[np.datetime64]
            Array of datetime64[D] values

        Returns
        -------
        FloatArray
            Discount factors at the requested dates
        """
        return self.get_discount_factors(date)

    def zero_rate(self, date: NDArray[np.datetime64]) -> FloatArray:
        """Get zero rates at the given dates.

        Parameters
        ----------
        date : NDArray[np.datetime64]
            Array of datetime64[D] values

        Returns
        -------
        FloatArray
            Zero rates at the requested dates
        """
        return self.get_zero_rates(date)

    # -------------------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ZeroCurve("
            f"curve_date={self._curve_date!r}, "
            f"n_points={len(self._offsets)}, "
            f"interp={self._interp.name}, "
            f"extrap={self._extrap.name})"
        )
