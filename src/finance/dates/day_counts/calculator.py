from typing import Callable
import numpy as np
from common.registry import create_factory, register_with, Registry
from finance.dates.enums.day_count_methods import DayCountMethod

_period_fraction_registry: Registry[Callable[[DayCountMethod, np.ndarray, np.ndarray], np.ndarray | None]] = Registry("Period Fractions")

def _datetime64_to_ymd(dates: np.ndarray) -> np.ndarray:
    """Convert datetime64[D] array to (N, 3) array of [year, month, day]."""
    years = dates.astype('datetime64[Y]').astype(np.int64) + 1970
    months = dates.astype('datetime64[M]').astype(np.int64) % 12 + 1
    days = (dates - dates.astype('datetime64[M]')).astype(np.int64) + 1
    dtype = np.dtype([('year', np.int64), ('month', np.int64), ('day', np.int64)])
    ymd = np.empty_like(dates, dtype=dtype)
    ymd['year'] = years
    ymd['month'] = months
    ymd['day'] = days
    return ymd

@register_with(_period_fraction_registry, DayCountMethod.Unused)
def _unused(start_dates: np.ndarray, end_dates, out: np.ndarray | None = None) -> np.ndarray:
    if out is None:
        out = np.empty_like(start_dates, dtype=np.float64)
    out[:] = 0
    return out

@register_with(_period_fraction_registry, DayCountMethod.Actual360)
def _actual_360(start_dates: np.ndarray, end_dates, out: np.ndarray | None = None) -> np.ndarray:
    """Calculate the period fraction using the Actual/360 method"""
    if out is None:
        out = np.empty_like(start_dates, dtype=np.float64)
    np.subtract(end_dates.view(np.int64), start_dates.view(np.int64), out=out)
    np.divide(out, 360, out=out)
    return out

@register_with(_period_fraction_registry, DayCountMethod.Actual365)
def _actual_365(start_dates: np.ndarray, end_dates, out: np.ndarray | None = None) -> np.ndarray:
    """Calculate the period fraction using the Actual/365 method"""
    if out is None:
        out = np.empty_like(start_dates, dtype=np.float64)
    np.subtract(end_dates.view(np.int64), start_dates.view(np.int64), out=out)
    np.divide(out, 365, out=out)
    return out

# for american style 30/360
def _thirty_style(start_dates: np.ndarray, end_dates: np.ndarray, out: np.ndarray | None = None, denominator=360) -> np.ndarray:
    """Calculate the period fraction using the 30/360 method"""
    if out is None:
        out = np.empty_like(start_dates, dtype=np.float64)
    tmp = np.empty_like(out)
    start_ymd = _datetime64_to_ymd(start_dates)
    end_ymd = _datetime64_to_ymd(end_dates)

    # (Y2 - Y1) * 360
    np.subtract(end_ymd['year'], start_ymd['year'], out=out)
    out *= 360
    # + (M2 - M1) * 30
    np.subtract(end_ymd['month'], start_ymd['month'], out=tmp)
    tmp *= 30
    out += tmp
    # + min(D2, 30) - min(D1, 30)
    np.minimum(end_ymd['day'], 30, out=tmp)
    out += tmp
    np.minimum(start_ymd['day'], 30, out=tmp)
    out -= tmp
    # / 360
    out /= denominator
    return out

@register_with(_period_fraction_registry, DayCountMethod.Thirty360)
def _thirty_360(start_dates: np.ndarray, end_dates: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Calculate the period fraction using the 30/360 method"""
    return _thirty_style(start_dates, end_dates, out=out, denominator=360)

@register_with(_period_fraction_registry, DayCountMethod.Thirty365)
def _thirty_365(start_dates: np.ndarray, end_dates: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Calculate the period fraction using the 30/360 method"""
    return _thirty_style(start_dates, end_dates, out=out, denominator=365)

# and define the factor
period_fractions = create_factory(_period_fraction_registry)