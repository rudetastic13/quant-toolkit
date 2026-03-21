"""Define the registry and factories for the term math operations"""

from typing import Callable, cast
import numpy as np
from common.registry import Registry, register_with, create_factory
from finance.dates import Date
from .term_type import TermType
from finance.dates.schedules.helpers import _DAYS_IN_MONTH_YEAR
from finance.dates.types import (
    IntScalar,
    IntNpType,
    DateNpType
)


# region, simple date registry
_date_based_registry: Registry[Callable[[Date, int], Date]] = Registry("Simple Date Based Registry")

@register_with(_date_based_registry, TermType.Days)
def _add_days(dt: Date, value: IntScalar) -> Date:
    """Simple adding of days to a date"""
    return Date.from_ordinal(dt.to_ordinal() + value)

@register_with(_date_based_registry, TermType.Weeks)
def _add_weeks(dt: Date, value: IntScalar) -> Date:
    """Adding weeks to a date"""
    return _add_days(dt, value * 7)

@register_with(_date_based_registry, TermType.Months)
def _add_months(dt: Date, value: IntScalar) -> Date:
    month = dt.month - 1 + value
    year = dt.year + month // 12
    month = month % 12 + 1
    if dt.day <= 28:
        return Date(year, month, dt.day)
    day = min(dt.day, _DAYS_IN_MONTH_YEAR[year, month])
    return Date(year, month, day)

@register_with(_date_based_registry, TermType.Quarters)
def _add_quarters(dt: Date, value: IntScalar) -> Date:
    return _add_months(dt, value * 3)

@register_with(_date_based_registry, TermType.Years)
def _add_years(dt: Date, value: IntScalar) -> Date:
    year = dt.year + value
    try:
        return Date(year, dt.month, dt.day)
    except:
        return Date(year, dt.month, 28)

@register_with(_date_based_registry, TermType.BusinessDays)
def _add_business_days(*_args, **_kwargs) -> Date:
    raise ArithmeticError("Business day addition requires calendar, invalid in term operations!")

# endregion

# region, numpy-based term math
_numpy_registry: Registry[Callable[[DateNpType, IntNpType], DateNpType]] = Registry("Numpy Date Based Registry")

@register_with(_numpy_registry, TermType.Days)
def _add_np_days(dt: DateNpType, value: IntNpType) -> DateNpType:
    return dt + value.astype("timedelta64[D]")

@register_with(_numpy_registry, TermType.Weeks)
def _add_np_weeks(dt: DateNpType, value: IntNpType) -> DateNpType:
    return _add_np_days(dt, value * 7)

@register_with(_numpy_registry, TermType.Months)
def _add_np_months(dt: DateNpType, value: IntNpType) -> DateNpType:
    start = dt.astype("datetime64[M]")
    target = start + value.astype("timedelta64[M]")
    days = (dt - start) + 1
    days_in_target = (target + np.timedelta64(1, "M")).astype("datetime64[D]") - target
    days_clamped = np.minimum(days, days_in_target)
    return target.astype("datetime64[D]") + days_clamped - 1

@register_with(_numpy_registry, TermType.Quarters)
def _add_np_quarters(dt: DateNpType, value: IntNpType) -> DateNpType:
    return _add_np_months(dt, value * 3)

@register_with(_numpy_registry, TermType.Years)
def _add_np_years(dt: DateNpType, value: IntNpType) -> DateNpType:
    return _add_np_months(dt, value * 12)

@register_with(_numpy_registry, TermType.BusinessDays)
def _add_np_business_days(*_args, **_kwargs) -> None:
    raise ArithmeticError("Business day addition requires calendar, invalid in term operations!")


# endregion

# region, factories
date_term_math = create_factory(_date_based_registry)
np_term_math = create_factory(_numpy_registry)
# endregion
