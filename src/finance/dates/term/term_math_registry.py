from typing import TypeVar, Callable
import numpy as np
import numpy.typing as npt
from common.registry import Registry, register_with, create_factory
from finance.dates import Date
from .term_type import TermType
from finance.dates.schedules.helpers import _DAYS_IN_MONTH_YEAR


# define types
IntScalar = np.int64
IntArray = npt.NDArray[np.int64]
DateScalar = np.datetime64
DateArray = npt.NDArray[np.datetime64]
DateNpT = TypeVar("DateNpT", bound=DateScalar | DateArray)
IntNpt = TypeVar("IntNpt", bound=IntScalar | IntArray)
DateT = TypeVar("DateT", bound=DateScalar | DateArray | Date)
IntT = TypeVar("IntT", bound=IntScalar | IntArray | int)

# region, simple date registry
_date_based_registry: Registry[Callable[[Date, int], Date]] = Registry("Simple Date Based Registry")

@register_with(_date_based_registry, TermType.Days)
def _add_days(dt: Date, value: int | IntScalar) -> Date:
    return Date.from_ordinal(dt.to_ordinal() + value)

@register_with(_date_based_registry, TermType.Weeks)
def _add_weeks(dt: Date, value: int | IntScalar) -> Date:
    return _add_days(dt, value * 7)

@register_with(_date_based_registry, TermType.Months)
def _add_months(dt: Date, value: int | IntScalar) -> Date:
    month = dt.month - 1 + value
    year = dt.year + month // 12
    month = month % 12 + 1
    if dt.day <= 28:
        return Date(year, month, dt.day)
    day = min(dt.day, _DAYS_IN_MONTH_YEAR[year, month])
    return Date(year, month, day)

@register_with(_date_based_registry, TermType.Quarters)
def _add_quarters(dt: Date, value: int | IntScalar) -> Date:
    return _add_months(dt, value * 3)

@register_with(_date_based_registry, TermType.Years)
def _add_years(dt: Date, value: int | IntScalar) -> Date:
    year = dt.year + value
    try:
        return Date(year, dt.month, dt.day)
    except:
        return Date(year, dt.month, 28)

@register_with(_date_based_registry, TermType.BusinessDays)
def _add_business_days(dt: Date, value: int | IntScalar) -> Date:
    raise ArithmeticError("Business day addition requires calendar, invalid in term operations!")

# endregion

# region, numpy-based term math
_numpy_registry: Registry[Callable[[IntNpt, DateNpT], Date]] = Registry("Numpy Date Based Registry")

@register_with(_numpy_registry, TermType.Days)
def _add_np_days(dt: DateNpT, value: IntNpt) -> DateNpT:
    return dt + value.astype("timedelta[D]")

@register_with(_numpy_registry, TermType.Weeks)
def _add_np_weeks(dt: DateNpT, value: IntNpt) -> DateNpT:
    return _add_weeks(dt, value * 7)

@register_with(_numpy_registry, TermType.Months)
def _add_np_months(dt: DateNpT, value: IntNpt) -> DateNpT:
    start = dt.astype("datetime64[M]")
    target = start + value.astype("timedelta[M]")
    days = (dt - start) + 1
    days_in_target = (target + np.timedelta64(1, "M")).astype("datetime64[D]") - target
    days_clamped = np.minimum(days, days_in_target)
    return target.astype("datetime64[D]") + days_clamped - 1

@register_with(_numpy_registry, TermType.Quarters)
def _add_np_quarters(dt: DateNpT, value: IntNpt) -> DateNpT:
    return _add_np_months(dt, value * 3)

@register_with(_numpy_registry, TermType.Years)
def _add_np_years(dt: DateNpT, value: IntNpt) -> DateNpT:
    return _add_np_months(dt, value * 12)

@register_with(_numpy_registry, TermType.BusinessDays)
def _add_np_business_days(*_args, **_kwargs) -> None:
    return _add_business_days(*_args, **_kwargs)

# endregion

# region, factories
date_term_math = create_factory(_date_based_registry)
np_term_math = create_factory(_numpy_registry)
# endregion
