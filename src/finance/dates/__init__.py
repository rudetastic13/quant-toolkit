import numpy as np

def _setup_lookup_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.arange(0, 10_000, dtype=np.uint16)

    # get leap years
    is_leap = ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)
    is_leap = is_leap.astype(np.bool_)

    # days for given year/month, use fast lookup table
    days_in_year = np.empty((10_000, 13), dtype=np.uint8)
    days_in_year[:, 0:] = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.uint8)
    days_in_year[is_leap, 2] = 29  # adjust February

    # months to days
    dts_d_n = np.arange("0000-01", "10000-01", dtype="datetime64[M]").astype("datetime64[D]").astype(np.int32)

    # done
    return days_in_year, is_leap, dts_d_n

# static arrays
_DAYS_IN_MONTH_YEAR, _IS_LEAP_YEAR, _MONTH_TO_DAYS_OFFSET = _setup_lookup_arrays()

# getters of statics
def get_days_in_month_year() -> np.ndarray:
    return _DAYS_IN_MONTH_YEAR

def get_is_leap_year() -> np.ndarray:
    return _IS_LEAP_YEAR

def get_month_to_days_offset() -> np.ndarray:
    return _MONTH_TO_DAYS_OFFSET

from finance.dates.date import Date
from finance.dates.term import Term, TermType
from finance.dates.enums import BDC, Roll, Direction, Frequency, DayCountMethod
from finance.dates.calendars import Calendars, Calendar
from finance.dates.day_counts import period_fractions
from finance.dates.utils import (
    date_based_utils_module,
    np_date_based_utils_module,
    is_good_bd,
    adjust_date,
    add_business_days,
    add_term,
    subtract_term,
    add_frequency,
    subtract_frequency,
    next_imm_date,
    prior_imm_date,
)
from common.object.serializer import register_type

_ = Calendars() # instantiate the singleton at import

# Register Date and Term for CommonObject serialization
register_type(Date, lambda d: d.to_str(), lambda s: Date.from_str(s))
register_type(Term, lambda t: f"{t.term_length}{t.term_type.value}", lambda s: Term.from_str(s))

__all__ = [
    "Date",
    "Term",
    "BDC",
    "Roll",
    "TermType",
    "Direction",
    "Frequency",
    "DayCountMethod",
    "Calendars",
    "Calendar",
    "period_fractions",
    "date_based_utils_module",
    "np_date_based_utils_module",
    "is_good_bd",
    "adjust_date",
    "add_business_days",
    "add_term",
    "subtract_term",
    "add_frequency",
    "subtract_frequency",
    "next_imm_date",
    "prior_imm_date",
]
