from typing import overload
import numpy as np
from finance.dates import Date, get_days_in_month_year, get_is_leap_year, get_month_to_days_offset


# static arrays
_DAYS_IN_MONTH_YEAR = get_days_in_month_year()
_MONTH_TO_DAYS_OFFSET = get_month_to_days_offset()
_IS_LEAP_YEAR = get_is_leap_year()
_MARCH_EPOCH = 719_468

def ymd_from_days(days: int) -> tuple[int, int, int]:
    return Date.fromordinal(days).to_ymd()

def days_from_ymd(y: int, m: int, d: int) -> int:
    # adjust month/year for march-based calculation
    return Date(y, m, d).toordinal()

def days_to_month_index(days: int) -> tuple[int, int]:
    y, m, d = ymd_from_days(days)
    return y * 12 + (m - 1), d

def month_index_to_days(month_index: int, day: int) -> int:
    y = month_index // 12
    m = (month_index % 12) + 1
    base_value = days_from_ymd(y, m, 1)
    day = min(day, _DAYS_IN_MONTH_YEAR[y, m])
    return base_value + (day - 1)

@overload
def is_leap_year(y: int) -> np.bool_: ...

@overload
def is_leap_year(y: np.ndarray) -> np.ndarray: ...

def is_leap_year(y: int) -> bool:
    return _IS_LEAP_YEAR[y]

def is_eom_ymd(y: int, m: int, d: int) -> bool:
    return d == _DAYS_IN_MONTH_YEAR[y, m]

def is_eom_days(days: int) -> bool:
    y, m, d = ymd_from_days(days)
    return is_eom_ymd(y, m, d)

def align_eom_day(days: int) -> int:
    y, m, d = ymd_from_days(days)
    dom = _DAYS_IN_MONTH_YEAR[y, m]
    return days + (dom - d)

def third_wednesday(y: int, m: int) -> int:
    fom = days_from_ymd(y, m, 1)
    dow = (fom + 3) % 7  # Monday=0; ordinal 0 (1970-01-01) is a Thursday
    return fom + (2 - dow) % 7 + 14

def roll_day(days:int, roll_convention: int) -> int:
    if roll_convention == -2:
        y, m, _ = ymd_from_days(days)
        return third_wednesday(y, m)
    if roll_convention == -1:
        return align_eom_day(days)
    if roll_convention == 0:
        return days
    y, m, d = ymd_from_days(days)
    eom_day = _DAYS_IN_MONTH_YEAR[y, m]
    if roll_convention >= eom_day:
        return days + (eom_day - d)
    return days + (roll_convention - d)

def days_in_month(y: int, m: int) -> int:
    return _DAYS_IN_MONTH_YEAR[y, m]

def guess_array_size(start_day: int, end_day: int, freq_type: int, freq_value: int) -> int:
    max_dates = 0
    if freq_type != -1:
        max_dates = end_day - start_day + 1
        if freq_type == 1:
            max_dates = (max_dates // 28) // freq_value
        elif freq_type == 2:
            max_dates = max_dates // freq_value
    return max_dates + 4