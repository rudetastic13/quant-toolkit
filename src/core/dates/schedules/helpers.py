from typing import overload
import numpy as np
from core.dates import Date

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
_MARCH_EPOCH = 719_468

def ymd_from_days(days: int) -> tuple[int, int, int]:
    return Date.from_ordinal(days).to_ymd()

def days_from_ymd(y: int, m: int, d: int) -> int:
    # adjust month/year for march-based calculation
    return Date(y, m, d).to_ordinal()

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

def roll_day(days:int, roll_convention: int) -> int:
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