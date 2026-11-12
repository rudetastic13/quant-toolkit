from __future__ import annotations
from datetime import date, timedelta
from typing import Callable

ObservanceFn = Callable[[date], date]
FilterFn = Callable[[date], bool]

def nearest_weekday(dt: date) -> date:
    wd = dt.weekday()
    if wd == 5:
        return dt - timedelta(days=1)
    elif wd == 6:
        return dt + timedelta(days=1)
    return dt

def next_monday(dt: date) -> date:
    wd = dt.weekday()
    if wd == 5:
        return dt + timedelta(days=2)
    elif wd == 6:
        return dt + timedelta(days=1)
    return dt

def previous_friday(dt: date) -> date:
    wd = dt.weekday()
    if wd == 5:
        return dt - timedelta(days=1)
    elif wd == 6:
        return dt - timedelta(days=2)
    return dt

def sunday_to_monday(dt: date) -> date:
    wd = dt.weekday()
    if wd == 6:
        return dt + timedelta(days=1)
    return dt

def not_first_week_of_month(dt: date) -> bool:
    return dt.day > 7
