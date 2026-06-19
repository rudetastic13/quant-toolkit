"""Excel value <-> domain type conversion.

Excel hands UDFs Python primitives (``float``/``str``/``datetime``) and ranges as nested
lists, with dates as either ``datetime`` objects or 1900-system serial numbers depending on
configuration. Every value is funnelled through a ``finance.dates.Date`` converter so date
semantics (including the Excel serial offset) live in one place — the core ``Date`` class —
and never get re-implemented here.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from finance.dates import Date


def flatten(values: object) -> list:
    """Flatten a scalar / 1-D list / 2-D Excel range into a flat Python list."""
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return list(values.ravel())
    if isinstance(values, (list, tuple)):
        out: list = []
        for v in values:
            if isinstance(v, (list, tuple, np.ndarray)):
                out.extend(flatten(v))
            else:
                out.append(v)
        return out
    return [values]


def to_date(value: object) -> Date:
    """Coerce a single Excel value to a ``Date``, dispatching on its type.

    Accepts ``Date``, ``datetime``/``date``, ``datetime64``, ISO strings, and Excel
    1900-system serial numbers — each handled by the matching ``Date`` converter.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        flat = flatten(value)
        if not flat:
            raise ValueError("expected a date, got an empty range")
        value = flat[0]
    if isinstance(value, Date):
        return value
    if isinstance(value, _dt.datetime):
        return Date.from_datetime(value)
    if isinstance(value, _dt.date):
        return Date.from_date(value)
    if isinstance(value, np.datetime64):
        return Date.from_numpy(value)
    if isinstance(value, str):
        return Date.from_str(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return Date.from_excel(value)
    raise TypeError(f"cannot interpret {value!r} as a date")


def to_dates(values: object) -> list[Date]:
    """A range/list of Excel date cells -> list of ``Date``."""
    return [to_date(v) for v in flatten(values)]


def to_datetime64_array(values: object) -> np.ndarray:
    """A range/list of Excel dates -> ``np.ndarray[datetime64[D]]`` (via ``Date``)."""
    days = [d.to_numpy() for d in to_dates(values)]
    return np.array(days, dtype="datetime64[D]")


def to_float_array(values: object) -> np.ndarray:
    """A range/list of numbers -> ``np.ndarray[float64]``."""
    return np.asarray([float(v) for v in flatten(values)], dtype=np.float64)


def handles(values: object) -> list[str]:
    """A range/list of handle cells -> list of handle strings (blanks dropped)."""
    return [str(v) for v in flatten(values) if v not in (None, "")]
