"""
Date Utility Dispatchers
========================

Type-aware façade that routes every call to the correct implementation
module based on the runtime type of the ``dt`` argument.

Routing Table
-------------
=========== ==========================================
Type        Backend
=========== ==========================================
``Date``    ``finance.dates.utils.impl.dt_based``
``datetime64`` / ``ndarray``  ``impl.np_based``
=========== ==========================================

Modules are imported lazily so heavy numpy helpers are not loaded until
they are actually needed.
"""
from importlib import import_module
from typing import Any, Callable
from types import ModuleType
import numpy as np
from finance.dates import Date, Term, Frequency, Calendar, BDC
from finance.dates.types import DateType, BoolType


def date_based_utils_module() -> ModuleType:
    return import_module("finance.dates.utils.impl.dt_based")

def np_date_based_utils_module() -> ModuleType:
    return import_module("finance.dates.utils.impl.np_based")

_date_based_utils_module = date_based_utils_module
_np_date_based_utils_module = np_date_based_utils_module

type_map: dict[Any, Callable[[], ModuleType]] = {
    Date: _date_based_utils_module,
    np.datetime64: _np_date_based_utils_module,
    np.ndarray: _np_date_based_utils_module,
}


def is_good_bd(dt: DateType, calendar: str | Calendar) -> BoolType:
    """
    Check whether ``dt`` falls on a good business day.

    Parameters
    ----------
    dt : Date, np.datetime64, or np.ndarray[datetime64[D]]
        The date(s) to check.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    bool, np.bool\_, or np.ndarray[bool]
        ``True`` for each element that is a good business day.

    Examples
    --------
    >>> is_good_bd(Date(2025, 12, 25), "USD")
    False

    >>> dates = np.array(["2025-12-24", "2025-12-25"], dtype="datetime64[D]")
    >>> is_good_bd(dates, "USD")
    array([ True, False])
    """
    return type_map[type(dt)]().is_good_bd(dt, calendar)


def adjust_date(dt: DateType, bdc: str | BDC, calendar: str | Calendar) -> DateType:
    """
    Roll ``dt`` to the nearest good business day using ``bdc`` convention.

    Parameters
    ----------
    dt : Date, np.datetime64, or np.ndarray[datetime64[D]]
        The date(s) to adjust.
    bdc : str or BDC
        Business day convention (e.g. ``BDC.ModifiedFollowing`` or ``"MF"``).
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date, np.datetime64, or np.ndarray[datetime64[D]]
        Adjusted date(s).

    Examples
    --------
    >>> adjust_date(Date(2025, 8, 30), BDC.ModifiedFollowing, "USD")
    Date(2025, 8, 29)

    >>> adjust_date(np.datetime64("2025-08-30"), "MF", "USD")
    numpy.datetime64('2025-08-29')
    """
    return type_map[type(dt)]().adjust_date(dt, bdc, calendar)


def add_business_days(dt: DateType, days: int, bdc: str | BDC, calendar: str | Calendar) -> DateType:
    """
    Advance ``dt`` by ``days`` business days.

    Parameters
    ----------
    dt : Date, np.datetime64, or np.ndarray[datetime64[D]]
        Starting date(s).
    days : int
        Number of business days to advance (negative to go back).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date, np.datetime64, or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> add_business_days(Date(2025, 1, 2), 5, BDC.Following, "USD")
    Date(2025, 1, 9)

    >>> starts = np.array(["2025-01-02", "2025-01-03"], dtype="datetime64[D]")
    >>> add_business_days(starts, 5, BDC.Following, "USD")
    array(['2025-01-09', '2025-01-10'], dtype='datetime64[D]')
    """
    return type_map[type(dt)]().add_business_days(dt, days, bdc, calendar)


def add_term(dt: DateType, term: Term, bdc: str | BDC, calendar: str | Calendar) -> DateType:
    """
    Add a ``Term`` offset to ``dt`` and adjust to a good business day.

    Parameters
    ----------
    dt : Date, np.datetime64, or np.ndarray[datetime64[D]]
        Starting date(s).
    term : Term
        Offset to add (e.g. ``Term(3, TermType.Months)``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date, np.datetime64, or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> add_term(Date(2025, 1, 15), Term(3, TermType.Months), BDC.ModifiedFollowing, "USD")
    Date(2025, 4, 15)

    >>> add_term(np.datetime64("2025-01-15"), Term(3, TermType.Months), "MF", "USD")
    numpy.datetime64('2025-04-15')
    """
    return type_map[type(dt)]().add_term(dt, term, bdc, calendar)


def subtract_term(dt: DateType, term: Term, bdc: str | BDC, calendar: str | Calendar) -> DateType:
    """
    Subtract a ``Term`` offset from ``dt`` and adjust to a good business day.

    Parameters
    ----------
    dt : Date, np.datetime64, or np.ndarray[datetime64[D]]
        Starting date(s).
    term : Term
        Offset to subtract (e.g. ``Term(6, TermType.Months)``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date, np.datetime64, or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> subtract_term(Date(2025, 7, 15), Term(6, TermType.Months), BDC.Following, "USD")
    Date(2025, 1, 15)

    >>> subtract_term(np.datetime64("2025-07-15"), Term(6, TermType.Months), "F", "USD")
    numpy.datetime64('2025-01-15')
    """
    return type_map[type(dt)]().subtract_term(dt, term, bdc, calendar)


def add_frequency(dt: DateType, frequency: Frequency, bdc: str | BDC, calendar: str | Calendar) -> DateType:
    """
    Advance ``dt`` by one period of ``frequency``.

    Parameters
    ----------
    dt : Date, np.datetime64, or np.ndarray[datetime64[D]]
        Starting date(s).
    frequency : Frequency
        Period frequency (e.g. ``Frequency.Quarterly``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date, np.datetime64, or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> add_frequency(Date(2025, 3, 31), Frequency.Quarterly, BDC.ModifiedFollowing, "USD")
    Date(2025, 6, 30)

    >>> dates = np.array(["2025-03-31", "2025-06-30"], dtype="datetime64[D]")
    >>> add_frequency(dates, Frequency.Quarterly, "MF", "USD")
    array(['2025-06-30', '2025-09-30'], dtype='datetime64[D]')
    """
    return type_map[type(dt)]().add_frequency(dt, frequency, bdc, calendar)


def subtract_frequency(dt: DateType, frequency: Frequency, bdc: str | BDC, calendar: str | Calendar) -> DateType:
    """
    Move ``dt`` back by one period of ``frequency``.

    Parameters
    ----------
    dt : Date, np.datetime64, or np.ndarray[datetime64[D]]
        Starting date(s).
    frequency : Frequency
        Period frequency (e.g. ``Frequency.Quarterly``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date, np.datetime64, or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> subtract_frequency(Date(2025, 6, 30), Frequency.Quarterly, BDC.ModifiedFollowing, "USD")
    Date(2025, 3, 31)

    >>> subtract_frequency(np.datetime64("2025-06-30"), Frequency.Quarterly, "MF", "USD")
    numpy.datetime64('2025-03-31')
    """
    return type_map[type(dt)]().subtract_frequency(dt, frequency, bdc, calendar)
