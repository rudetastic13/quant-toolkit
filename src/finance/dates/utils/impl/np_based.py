"""
NumPy ``datetime64`` / ``ndarray`` Utilities
=============================================

Business-day checks, date adjustment by convention, and calendar-aware
date arithmetic for ``np.datetime64`` scalars and ``np.ndarray`` date
arrays, backed by vectorized NumPy operations.
"""
from typing import overload, Literal
import numpy as np
from .helpers import CalendarLike, BdcLike
from .helpers import calendar as clean_calendar, bdc as clean_bdc
from finance.dates import BDC, Term, TermType, Frequency
from finance.dates.types import DateNpType, BoolNpType

NpBDC = Literal["following", "preceding", "modifiedfollowing", "modifiedpreceding"]

_NP_BDC_MAP: dict[BDC, NpBDC | None] = {
    BDC.NoAdjustment: None,
    BDC.Following: "following",
    BDC.Preceding: "preceding",
    BDC.ModifiedFollowing: "modifiedfollowing",
    BDC.ModifiedPreceding: "modifiedpreceding",
}


@overload
def is_good_bd(dt: np.datetime64, calendar: CalendarLike) -> np.bool_:
    ...

@overload
def is_good_bd(dt: np.ndarray, calendar: CalendarLike) -> np.ndarray:
    ...

def is_good_bd(dt: DateNpType, calendar: CalendarLike) -> BoolNpType:
    """
    Check whether each element of ``dt`` is a good business day.

    Parameters
    ----------
    dt : np.datetime64 or np.ndarray[datetime64[D]]
        Scalar or array of dates to check.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    np.bool_ or np.ndarray[bool]
        ``True`` for each element that is a good business day.

    Examples
    --------
    >>> is_good_bd(np.datetime64("2025-12-25"), "USD")
    False

    >>> dates = np.array(["2025-12-24", "2025-12-25"], dtype="datetime64[D]")
    >>> is_good_bd(dates, "USD")
    array([ True, False])
    """
    np_calendar = clean_calendar(calendar).np_calendar
    return np.is_busday(dt, np_calendar.weekmask, np_calendar.holidays)


@overload
def adjust_date(dt: np.datetime64, bdc: BdcLike, calendar: CalendarLike) -> np.datetime64:
    ...

@overload
def adjust_date(dt: np.ndarray, bdc: BdcLike, calendar: CalendarLike) -> np.ndarray:
    ...

def adjust_date(dt: DateNpType, bdc: BdcLike, calendar: CalendarLike) -> DateNpType:
    """
    Roll ``dt`` to the nearest good business day using ``bdc`` convention.

    Parameters
    ----------
    dt : np.datetime64 or np.ndarray[datetime64[D]]
        Scalar or array of dates to adjust.
    bdc : str or BDC
        Business day convention (e.g. ``BDC.ModifiedFollowing`` or ``"MF"``).
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    np.datetime64 or np.ndarray[datetime64[D]]
        Adjusted date(s).

    Examples
    --------
    >>> adjust_date(np.datetime64("2025-08-30"), "MF", "USD")
    numpy.datetime64('2025-08-29')

    >>> dates = np.array(["2025-08-30", "2025-08-31"], dtype="datetime64[D]")
    >>> adjust_date(dates, BDC.ModifiedFollowing, "USD")
    array(['2025-08-29', '2025-08-29'], dtype='datetime64[D]')
    """
    np_calendar: np.busdaycalendar = clean_calendar(calendar).np_calendar
    bdc = _NP_BDC_MAP[clean_bdc(bdc)]
    if not bdc:
        return dt
    return np.busday_offset(
        dt,
        offsets=0,
        roll=bdc,
        weekmask=np_calendar.weekmask,
        holidays=np_calendar.holidays
    )


def add_business_days(dt: DateNpType, days: int, bdc: BdcLike, calendar: CalendarLike) -> DateNpType:
    """
    Advance ``dt`` by ``days`` business days.

    Parameters
    ----------
    dt : np.datetime64 or np.ndarray[datetime64[D]]
        Starting date(s).
    days : int
        Number of business days to advance (negative to go back).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    np.datetime64 or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> add_business_days(np.datetime64("2025-01-02"), 5, BDC.Following, "USD")
    numpy.datetime64('2025-01-09')

    >>> starts = np.array(["2025-01-02", "2025-01-03"], dtype="datetime64[D]")
    >>> add_business_days(starts, 5, "F", "USD")
    array(['2025-01-09', '2025-01-10'], dtype='datetime64[D]')
    """
    np_calendar: np.busdaycalendar = clean_calendar(calendar).np_calendar
    bdc = _NP_BDC_MAP[clean_bdc(bdc)]
    if not bdc:
        return dt + Term(days, TermType.Days)
    return np.busday_offset(
        dt,
        offsets=days,
        roll=bdc,
        weekmask=np_calendar.weekmask,
        holidays=np_calendar.holidays
    )


def add_term(dt: DateNpType, term: Term, bdc: BdcLike, calendar: CalendarLike):
    """
    Add a ``Term`` offset to ``dt`` and adjust to a good business day.

    Parameters
    ----------
    dt : np.datetime64 or np.ndarray[datetime64[D]]
        Starting date(s).
    term : Term
        Offset to add (e.g. ``Term(3, TermType.Months)``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    np.datetime64 or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> add_term(np.datetime64("2025-01-15"), Term(3, TermType.Months), "MF", "USD")
    numpy.datetime64('2025-04-15')

    >>> dates = np.array(["2025-01-15", "2025-04-15"], dtype="datetime64[D]")
    >>> add_term(dates, Term(3, TermType.Months), BDC.ModifiedFollowing, "USD")
    array(['2025-04-15', '2025-07-15'], dtype='datetime64[D]')
    """
    if term.term_type == TermType.BusinessDays:
        return add_business_days(dt, term.term_length, bdc, calendar)
    val = dt + term
    return adjust_date(val, bdc, calendar)


def subtract_term(dt: DateNpType, term: Term, bdc: BdcLike, calendar: CalendarLike) -> DateNpType:
    """
    Subtract a ``Term`` offset from ``dt`` and adjust to a good business day.

    Parameters
    ----------
    dt : np.datetime64 or np.ndarray[datetime64[D]]
        Starting date(s).
    term : Term
        Offset to subtract (e.g. ``Term(6, TermType.Months)``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    np.datetime64 or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> subtract_term(np.datetime64("2025-07-15"), Term(6, TermType.Months), "F", "USD")
    numpy.datetime64('2025-01-15')

    >>> dates = np.array(["2025-07-15", "2025-10-15"], dtype="datetime64[D]")
    >>> subtract_term(dates, Term(6, TermType.Months), BDC.Following, "USD")
    array(['2025-01-15', '2025-04-15'], dtype='datetime64[D]')
    """
    return add_term(dt, -term, bdc, calendar)


def add_frequency(dt: DateNpType, frequency: Frequency, bdc: BdcLike, calendar: CalendarLike) -> DateNpType:
    """
    Advance ``dt`` by one period of ``frequency``.

    Parameters
    ----------
    dt : np.datetime64 or np.ndarray[datetime64[D]]
        Starting date(s).
    frequency : Frequency
        Period frequency (e.g. ``Frequency.Quarterly``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    np.datetime64 or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> add_frequency(np.datetime64("2025-03-31"), Frequency.Quarterly, "MF", "USD")
    numpy.datetime64('2025-06-30')

    >>> dates = np.array(["2025-03-31", "2025-06-30"], dtype="datetime64[D]")
    >>> add_frequency(dates, Frequency.Quarterly, BDC.ModifiedFollowing, "USD")
    array(['2025-06-30', '2025-09-30'], dtype='datetime64[D]')
    """
    term = Term.from_frequency(frequency)
    return add_term(dt, term, bdc, calendar)


def subtract_frequency(dt: DateNpType, frequency: Frequency, bdc: BdcLike, calendar: CalendarLike) -> DateNpType:
    """
    Move ``dt`` back by one period of ``frequency``.

    Parameters
    ----------
    dt : np.datetime64 or np.ndarray[datetime64[D]]
        Starting date(s).
    frequency : Frequency
        Period frequency (e.g. ``Frequency.Quarterly``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    np.datetime64 or np.ndarray[datetime64[D]]
        Resulting date(s).

    Examples
    --------
    >>> subtract_frequency(np.datetime64("2025-06-30"), Frequency.Quarterly, "MF", "USD")
    numpy.datetime64('2025-03-31')

    >>> dates = np.array(["2025-06-30", "2025-09-30"], dtype="datetime64[D]")
    >>> subtract_frequency(dates, Frequency.Quarterly, BDC.ModifiedFollowing, "USD")
    array(['2025-03-31', '2025-06-30'], dtype='datetime64[D]')
    """
    term = Term.from_frequency(frequency)
    return subtract_term(dt, term, bdc, calendar)