"""
Scalar ``Date`` Utilities
=========================

Business-day checks, date adjustment by convention, and calendar-aware
date arithmetic for scalar :class:`~finance.dates.Date` inputs.

All public functions accept ``str`` or typed arguments for ``calendar``
and ``bdc``; raw strings are resolved via the helpers in this package.
"""

from finance.dates import (
    Date,
    Term,
    TermType,
    BDC,
    Calendar,
    Frequency,
)
from .helpers import CalendarLike, BdcLike
from .helpers import (
    calendar as clean_calendar,
    bdc as clean_bdc
)
_1D_OFFSET = Term(1, TermType.Days)


def is_good_bd(dt: Date, calendar: CalendarLike) -> bool:
    """
    Check whether ``dt`` falls on a good business day.

    Parameters
    ----------
    dt : Date
        Scalar date to check.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    bool
        ``True`` if ``dt`` is a good business day.

    Examples
    --------
    >>> is_good_bd(Date(2025, 12, 25), "USD")
    False
    """
    return clean_calendar(calendar).is_business_day(dt)


def _following(dt: Date, calendar: Calendar):
    while not calendar.is_business_day(dt):
        dt += _1D_OFFSET
    return dt

def _preceding(dt, calendar: Calendar):
    while not calendar.is_business_day(dt):
        dt -= _1D_OFFSET
    return dt

def _mod_follow(dt, calendar: Calendar):
    end = _following(dt, calendar)
    if dt.month != end.month:
        end = _preceding(dt, calendar)
    return end

def _mod_preceding(dt, calendar: Calendar):
    end = _preceding(dt, calendar)
    if dt.month != end.month:
        end = _following(dt, calendar)
    return end

def _adjust_dt(dt, bdc: BDC, calendar: Calendar):
    func_map = {
        BDC.NoAdjustment: lambda x, _: x,
        BDC.Following: _following,
        BDC.Preceding: _preceding,
        BDC.ModifiedFollowing: _mod_follow,
        BDC.ModifiedPreceding: _mod_preceding,
    }
    return func_map[bdc](dt, calendar)


def adjust_date(dt: Date, bdc: BdcLike, calendar: CalendarLike) -> Date:
    """
    Roll ``dt`` to the nearest good business day using ``bdc`` convention.

    Parameters
    ----------
    dt : Date
        Scalar date to adjust.
    bdc : str or BDC
        Business day convention (e.g. ``BDC.ModifiedFollowing`` or ``"MF"``).
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date
        Adjusted date.

    Examples
    --------
    >>> adjust_date(Date(2025, 8, 30), BDC.ModifiedFollowing, "USD")
    Date(2025, 8, 29)
    """
    bdc = clean_bdc(bdc)
    calendar = clean_calendar(calendar)
    return _adjust_dt(dt, bdc, calendar)


def add_business_days(dt: Date, days: int, bdc: BdcLike, calendar: CalendarLike) -> Date:
    """
    Advance ``dt`` by ``days`` business days.

    Parameters
    ----------
    dt : Date
        Starting date.
    days : int
        Number of business days to advance (negative to go back).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date
        Resulting date.

    Examples
    --------
    >>> add_business_days(Date(2025, 1, 2), 5, BDC.Following, "USD")
    Date(2025, 1, 9)
    """
    bdc = clean_bdc(bdc)
    calendar = clean_calendar(calendar)
    if not bdc:
        return dt + Term(days, TermType.Days)
    dt = adjust_date(dt, bdc, calendar)
    func = lambda x, y: x + Term(y, TermType.Days) if y > 0 else x - Term(y, TermType.Days)
    days = abs(days)
    while days != 0 and calendar.is_business_day(dt):
        dt = func(dt, 1)
        days -= 1
    return dt


def add_term(dt, term: Term, bdc: BdcLike, calendar: CalendarLike) -> Date:
    """
    Add a ``Term`` offset to ``dt`` and adjust to a good business day.

    Parameters
    ----------
    dt : Date
        Starting date.
    term : Term
        Offset to add (e.g. ``Term(3, TermType.Months)``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date
        Resulting date.

    Examples
    --------
    >>> add_term(Date(2025, 1, 15), Term(3, TermType.Months), BDC.ModifiedFollowing, "USD")
    Date(2025, 4, 15)
    """
    if term.term_type == TermType.BusinessDays:
        return add_business_days(dt, term.term_length, bdc, calendar)
    bdc = clean_bdc(bdc)
    calendar = clean_calendar(calendar)
    val = dt + term
    return adjust_date(val, bdc, calendar)


def subtract_term(dt: Date, term: Term, bdc: BdcLike, calendar: CalendarLike) -> Date:
    """
    Subtract a ``Term`` offset from ``dt`` and adjust to a good business day.

    Parameters
    ----------
    dt : Date
        Starting date.
    term : Term
        Offset to subtract (e.g. ``Term(6, TermType.Months)``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date
        Resulting date.

    Examples
    --------
    >>> subtract_term(Date(2025, 7, 15), Term(6, TermType.Months), BDC.Following, "USD")
    Date(2025, 1, 15)
    """
    return add_term(dt, -term, bdc, calendar)


def add_frequency(dt: Date, frequency: Frequency, bdc: BdcLike, calendar: CalendarLike) -> Date:
    """
    Advance ``dt`` by one period of ``frequency``.

    Parameters
    ----------
    dt : Date
        Starting date.
    frequency : Frequency
        Period frequency (e.g. ``Frequency.Quarterly``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date
        Resulting date.

    Examples
    --------
    >>> add_frequency(Date(2025, 3, 31), Frequency.Quarterly, BDC.ModifiedFollowing, "USD")
    Date(2025, 6, 30)
    """
    term = Term.from_frequency(frequency)
    return add_term(dt, term, bdc, calendar)


def subtract_frequency(dt: Date, frequency: Frequency, bdc: BdcLike, calendar: CalendarLike) -> Date:
    """
    Move ``dt`` back by one period of ``frequency``.

    Parameters
    ----------
    dt : Date
        Starting date.
    frequency : Frequency
        Period frequency (e.g. ``Frequency.Quarterly``).
    bdc : str or BDC
        Business day convention.
    calendar : str or Calendar
        Holiday calendar name or instance.

    Returns
    -------
    Date
        Resulting date.

    Examples
    --------
    >>> subtract_frequency(Date(2025, 6, 30), Frequency.Quarterly, BDC.ModifiedFollowing, "USD")
    Date(2025, 3, 31)
    """
    term = Term.from_frequency(frequency)
    return add_term(dt, -term, bdc, calendar)