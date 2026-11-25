"""``QuantToolkit.Dates.*`` worksheet functions (xlOil).

Dates cross the boundary as Excel serial numbers: ``Date.from_excel`` on the way in,
``Date.to_excel`` on the way out, so format the result cells as dates.

The scalar functions also take ranges. Arguments are broadcast against each other, so a column
of dates with one term gives a column, and a column of dates against a row of terms gives a
grid; the result spills.

Enums are passed by name (``"ModifiedFollowing"``, ``"Quarterly"``, ``"EOM"``, ``"Backward"``);
business day conventions also take their short code (``"MF"``) and rolls a day of the month.
Calendars are registry names in any case (``"US_FED"``), joined with ``+`` to combine them.
"""
from __future__ import annotations

import numpy as np

from quant_toolkit.finance import dates as qd
from quant_toolkit.finance.dates.schedules import generate_schedule

from quant_toolkit.xl.core.convert import enum_member as _enum
from quant_toolkit.xl.core.register import xl_func

_EPOCH_SERIAL = qd.Date.fromordinal(0).to_excel()  # schedule arrays are 1970-epoch day counts


def _calendar(name) -> str:
    """A calendar cell as the registry name.

    Parameters
    ----------
    name : str
        A calendar name in any case, or several joined with ``+`` (``"US_FED+US_NYSE"``).

    Returns
    -------
    str
        The lower-case name the calendar registry is keyed on (``"us_fed+us_nyse"``).
    """
    return str(name).strip().lower()


def _rules(bdc, calendar) -> tuple:
    """The (business day convention, calendar) pair the library's date functions take.

    Parameters
    ----------
    bdc : str
        Convention name or short code (``"ModifiedFollowing"``, ``"MF"``).
    calendar : str
        Calendar name, any case.

    Returns
    -------
    tuple of (BDC, str)
    """
    return _enum(qd.BDC, bdc), _calendar(calendar)


@xl_func("Dates", broadcast=True)
def AddBusinessDays(Date, Days, BusinessDayConvention, Calendar):  # noqa: N802, N803
    """Move a date by a number of business days on a calendar.

    Parameters
    ----------
    Date : date
        The start date. A range is taken element by element.
    Days : int
        Business days to move; negative goes back.
    BusinessDayConvention : str
        How the start is rolled if it is not a business day: "Following", "ModifiedFollowing"
        ("MF"), "Preceding", "NoAdjustment".
    Calendar : str
        Calendar name, e.g. "US_FED". Join several with "+".

    Returns
    -------
    date
        Excel serial number; format the cell as a date. Ranges give a range of the broadcast shape.

    Examples
    --------
    ``=QuantToolkit.Dates.AddBusinessDays(DATE(2026,6,1), 2, "MF", "US_FED")``
    """
    # through the numpy backend: the scalar Date one steps calendar days and ignores the sign of Days
    start = qd.Date.from_excel(Date).to_numpy()
    end = qd.add_business_days(start, int(Days), *_rules(BusinessDayConvention, Calendar))
    return qd.Date.from_numpy(end).to_excel()


@xl_func("Dates", broadcast=True)
def AdjustDate(Date, BusinessDayConvention, Calendar):  # noqa: N802, N803
    """Roll a date to a good business day on a calendar.

    Parameters
    ----------
    Date : date
        The date to adjust. A range is taken element by element.
    BusinessDayConvention : str
        "Following", "ModifiedFollowing" ("MF"), "Preceding" or "NoAdjustment".
    Calendar : str
        Calendar name, e.g. "US_FED". Join several with "+".

    Returns
    -------
    date
        The date itself if it is a business day, otherwise the rolled date, as an Excel serial number.
    """
    return qd.adjust_date(qd.Date.from_excel(Date), *_rules(BusinessDayConvention, Calendar)).to_excel()


@xl_func("Dates", broadcast=True)
def AddTerm(Date, Term, BusinessDayConvention, Calendar):  # noqa: N802, N803
    """Add a term to a date and roll the result to a good business day.

    Parameters
    ----------
    Date : date
        The start date. A range is taken element by element.
    Term : str
        A period such as "5D", "2BD", "1W", "3M", "1Y".
    BusinessDayConvention : str
        How the result is rolled: "Following", "ModifiedFollowing" ("MF"), "Preceding", "NoAdjustment".
    Calendar : str
        Calendar name, e.g. "US_FED". Join several with "+".

    Returns
    -------
    date
        Excel serial number. A column of dates against a row of terms gives a grid.

    Examples
    --------
    ``=QuantToolkit.Dates.AddTerm(A2:A10, B1:E1, "MF", "US_FED")`` spills a 9 by 4 grid.
    """
    start = qd.Date.from_excel(Date)
    return qd.add_term(start, qd.Term.from_str(Term), *_rules(BusinessDayConvention, Calendar)).to_excel()


@xl_func("Dates", broadcast=True)
def SubtractTerm(Date, Term, BusinessDayConvention, Calendar):  # noqa: N802, N803
    """Subtract a term from a date and roll the result to a good business day.

    Parameters
    ----------
    Date : date
        The start date. A range is taken element by element.
    Term : str
        A period such as "5D", "2BD", "1W", "3M", "1Y".
    BusinessDayConvention : str
        How the result is rolled: "Following", "ModifiedFollowing" ("MF"), "Preceding", "NoAdjustment".
    Calendar : str
        Calendar name, e.g. "US_FED". Join several with "+".

    Returns
    -------
    date
        Excel serial number. Ranges give a range of the broadcast shape.
    """
    start = qd.Date.from_excel(Date)
    return qd.subtract_term(start, qd.Term.from_str(Term), *_rules(BusinessDayConvention, Calendar)).to_excel()


@xl_func("Dates", broadcast=True)
def IsBusinessDay(Date, Calendar):  # noqa: N802, N803
    """Whether a date is a good business day on a calendar.

    Parameters
    ----------
    Date : date
        The date to test. A range is taken element by element.
    Calendar : str
        Calendar name, e.g. "US_FED". Join several with "+".

    Returns
    -------
    bool
        TRUE if the date is neither a weekend nor a holiday on the calendar.
    """
    return bool(qd.is_good_bd(qd.Date.from_excel(Date), _calendar(Calendar)))


@xl_func("Dates", broadcast=True)
def NextImmDate(Date, Frequency):  # noqa: N802, N803
    """First IMM date (third Wednesday) strictly after a date.

    Parameters
    ----------
    Date : date
        The reference date. A range is taken element by element.
    Frequency : str
        The IMM cycle: "Quarterly" (March, June, September, December) or "Monthly".

    Returns
    -------
    date
        Excel serial number. A date that is itself an IMM date gives the following one.
    """
    return qd.next_imm_date(qd.Date.from_excel(Date), _enum(qd.Frequency, Frequency)).to_excel()


@xl_func("Dates", broadcast=True)
def PriorImmDate(Date, Frequency):  # noqa: N802, N803
    """Last IMM date (third Wednesday) strictly before a date.

    Parameters
    ----------
    Date : date
        The reference date. A range is taken element by element.
    Frequency : str
        The IMM cycle: "Quarterly" (March, June, September, December) or "Monthly".

    Returns
    -------
    date
        Excel serial number. A date that is itself an IMM date gives the preceding one.
    """
    return qd.prior_imm_date(qd.Date.from_excel(Date), _enum(qd.Frequency, Frequency)).to_excel()


@xl_func("Dates")
def UnadjustedSchedule(StartDate, EndDate, Frequency, RollConvention, Direction):  # noqa: N802, N803
    """Unadjusted period dates between two dates, as a column.

    Parameters
    ----------
    StartDate : date
        First date of the schedule.
    EndDate : date
        Last date of the schedule.
    Frequency : str
        Period length: "Monthly", "Quarterly", "SemiAnnually", "Annually".
    RollConvention : str or int
        "Empty" (roll on the anchor date's day), "EOM", "IMM", or a day of the month.
    Direction : str
        "Forward" generates from the start (stub at the end); "Backward" from the end (stub at the front).

    Returns
    -------
    column of dates
        Start date, each period boundary, end date, as Excel serial numbers. No business day
        adjustment is applied; pass the result to AdjustDate for that.

    Examples
    --------
    ``=QuantToolkit.Dates.UnadjustedSchedule(DATE(2026,6,1), DATE(2027,6,1), "Quarterly", "Empty", "Backward")``
    """
    schedule = generate_schedule(
        qd.Date.from_excel(StartDate),
        qd.Date.from_excel(EndDate),
        _enum(qd.Frequency, Frequency),
        roll_convention=_enum(qd.Roll, RollConvention),
        direction=_enum(qd.Direction, Direction),
    )
    return (schedule.astype(np.int64) + _EPOCH_SERIAL).astype(np.float64).reshape(-1, 1)


@xl_func("Dates")
def AdjustedSchedule(  # noqa: N802
    StartDate,
    EndDate,
    Frequency,
    RollConvention,
    Direction,
    BusinessDayConvention,
    Calendar,
    ShowStartEnd=None,
):
    """Period dates between two dates, each rolled to a good business day, as a column.

    Parameters
    ----------
    StartDate : date
        First date of the schedule.
    EndDate : date
        Last date of the schedule.
    Frequency : str
        Period length: "Monthly", "Quarterly", "SemiAnnually", "Annually".
    RollConvention : str or int
        "Empty" (roll on the anchor date's day), "EOM", "IMM", or a day of the month.
    Direction : str
        "Forward" generates from the start (stub at the end); "Backward" from the end (stub at the front).
    BusinessDayConvention : str
        How each date is rolled: "Following", "ModifiedFollowing" ("MF"), "Preceding", "NoAdjustment".
    Calendar : str
        Calendar name, e.g. "US_FED". Join several with "+".
    ShowStartEnd : bool, optional
        TRUE returns one row per period with its start and end date. FALSE or omitted returns the dates in one column.

    Returns
    -------
    column of dates, or two columns
        The unadjusted schedule's dates after rolling, as Excel serial numbers, with any dates
        that roll onto the same day merged. With ShowStartEnd, column 1 is each period's start
        and column 2 its end, so there is one row fewer.

    Examples
    --------
    ``=QuantToolkit.Dates.AdjustedSchedule(A1, A2, "Quarterly", "Empty", "Backward", "MF", "US_FED", TRUE)``
    """
    schedule = generate_schedule(
        qd.Date.from_excel(StartDate),
        qd.Date.from_excel(EndDate),
        _enum(qd.Frequency, Frequency),
        roll_convention=_enum(qd.Roll, RollConvention),
        direction=_enum(qd.Direction, Direction),
    )
    adjusted = np.unique(qd.adjust_date(schedule, *_rules(BusinessDayConvention, Calendar)))
    serials = (adjusted.astype(np.int64) + _EPOCH_SERIAL).astype(np.float64)
    if ShowStartEnd:
        return np.column_stack([serials[:-1], serials[1:]])
    return serials.reshape(-1, 1)
