"""Numpy based date utility implementations"""
from typing import overload
import numpy as np
from .helpers import CalendarLike, BdcLike
from .helpers import calendar as clean_calendar, bdc as clean_bdc
from finance.dates import BDC, Term, TermType, Frequency
from finance.dates.types import DateNpType, BoolNpType

_NP_BDC_MAP: dict[BDC, str | None] = {
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
    np_calendar = clean_calendar(calendar).np_calendar
    return np.is_busday(dt, np_calendar.weekmask, np_calendar.holidays)


@overload
def adjust_date(dt: np.datetime64, bdc: BdcLike, calendar: CalendarLike) -> np.datetime64:
    ...

@overload
def adjust_date(dt: np.ndarray, bdc: BdcLike, calendar: CalendarLike) -> np.ndarray:
    ...


def adjust_date(dt: DateNpType, bdc: BdcLike, calendar: CalendarLike) -> DateNpType:
    np_calendar: np.busdaycalendar = clean_calendar(calendar).np_calendar
    bdc: str | None = _NP_BDC_MAP[clean_bdc(bdc)]
    if not bdc:
        return dt
    return np.busday_offset(
        dt,
        offsets=0,
        roll=bdc,
        weekmask=np_calendar.weekmask,
        holidays=np_calendar.holidays
    )

def add_business_days(dt: DateNpType, days: int, bdc:BdcLike, calendar: CalendarLike) -> DateNpType:
    np_calendar: np.busdaycalendar = clean_calendar(calendar).np_calendar
    bdc: str | None = _NP_BDC_MAP[clean_bdc(bdc)]
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
    if term.term_type == TermType.BusinessDays:
        return add_business_days(dt, term.term_length, bdc, calendar)
    bdc: str | None = _NP_BDC_MAP[clean_bdc(bdc)]
    val = dt + term
    return adjust_date(val, bdc, calendar)

def subtract_term(dt: DateNpType, term: Term, bdc: BdcLike, calendar: CalendarLike) -> DateNpType:
    return add_term(dt, -term, bdc, calendar)

def add_frequency(dt: DateNpType, frequency: Frequency, bdc: BdcLike, calendar: CalendarLike) -> DateNpType:
    term = Term.from_frequency(frequency)
    return add_term(dt, term, bdc, calendar)