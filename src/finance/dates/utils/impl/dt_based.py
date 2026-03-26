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
    bdc = clean_bdc(bdc)
    calendar = clean_calendar(calendar)
    return _adjust_dt(dt, bdc, calendar)

def add_business_days(dt: Date, days: int, bdc: BdcLike, calendar: CalendarLike) -> Date:
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
    if term.term_type == TermType.BusinessDays:
        return add_business_days(dt, term.term_length, bdc, calendar)
    bdc = clean_bdc(bdc)
    calendar = clean_calendar(calendar)
    val = dt + term
    return adjust_date(val, bdc, calendar)

def subtract_term(dt: Date, term: Term, bdc: BdcLike, calendar: CalendarLike) -> Date:
    return add_term(dt, -term, bdc, calendar)

def add_frequency(dt: Date, frequency: Frequency, bdc: BdcLike, calendar: CalendarLike) -> Date:
    term = Term.from_frequency(frequency)
    return add_term(dt, term, bdc, calendar)