from typing import TypeVar
from finance.dates.calendars import Calendar, Calendars
from finance.dates.enums import BDC, Frequency

BdcLike = TypeVar("BdcLike", bound=BDC | str)
CalendarLike = TypeVar("CalendarLike", bound=Calendar | str)


def imm_month_step(frequency: Frequency) -> int:
    """Month step of an IMM cycle; only month-based frequencies dividing a year qualify.

    The cycle is anchored so December is always on-cycle: Quarterly gives the
    Mar/Jun/Sep/Dec (H/M/U/Z) futures months, Monthly gives all serial months.
    """
    freq_type, months = frequency.int_based_mapping()
    if freq_type != 1 or 12 % months != 0:
        raise ValueError(f"IMM cycle requires a month-based frequency dividing one year, got {frequency.name}")
    return months

def calendar(val: CalendarLike) -> Calendar:
    if type(val) is str:
        return Calendars.get(val)
    elif isinstance(val, Calendar):
        return val
    raise TypeError("Calendar provided is not a string or Calendar")

def bdc(val: BdcLike) -> BDC:
    if type(val) is str:
        return BDC[val]
    elif type(val) is BDC:
        return val
    raise TypeError("BDC provided is not a string or BDC")



