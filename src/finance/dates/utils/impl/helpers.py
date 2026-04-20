from typing import TypeVar
from finance.dates.calendars import Calendar, Calendars
from finance.dates.enums import BDC

BdcLike = TypeVar("BdcLike", bound=BDC | str)
CalendarLike = TypeVar("CalendarLike", bound=Calendar | str)

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



