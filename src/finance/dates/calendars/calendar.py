"""Define common calendar implementations"""
from typing import Protocol, Sequence
import numpy as np
from finance.dates import Date

class ACalendar(Protocol):
    """Protocol for a calendar, which defines the holidays and business day logic"""
    def weekday(self, date: Date) -> bool:
        """Return the weekday of a given date, where 0 is Monday and 6 is Sunday"""
        ...

    def is_holiday(self, date: Date) -> bool:
        """Check if a given date is a holiday"""
        ...

    def is_business_day(self, date: Date) -> bool:
        """Check if a given date is a business day"""
        return not self.is_holiday(date)

    @property
    def holidays(self) -> set[Date]:
        """Return the set of holidays in the calendar"""
        ...

    @property
    def business_days(self) -> str:
        """Return the string representation of business days in the calendar, e.g. "Mon-Sun" """
        ...

    @property
    def np_calendar(self) -> np.busdaycalendar:
        """Return the numpy busdaycalendar representation of the calendar"""
        ...
