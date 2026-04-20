"""Define common calendar implementations"""
from __future__ import annotations
from typing import Protocol, Sequence, runtime_checkable
import numpy as np
from finance.dates import Date
from dataclasses import dataclass, field

@runtime_checkable
class Calendar(Protocol):
    """Protocol for a calendar, which defines the holidays and business day logic"""
    name: str

    def is_weekday(self, date: Date) -> bool:
        """Return the weekday of a given date, where 0 is Monday and 6 is Sunday"""
        ...

    def is_holiday(self, date: Date) -> bool:
        """Check if a given date is a holiday"""
        ...

    def is_business_day(self, date: Date) -> bool:
        """Check if a given date is a business day"""
        ...

    @property
    def np_calendar(self) -> np.busdaycalendar:
        """Return the numpy busdaycalendar representation of the calendar"""
        ...

@dataclass
class StaticCalendar(Calendar):
    name: str
    week_mask: Sequence[int | str | bool]
    holidays: set[Date] = field(init=True, default_factory=set)
    _holidays_arr: np.ndarray = field(init=False, repr=False)
    _np_calendar: np.busdaycalendar = field(init=False, repr=False)

    @staticmethod
    def _holidays_to_arr(dt_set: set[Date]) -> np.ndarray:
        return np.array([holiday.to_str() for holiday in dt_set], dtype="datetime64[D]")

    def __post_init__(self):
        if len(self.week_mask) != 7:
            raise ValueError(f"Week mask must be of length 7, got {len(self.week_mask)}")
        self.week_mask = "".join(str(int(c)) for c in self.week_mask)
        self._holidays_arr = self._holidays_to_arr(self.holidays)
        self._np_calendar = np.busdaycalendar(
            weekmask=self.week_mask,
            holidays=self._holidays_arr
        )

    def is_holiday(self, date: Date) -> bool:
        return date in self.holidays

    def is_weekday(self, date: Date) -> bool:
        return self.week_mask[date.isoweekday()] == "1"

    def is_business_day(self, date: Date) -> bool:
        return self.is_weekday(date) and not self.is_holiday(date)

    @property
    def np_calendar(self) -> np.busdaycalendar:
        return self._np_calendar

    def add_holidays(self, *holidays: Date):
        """Add holidays to the calendar, and update the numpy calendar accordingly"""
        self.holidays.update(holidays)
        self._holidays_arr = self._holidays_to_arr(self.holidays)
        self._np_calendar = np.busdaycalendar(
            weekmask=self.week_mask,
            holidays=self._holidays_arr
        )

    def remove_holidays(self, *holidays: Date):
        """Remove holidays from the calendar, and update the numpy calendar accordingly"""
        self.holidays.difference_update(holidays)
        self._holidays_arr = self._holidays_to_arr(self.holidays)
        self._np_calendar = np.busdaycalendar(
            weekmask=self.week_mask,
            holidays=self._holidays_arr
        )

    def __add__(self, other: StaticCalendar) -> StaticCalendar:
        """Combine two calendars by unioning their holidays and taking the max of their week masks"""
        if not isinstance(other, StaticCalendar):
            raise TypeError(f"Can only combine with another StaticCalendar, got {type(other)}")
        combined_week_mask = "".join(
            str(int(min(int(a), int(b))))
            for a, b in zip(self.week_mask, other.week_mask)
        )
        combined_holidays = self.holidays.union(other.holidays)
        return StaticCalendar(
            name=f"{self.name}+{other.name}",
            week_mask=combined_week_mask,
            holidays=combined_holidays
        )

