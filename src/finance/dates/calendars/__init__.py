"""Module for calendar definitions and singleton calendar instances"""
from typing import TypeVar, Generic
from common.singleton import Singleton
from finance.dates.calendars.calendar import Calendar, StaticCalendar

# define the no holidays calendar instance
NoHolidaysCalendar = StaticCalendar(
    name="no_holidays",
    week_mask="1111100",
    holidays=set()
)

class Calendars(Singleton):
    """The calendars singleton registry"""
    _calendars: dict[str, Calendar] = {}

    def __init__(self):
        Calendars._calendars["no_holidays"] = NoHolidaysCalendar

    @classmethod
    def register(cls, cal: Calendar) -> None:
        """Register a calendar with a name"""
        cls._calendars[cal.name.lower()] = cal

    @classmethod
    def get(cls, name: str) -> Calendar:
        """Get a calendar by name"""
        if "+" in name:
            parts = [p.strip().lower() for p in name.split("+")]
            parts.sort()
            joined_name = "+".join(parts)
            if joined_name in cls._calendars:
                return cls._calendars[joined_name]
            else:
                combined = cls._calendars[parts[0]]
                for part in parts[1:]:
                    combined = combined + cls._calendars[part]
                cls.register(combined)
                name = joined_name
        return cls._calendars[name]

    @classmethod
    def list_calendars(cls) -> list[str]:
        return list(cls._calendars.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered calendars - for testing purposes only"""
        cls._calendars.clear()
        del cls._instances[cls]

__all__ = [
    "Calendar",
    "Calendars",
]