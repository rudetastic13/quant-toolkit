"""Module for calendar definitions and singleton calendar instances"""
from finance.dates.calendars.calendar import GenericCalendar, StaticCalendar
from common.singleton import Singleton

# define the no holidays calendar instance
NoHolidaysCalendar = StaticCalendar(
    name="no_holidays",
    week_mask="1111100",
    holidays=set()
)

class Calendars(Singleton):
    """The calendars singleton registry"""
    _calendars: dict[str, GenericCalendar] = {}
    _initialized: bool = False

    def __init__(self):
        if not Calendars._initialized:
            Calendars._calendars["no_holidays"] = NoHolidaysCalendar
            Calendars._initialized = True


    @classmethod
    def register(cls, name: str, cal: GenericCalendar) -> None:
        """Register a calendar with a name"""
        cls._calendars[name.lower()] = cal

    @classmethod
    def get(cls, name: str) -> GenericCalendar:
        """Get a calendar by name"""
        if "+" in name:
            parts = [p.strip().lower() for p in name.split("+")]
            combined = cls._calendars[parts[0]]
            for part in parts[1:]:
                combined = combined + cls._calendars[part]
            name = " + ".join(parts)
            cls.register(name, combined)
        return cls._calendars[name]

    @classmethod
    def list_calendars(cls) -> list[str]:
        return list(cls._calendars.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered calendars - for testing purposes only"""
        cls._calendars.clear()
        cls._initialized = False

__all__ = [
    "GenericCalendar",
    "Calendars",
]