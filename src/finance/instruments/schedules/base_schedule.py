from dataclasses import dataclass, field
from typing import Any, Self
from finance.dates import Date

@dataclass
class BaseEvent:
    """Base class for defining a schedule of misc info, anchors to accrual grid"""
    start_date: Date = field(init=True)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            raise ValueError(f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}")
        other: Self = other
        return self.start_date < other.start_date

@dataclass
class BaseSchedule:
    events: list[BaseEvent] = field(default_factory=list)

    def __post_init__(self):
        self.events.sort()

