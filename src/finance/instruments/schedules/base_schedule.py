from dataclasses import dataclass, field
from typing import Any
from finance.dates import Date

@dataclass
class BaseEvent:
    """Base class for defining a schedule of misc info, anchors to accrual grid"""
    start_date: Date = field(init=True)

    def __lt__(self, other: Any) -> bool:
        # Events in a schedule are ordered by their anchor date regardless of subtype, so a
        # heterogeneous schedule (e.g. fixed + compounded events) sorts cleanly.
        if not isinstance(other, BaseEvent):
            raise ValueError(f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}")
        return self.start_date < other.start_date

@dataclass
class BaseSchedule:
    events: list[BaseEvent] = field(default_factory=list)

    def __post_init__(self):
        self.events.sort()

