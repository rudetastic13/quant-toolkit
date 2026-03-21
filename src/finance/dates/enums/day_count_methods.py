from enum import Enum, auto

class DayCountMethod(Enum):
    Unused = auto()
    Actual360 = auto()
    Actual365 = auto()
    ActualActual = auto()
    Thirty360 = auto()
    ThirtyE360 = auto()
    ThirtyE360ISDA = auto()
    ThirtyE365 = auto()
    Thirty365 = auto()
    Bus252 = auto()

    def is_supported(self) -> bool:
        return self.value >= 0
