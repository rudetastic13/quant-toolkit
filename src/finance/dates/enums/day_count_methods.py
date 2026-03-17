from enum import Enum, auto

class DayCountMethod(Enum):
    Bus252 = -1
    Unused = auto()
    Actual360 = auto()
    Actual365 = auto()
    ActualActual = auto()
    Thirty360 = auto()
    ThirtyE360 = auto()
    ThirtyE360ISDA = auto()
    ThirtyE365 = auto()
    Thirty365 = auto()

    def is_supported(self) -> bool:
        return self.value >= 0
