from enum import Enum, auto

class DayCountMethod(Enum):
    Unused = auto()
    Actual360 = auto()
    Actual365 = auto()
    ActualActual = auto()
    Thirty360 = auto()
    ThirtyE360 = auto()
    ThirtyE360ISDA = auto()
    Thirty365 = auto()
