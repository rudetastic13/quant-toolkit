from enum import IntEnum, auto

class CurveType(IntEnum):
    """Enumeration of zero curve constructor"""
    Discount = auto()
    Forward = auto()
    DualCurve = auto()

class ValueType(IntEnum):
    """Enumeration of rate value types for financial curves"""
    ZeroRate = auto()
    DiscountFactor = auto()
    LogDiscountFactor = auto()

class RateType(IntEnum):
    """Enumeration of rate compounding types for financial curves"""
    Cash = auto()
    DailyCompounded = auto()
    DailyAveraged = auto()
    Swap = auto()
    IborFallback = auto()

class CurveInterpolator(IntEnum):
    """Enumeration of interpolation methods for financial curves"""
    LinearZero = auto()
    LinearLogDF = auto()
    FlatLogDF = auto()
    FlatZero = auto()

class CurveExtrapolator(IntEnum):
    """Enumeration of extrapolation methods for financial curves"""
    NotAllowed = auto()
    FlatForward = auto()

