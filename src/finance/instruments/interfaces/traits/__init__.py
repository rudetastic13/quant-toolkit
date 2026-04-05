from finance.instruments.interfaces.traits.schedule import HasScheduleParams, HasStubDates
from finance.instruments.interfaces.traits.rate import HasFixedRate, HasFloatingRate, HasRateBounds
from finance.instruments.interfaces.traits.balance import HasNotional, HasAmortization

__all__ = [
    "HasScheduleParams",
    "HasStubDates",
    "HasFixedRate",
    "HasFloatingRate",
    "HasRateBounds",
    "HasNotional",
    "HasAmortization",
]
