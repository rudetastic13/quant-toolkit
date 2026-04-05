from finance.instruments.interfaces.traits import (
    HasScheduleParams,
    HasStubDates,
    HasFixedRate,
    HasFloatingRate,
    HasRateBounds,
    HasNotional,
    HasAmortization,
)
from finance.instruments.interfaces.products import Bond, Loan, FRA, Future, SwapLeg, Swap

__all__ = [
    # Traits
    "HasScheduleParams",
    "HasStubDates",
    "HasFixedRate",
    "HasFloatingRate",
    "HasRateBounds",
    "HasNotional",
    "HasAmortization",
    # Products
    "Bond",
    "Loan",
    "FRA",
    "Future",
    "SwapLeg",
    "Swap",
]
