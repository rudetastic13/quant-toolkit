from finance.instruments.common_instrument import CommonInstrument
from finance.instruments.enums import AmortizationType, CouponType, LegType
from finance.instruments.interfaces import (
    # Traits
    HasScheduleParams,
    HasStubDates,
    HasFixedRate,
    HasFloatingRate,
    HasRateBounds,
    HasNotional,
    HasAmortization,
    # Products
    Bond,
    Loan,
    FRA,
    Future,
    SwapLeg,
    Swap,
)
