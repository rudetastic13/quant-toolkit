from dataclasses import dataclass, field

from common.object import ValidationResult
from common.object.common_object import CommonObject
from finance.dates import (
    Date,
    Term,
    DayCountMethod,
    BDC,
    Roll,
    Direction,
    Frequency,
)
from .enums import AmortizationType, CouponType

@dataclass(kw_only=True)
class CommonInstrument(CommonObject):

    # for grid generation
    effective: Date
    maturity: Date
    currency: str
    notional: float
    payment_frequency: Frequency
    day_count_method: DayCountMethod = field(default=DayCountMethod.Unused)
    business_day_convention: BDC = field(default=BDC.NoAdjustment)
    roll_convention: Roll = field(default=Roll.Empty)
    direction: Direction = field(default=Direction.Forward)
    pay_calendar: str = field(default="no_holidays")
    payment_delay: Term | None = field(default=None)
    first_regular_accrual_date: Date | None = field(default=None)
    last_regular_accrual_date: Date | None = field(default=None)

    # balance info
    original_notional: float | None = field(default=None)
    amortization_type: AmortizationType = field(default=AmortizationType.Unused)
    amortization_start: Date | None = field(default=None)
    amortization_end: Date | None = field(default=None)

    # coupon info
    reset_frequency: Frequency | None = field(default=None)
    coupon_rate: float = field(default=0.0)
    coupon_type: CouponType | None = field(default=CouponType.Zero)
    rate_index: str | None = field(default=None)
    spread: float | None = field(default=None)
    index_floor: float | None = field(default=None)
    floor: float | None = field(default=None)
    cap: float | None = field(default=None)
    rate_lookback: Term | None = field(default=None)
    rate_lockout: Term | None = field(default=None)
    rate_calendar: str | None = field(default=None)

    # pre-computed schedule info
    schedules: dict = field(default_factory=dict)

    def _validate_impl(self) -> ValidationResult:
        pass

