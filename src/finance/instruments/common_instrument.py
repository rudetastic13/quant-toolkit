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
from finance.dates.enums import FixingType
from .enums import AmortizationType, CouponType

@dataclass(kw_only=True)
class CommonInstrument(CommonObject):
    """The one concrete leg type: schedule + balance + coupon parameters, flat.

    Designated refactor (when a second leg-bearing product lands — Bond/Loan): split the
    field groups below into ``kw_only`` dataclass mixins the products compose, defining
    each trait surface once::

        @dataclass(kw_only=True)
        class ScheduleParamsMixin: effective: Date; maturity: Date; ...
        @dataclass(kw_only=True)
        class FloatingRateMixin: rate_index: str | None = None; ...

        @dataclass(kw_only=True)
        class CommonInstrument(ScheduleParamsMixin, NotionalMixin, FixedRateMixin,
                               FloatingRateMixin, AmortizationMixin, CommonObject): ...

    ``kw_only`` keeps every construction site identical, so the split is mechanical.  Until
    a second product shares a field group, the mixins would be structure without a consumer
    — don't introduce them early (a property-only Protocol layer was tried and deleted for
    exactly that reason).
    """

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
    fixing_type: FixingType = field(default=FixingType.Arrears)
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
        result = ValidationResult()
        if self.maturity <= self.effective:
            result.add_failure(
                f"maturity ({self.maturity}) must be after effective ({self.effective})"
            )
        if self.notional < 0:
            result.add_warning(
                "notional is negative; pay/receive direction is set at the product level "
                "(negative notional = pay), so legs should carry a positive notional"
            )
        if self.coupon_type.is_floating and not self.rate_index:
            result.add_failure(f"floating coupon ({self.coupon_type.name}) requires a rate_index")
        if self.cap is not None and self.floor is not None and self.cap < self.floor:
            result.add_failure(f"cap ({self.cap}) is below floor ({self.floor})")
        return result

