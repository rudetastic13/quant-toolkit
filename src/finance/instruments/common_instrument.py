from dataclasses import dataclass, field, replace

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
from finance.instruments.schedules.coupon_schedule import CouponSchedule
from finance.instruments.schedules.observation import LookbackStyle
from .enums import AmortizationType, CouponType, PrincipalExchange

# Flat coupon fields: the single-coupon convenience.  Changing one through ``replace`` on a
# leg whose schedule was derived from them rebuilds the schedule; see ``replace``.
_COUPON_FIELDS = frozenset({
    "coupon_type", "coupon_rate", "rate_index", "spread", "index_floor", "floor", "cap",
})


@dataclass(kw_only=True)
class CommonInstrument(CommonObject):
    """The one concrete leg type: schedule parameters + balance + coupon, flat.

    Two ways to say what the leg pays, one truth:

    - **Flat** (the simple path): ``coupon_type``, ``coupon_rate``, ``spread``, ``cap`` ...
      describe a single coupon for the whole life.  At construction they are lowered into a
      one-event :class:`CouponSchedule`.
    - **Schedule** (the general path): pass ``coupon_schedule`` directly for step-ups,
      fixed→float switches or anything piecewise.  The leg is then tagged
      ``coupon_type = CouponType.Custom`` and the other flat coupon fields are ignored.

    Either way ``coupon_schedule`` is what the pricer reads — the flat fields never reach the
    compiler, so the two cannot drift.  The leg-level projection parameters
    (``reset_frequency``, ``fixing_type``, ``rate_lookback``/``rate_lockout``,
    ``rate_calendar``) and ``rate_index`` apply to every floating segment.

    The same "flat type, or ``Custom`` + a supplied schedule" pattern is the intended shape
    for amortization: ``amortization_type`` names a generator (straight-line, level-pay, ...)
    or is ``Custom`` with a notional schedule attached.

    Designated refactor (when a second leg-bearing product lands — Bond/Loan): split the
    field groups below into ``kw_only`` dataclass mixins the products compose, defining
    each trait surface once.  ``kw_only`` keeps every construction site identical, so the
    split is mechanical.  Until a second product shares a field group, the mixins would be
    structure without a consumer — don't introduce them early.
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
    principal_exchange: PrincipalExchange = field(default=PrincipalExchange.Unused)

    # coupon info — leg-level projection parameters ...
    reset_frequency: Frequency | None = field(default=None)
    fixing_type: FixingType = field(default=FixingType.Arrears)
    rate_index: str | None = field(default=None)
    rate_lookback: Term | None = field(default=None)
    rate_lockout: Term | None = field(default=None)
    rate_calendar: str | None = field(default=None)
    # ISDA Lookback reads the overnight rate at each (shifted) sub-period start — the daily RFR
    # case; ObservationShift reads the whole (shifted) sub-period, which a term-rate compounder
    # (monthly fixings inside a quarterly coupon) needs.
    lookback_style: LookbackStyle = field(default=LookbackStyle.Lookback)
    # ... and the single-coupon convenience, lowered into ``coupon_schedule`` when none is given
    coupon_type: CouponType = field(default=CouponType.Zero)
    coupon_rate: float = field(default=0.0)
    spread: float | None = field(default=None)
    index_floor: float | None = field(default=None)
    floor: float | None = field(default=None)
    cap: float | None = field(default=None)

    # the source of truth the pricer reads; supplying one tags ``coupon_type = Custom``
    coupon_schedule: CouponSchedule | None = field(default=None)

    def __post_init__(self):
        if self.coupon_schedule is not None:
            self.coupon_type = CouponType.Custom
        elif self.coupon_type is CouponType.Custom:
            raise ValueError("coupon_type Custom means a supplied coupon_schedule; none was given")
        else:
            self.coupon_schedule = CouponSchedule.schedule_from_instrument(self)
        # invalid contracts fail at construction, whichever path built them
        # (raises ValidationException on failures; warnings pass through)
        self.validate()

    # -- what the leg pays, read off the schedule ------------------------------------------

    @property
    def is_floating(self) -> bool:
        return self.coupon_schedule.is_floating

    @property
    def needs_observation_grid(self) -> bool:
        return self.coupon_schedule.needs_observation_grid

    def serialize(self) -> dict:
        """A schedule is contract data only on a ``Custom`` leg; a flat leg rebuilds its own on load."""
        data = super().serialize()
        if self.coupon_type is not CouponType.Custom:
            del data["coupon_schedule"]
        return data

    def replace(self, **changes) -> "CommonInstrument":
        """``dataclasses.replace`` that keeps the schedule honest.

        A derived schedule is rebuilt when a flat coupon field changes (``coupon_rate=1.0``
        for a unit-coupon annuity, say); a ``Custom`` leg's supplied schedule is kept as is —
        it *is* the coupon, so pass a new ``coupon_schedule`` to change the coupon, or
        ``coupon_schedule=None`` with a flat ``coupon_type`` to go back to the flat path.
        """
        if self.coupon_type is not CouponType.Custom and _COUPON_FIELDS & changes.keys():
            changes.setdefault("coupon_schedule", None)
        return replace(self, **changes)

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
        if not self.coupon_schedule.events:
            result.add_failure("coupon_schedule has no events")
        if self.is_floating and not self.rate_index:
            floating = ", ".join(ct.name for ct in self.coupon_schedule.coupon_types if ct.is_floating)
            result.add_failure(f"floating coupon ({floating}) requires a rate_index")
        if self.cap is not None and self.floor is not None and self.cap < self.floor:
            result.add_failure(f"cap ({self.cap}) is below floor ({self.floor})")
        return result


__all__ = ["CommonInstrument"]
