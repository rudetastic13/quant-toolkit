"""Market conventions — per-leg and per-product convention descriptors.

Three tiers, composed bottom-up:

- :class:`RateIndex` (``rate_index.py``) — the index itself: projection day count,
  fixing calendar, fixing style.
- Leg conventions — :class:`FixedLegConventions` / :class:`FloatLegConventions`: the
  standard shape of one leg.  Field names mirror ``CommonInstrument`` so applying a leg
  convention to a leg is a straight field-for-field mapping.
- Product conventions — :class:`SwapConventions`, :class:`DepositConventions`,
  :class:`FraConventions`: leg conventions plus product-level dating (spot lag).

One :class:`MarketConventions` bundle per (currency, index) is what the
``ConventionRegistry`` stores: builders look up the bundle once and read their product's
section.  ``None`` on a leg/product day count means "use the index's" — the resolved
accessors on the bundle apply that defaulting.
"""
from __future__ import annotations

from dataclasses import dataclass

from finance.dates import Term
from finance.dates.enums import BDC, DayCountMethod, Frequency, Roll
from finance.instruments.enums import CouponType

from finance.conventions.rate_index import RateIndex


@dataclass(frozen=True, kw_only=True)
class FixedLegConventions:
    """Standard shape of a fixed leg."""

    payment_frequency: Frequency
    day_count_method: DayCountMethod
    business_day_convention: BDC
    roll_convention: Roll
    pay_calendar: str
    payment_delay: Term | None = None


@dataclass(frozen=True, kw_only=True)
class FloatLegConventions:
    """Standard shape of a floating leg.

    ``coupon_type`` is a convention, not a builder decision: it is exactly what differs
    between a compounded-in-arrears RFR leg (``GeometricAveraged``) and an
    arithmetic-average leg.  ``day_count_method=None`` means "use the index's" — the
    accrual basis then matches the projection basis by construction.
    """

    payment_frequency: Frequency
    coupon_type: CouponType
    reset_frequency: Frequency
    business_day_convention: BDC
    roll_convention: Roll
    pay_calendar: str
    day_count_method: DayCountMethod | None = None
    payment_delay: Term | None = None
    rate_lookback: Term | None = None
    rate_lockout: Term | None = None


@dataclass(frozen=True, kw_only=True)
class SwapConventions:
    """Standard fixed-vs-float swap: one convention set per leg plus spot dating."""

    fixed_leg: FixedLegConventions
    float_leg: FloatLegConventions
    spot_lag: Term
    spot_calendar: str


@dataclass(frozen=True, kw_only=True)
class DepositConventions:
    """Spot-starting cash deposit."""

    spot_lag: Term
    business_day_convention: BDC
    calendar: str
    day_count_method: DayCountMethod | None = None  # None -> index's


@dataclass(frozen=True, kw_only=True)
class FraConventions:
    """Forward rate agreement dated relative to spot."""

    spot_lag: Term
    business_day_convention: BDC
    calendar: str
    day_count_method: DayCountMethod | None = None  # None -> index's


@dataclass(frozen=True, kw_only=True)
class MarketConventions:
    """Everything the builders need for one (currency, index): the registry's unit.

    The resolved accessors apply the ``None``-means-index defaulting so builders never
    re-implement it.
    """

    index: RateIndex
    swap: SwapConventions
    deposit: DepositConventions
    fra: FraConventions

    @property
    def float_leg_day_count(self) -> DayCountMethod:
        dc = self.swap.float_leg.day_count_method
        return dc if dc is not None else self.index.day_count_method

    @property
    def deposit_day_count(self) -> DayCountMethod:
        dc = self.deposit.day_count_method
        return dc if dc is not None else self.index.day_count_method

    @property
    def fra_day_count(self) -> DayCountMethod:
        dc = self.fra.day_count_method
        return dc if dc is not None else self.index.day_count_method
