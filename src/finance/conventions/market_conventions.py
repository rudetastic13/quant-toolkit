"""Market conventions — per-leg and per-product convention descriptors.

Three tiers, composed bottom-up:

- :class:`IndexFamily` (``rate_index.py``) — the index itself: projection day count,
  fixing calendar, fixing style, published tenors.
- Leg conventions — :class:`FixedLegConventions` / :class:`FloatLegConventions`: the
  standard shape of one leg.  Field names mirror ``CommonInstrument`` so applying a leg
  convention to a leg is a straight field-for-field mapping.
- Product conventions — :class:`SwapConventions`, :class:`DepositConventions`,
  :class:`FraConventions`: leg conventions plus product-level dating (spot lag).

One :class:`MarketConventions` bundle per index family is what the ``ConventionRegistry``
stores: builders look up the bundle once and read their product's section.  ``None`` on a
leg/product day count means "use the index's"; :data:`MATCH_TENOR` on a float-leg frequency
means "one period per index tenor" (a 3M term index pays and resets quarterly).  The
resolved accessors on the bundle apply that defaulting for one :class:`RateIndex` (its tenor).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from finance.conventions.rate_index import IndexFamily, RateIndex
from finance.dates import Term
from finance.dates.enums import BDC, DayCountMethod, Frequency, Roll
from finance.dates.term.term_type import TermType
from finance.instruments.enums import CouponType


class MatchTenor(Enum):
    """Sentinel for a float-leg frequency that follows the index tenor."""

    MATCH_TENOR = "match_tenor"


MATCH_TENOR = MatchTenor.MATCH_TENOR

_FREQUENCY_OF_TENOR: dict[tuple[int, TermType], Frequency] = {
    (1, TermType.Days): Frequency.Daily,
    (1, TermType.Weeks): Frequency.Weekly,
    (2, TermType.Weeks): Frequency.BiWeekly,
    (1, TermType.Months): Frequency.Monthly,
    (2, TermType.Months): Frequency.BiMonthly,
    (3, TermType.Months): Frequency.Quarterly,
    (1, TermType.Quarters): Frequency.Quarterly,
    (6, TermType.Months): Frequency.SemiAnnually,
    (12, TermType.Months): Frequency.Annually,
    (1, TermType.Years): Frequency.Annually,
}


def frequency_for_tenor(tenor: Term) -> Frequency:
    """The payment/reset frequency of a leg with one period per ``tenor``.

    Raises
    ------
    ValueError
        If no ``Frequency`` member matches the tenor.
    """
    try:
        return _FREQUENCY_OF_TENOR[(int(tenor.term_length), tenor.term_type)]
    except KeyError:
        raise ValueError(f"no Frequency matches the index tenor {tenor!r}") from None


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
    between a compounded-in-arrears RFR leg (``GeometricAveraged``), an arithmetic-average
    leg and a single-fixing term-index leg (``Floating``).  ``day_count_method=None`` means
    "use the index's" — the accrual basis then matches the projection basis by construction.
    ``payment_frequency`` / ``reset_frequency`` may be :data:`MATCH_TENOR`: one period per
    index tenor, resolved per :class:`RateIndex` by the bundle.
    """

    payment_frequency: Frequency | MatchTenor
    coupon_type: CouponType
    reset_frequency: Frequency | MatchTenor
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
    """Everything the builders need for one index family: the registry's unit.

    The resolved accessors apply the ``None``-means-index and ``MATCH_TENOR`` defaulting so
    builders never re-implement it.
    """

    family: IndexFamily
    swap: SwapConventions
    deposit: DepositConventions
    fra: FraConventions

    def index(self, tenor: Term | str | None = None) -> RateIndex:
        """The family at ``tenor`` (omit it for an overnight family)."""
        return self.family.index(tenor)

    @property
    def float_leg_day_count(self) -> DayCountMethod:
        dc = self.swap.float_leg.day_count_method
        return dc if dc is not None else self.family.day_count_method

    @property
    def deposit_day_count(self) -> DayCountMethod:
        dc = self.deposit.day_count_method
        return dc if dc is not None else self.family.day_count_method

    @property
    def fra_day_count(self) -> DayCountMethod:
        dc = self.fra.day_count_method
        return dc if dc is not None else self.family.day_count_method

    def float_leg_payment_frequency(self, index: RateIndex) -> Frequency:
        freq = self.swap.float_leg.payment_frequency
        return frequency_for_tenor(index.tenor) if freq is MATCH_TENOR else freq

    def float_leg_reset_frequency(self, index: RateIndex) -> Frequency:
        freq = self.swap.float_leg.reset_frequency
        return frequency_for_tenor(index.tenor) if freq is MATCH_TENOR else freq
