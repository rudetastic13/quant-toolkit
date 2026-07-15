"""Static market-convention definitions, registered into the shared ConventionRegistry.

Importing this module has the side effect of registering the bundled conventions
(via the ``@register_convention`` decorator).  ``conventions/__init__.py`` imports it
so the definitions are available as soon as the package is loaded.

Calendar note
-------------
Only the ``no_holidays`` calendar is registered today (see
``finance.dates.calendars``).  SOFR really settles on SIFMA; until a SIFMA calendar
is registered we use ``no_holidays`` (weekday mask, no holiday set).  Swap the
calendar fields once a SIFMA calendar exists.
"""
from __future__ import annotations

from finance.dates import Term, DayCountMethod, BDC, Roll, Frequency
from finance.dates.term import TermType
from finance.instruments.enums import CouponType

from finance.conventions.market_conventions import (
    DepositConventions,
    FixedLegConventions,
    FloatLegConventions,
    FraConventions,
    MarketConventions,
    SwapConventions,
)
from finance.conventions.convention_registry import register_convention
from finance.conventions.rate_index import RateIndex


def _usd_overnight_market(index_name: str) -> MarketConventions:
    """USD overnight-index market template: compounded-in-arrears OIS, annual pay.

    Short-tenor (<=1Y) OIS keeps the annual frequency — the schedule algorithm produces
    the market-standard single period at maturity on its own.
    """
    index = RateIndex(
        currency="USD",
        name=index_name,
        day_count_method=DayCountMethod.Actual360,
        fixing_calendar="no_holidays",
        publication_lag=Term(1, TermType.BusinessDays),
    )
    spot_lag = Term(2, TermType.BusinessDays)
    return MarketConventions(
        index=index,
        swap=SwapConventions(
            fixed_leg=FixedLegConventions(
                payment_frequency=Frequency.Annually,
                day_count_method=DayCountMethod.Actual360,
                business_day_convention=BDC.ModifiedFollowing,
                roll_convention=Roll.Empty,
                pay_calendar="no_holidays",
            ),
            float_leg=FloatLegConventions(
                payment_frequency=Frequency.Annually,
                coupon_type=CouponType.GeometricAveraged,  # RFR compounds in arrears
                reset_frequency=Frequency.Daily,
                business_day_convention=BDC.ModifiedFollowing,
                roll_convention=Roll.Empty,
                pay_calendar="no_holidays",
            ),
            spot_lag=spot_lag,
            spot_calendar="no_holidays",
        ),
        deposit=DepositConventions(
            spot_lag=spot_lag,
            business_day_convention=BDC.ModifiedFollowing,
            calendar="no_holidays",
        ),
        fra=FraConventions(
            spot_lag=spot_lag,
            business_day_convention=BDC.ModifiedFollowing,
            calendar="no_holidays",
        ),
    )


@register_convention("USD", "SOFR", overwrite=True)
def _usd_sofr() -> MarketConventions:
    """Standard USD SOFR OIS market (compounded-in-arrears, annual pay)."""
    return _usd_overnight_market("SOFR")


@register_convention("USD", "FEDFUND", overwrite=True)
def _usd_fedfund() -> MarketConventions:
    """USD Fed Funds OIS market — same shape as SOFR (compounded, annual pay).

    Fed Funds is the projection index of a SOFR-discounted basis trade: a Fed-Funds-vs-fixed
    swap projects off ``USD.FEDFUND`` and discounts off ``USD.SOFR`` (``funding_id='SOFR'``),
    so projection and funding are genuinely different curves.
    """
    return _usd_overnight_market("FEDFUND")
