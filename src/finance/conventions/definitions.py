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

from finance.conventions.convention_registry import register_convention
from finance.conventions.keys import OVERNIGHT
from finance.conventions.market_conventions import (
    MATCH_TENOR,
    DepositConventions,
    FixedLegConventions,
    FloatLegConventions,
    FraConventions,
    MarketConventions,
    SwapConventions,
)
from finance.conventions.rate_index import IndexFamily
from finance.dates import BDC, DayCountMethod, Frequency, Roll, Term
from finance.dates.enums import FixingType
from finance.dates.term.term_type import TermType
from finance.instruments.enums import CouponType


def _usd_overnight_market(index_name: str) -> MarketConventions:
    """USD overnight-index market template: compounded-in-arrears OIS, annual pay.

    Short-tenor (<=1Y) OIS keeps the annual frequency — the schedule algorithm produces
    the market-standard single period at maturity on its own.
    """
    family = IndexFamily(
        currency="USD",
        name=index_name,
        day_count_method=DayCountMethod.Actual360,
        fixing_calendar="no_holidays",
        publication_lag=Term(1, TermType.BusinessDays),
        tenors=(OVERNIGHT,),
    )
    spot_lag = Term(2, TermType.BusinessDays)
    return MarketConventions(
        family=family,
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
    """Standard USD SOFR OIS market (compounded-in-arrears, annual pay); key ``USD.SOFR.ON``."""
    return _usd_overnight_market("SOFR")


@register_convention("USD", "FEDFUND", overwrite=True)
def _usd_fedfund() -> MarketConventions:
    """USD Fed Funds OIS market — same shape as SOFR (compounded, annual pay).

    Fed Funds is the projection index of a SOFR-discounted basis trade: a Fed-Funds-vs-fixed
    swap projects off ``USD.FEDFUND.ON`` and discounts off the market's ``USD.STDCSA`` curve,
    so projection and funding are genuinely different curves.
    """
    return _usd_overnight_market("FEDFUND")


@register_convention("USD", "CME-TERM-SOFR", overwrite=True)
def _usd_cme_term_sofr() -> MarketConventions:
    """CME Term SOFR: a forward-looking term family, Act/360.

    One family, four tenors (``USD.CME-TERM-SOFR.1M`` ... ``.12M``).  The float leg pays and
    resets once per index tenor (``MATCH_TENOR``) with a single fixing per period.  The rate
    is *set* at the start of each period but it is the term rate *over* that period, so its
    projection window is the accrual period itself: ``FixingType.Arrears`` in this library's
    reset-window vocabulary (``Advance`` would read the window ending at the accrual start).
    """
    family = IndexFamily(
        currency="USD",
        name="CME-TERM-SOFR",
        day_count_method=DayCountMethod.Actual360,
        fixing_calendar="no_holidays",
        fixing_type=FixingType.Arrears,
        tenors=tuple(Term.from_str(t) for t in ("1M", "3M", "6M", "12M")),
    )
    spot_lag = Term(2, TermType.BusinessDays)
    return MarketConventions(
        family=family,
        swap=SwapConventions(
            fixed_leg=FixedLegConventions(
                payment_frequency=Frequency.SemiAnnually,
                day_count_method=DayCountMethod.Thirty360,
                business_day_convention=BDC.ModifiedFollowing,
                roll_convention=Roll.Empty,
                pay_calendar="no_holidays",
            ),
            float_leg=FloatLegConventions(
                payment_frequency=MATCH_TENOR,
                coupon_type=CouponType.Floating,  # one fixing per period, set in advance
                reset_frequency=MATCH_TENOR,
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
