"""The bundled USD product conventions, registered beside their index families on package import.

Each definition looks its family up in ``finance.conventions`` (registered there on import) and
attaches the swap, deposit and FRA tiers.  See ``finance.conventions.definitions`` for the
families and the calendar note.
"""

from __future__ import annotations

from finance.conventions import IndexFamily, default_registry, register_convention
from finance.dates import BDC, DayCountMethod, Frequency, Roll, Term
from finance.dates.term.term_type import TermType
from finance.instruments.conventions.market_conventions import (
    MATCH_TENOR,
    DepositConventions,
    FixedLegConventions,
    FloatLegConventions,
    FraConventions,
    MarketConventions,
    SwapConventions,
)
from finance.instruments.enums import CouponType

_SPOT_LAG = Term(2, TermType.BusinessDays)


def _usd_overnight_market(family: IndexFamily) -> MarketConventions:
    """USD overnight-index market template: compounded-in-arrears OIS, annual pay.

    Short-tenor (<=1Y) OIS keeps the annual frequency — the schedule algorithm produces the
    market-standard single period at maturity on its own.
    """
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
            spot_lag=_SPOT_LAG,
            spot_calendar="no_holidays",
        ),
        deposit=DepositConventions(spot_lag=_SPOT_LAG, business_day_convention=BDC.ModifiedFollowing, calendar="no_holidays"),
        fra=FraConventions(spot_lag=_SPOT_LAG, business_day_convention=BDC.ModifiedFollowing, calendar="no_holidays"),
    )


@register_convention("USD", "SOFR", overwrite=True)
def _usd_sofr() -> MarketConventions:
    """Standard USD SOFR OIS market (compounded-in-arrears, annual pay); key ``USD.SOFR.1D``."""
    return _usd_overnight_market(default_registry.family("USD.SOFR"))


@register_convention("USD", "FEDFUND", overwrite=True)
def _usd_fedfund() -> MarketConventions:
    """USD Fed Funds OIS market — same shape as SOFR (compounded, annual pay)."""
    return _usd_overnight_market(default_registry.family("USD.FEDFUND"))


@register_convention("USD", "UST-OTR", overwrite=True)
def _usd_ust_otr() -> MarketConventions:
    """The Treasury curve as an overnight index: swaps on it are SOFR-shaped (asset-swap par spreads)."""
    return _usd_overnight_market(default_registry.family("USD.UST-OTR"))


@register_convention("USD", "XXX-FTP", overwrite=True)
def _usd_ftp() -> MarketConventions:
    """Funds-transfer-pricing index (placeholder name): swaps on it are SOFR-shaped."""
    return _usd_overnight_market(default_registry.family("USD.XXX-FTP"))


@register_convention("USD", "CME-TERM-SOFR", overwrite=True)
def _usd_cme_term_sofr() -> MarketConventions:
    """CME Term SOFR swaps: semi-annual 30/360 fixed; the float leg pays and resets once per index tenor
    (``MATCH_TENOR``) with a single fixing per period."""
    return MarketConventions(
        family=default_registry.family("USD.CME-TERM-SOFR"),
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
            spot_lag=_SPOT_LAG,
            spot_calendar="no_holidays",
        ),
        deposit=DepositConventions(spot_lag=_SPOT_LAG, business_day_convention=BDC.ModifiedFollowing, calendar="no_holidays"),
        fra=FraConventions(spot_lag=_SPOT_LAG, business_day_convention=BDC.ModifiedFollowing, calendar="no_holidays"),
    )
