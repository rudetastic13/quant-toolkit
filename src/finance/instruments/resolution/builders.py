"""Trader-facing builders — minimal input, conventions do the work.

``Swap(notional=100, rate_index="SOFR", fixed_rate=0.045, tenor="10Y", as_of=...)`` means a
100mm SOFR swap, **receive 4.5% fixed** vs pay compounded SOFR, settled spot, on standard
SOFR conventions.  Sign rides on the notional: **negative notional = pay fixed** (no
``direction`` argument).  ``**overrides`` is the escape hatch for a non-standard trade — it
is never required.

The result is a pure-data ``ResolvedSwap`` (two ``CommonInstrument`` legs).  It carries no
curve and no pricing methods — that separation is the whole point (cf. the legacy
``fixedfloatswap`` that fused contract + market + analytics and paid for it in scenario
rebuilds).
"""
from __future__ import annotations

from dataclasses import dataclass

from finance.dates import Date, Term
from finance.instruments.common_instrument import CommonInstrument
from finance.instruments.enums import CouponType
from finance.instruments.resolution.resolver import resolve_conventions, roll_spot
from finance.pricing.conventions import ConventionRegistry, default_registry


@dataclass
class ResolvedSwap:
    """Two fully-specified legs. ``receive_leg``/``pay_leg`` satisfy the Swap protocol.

    Pure data — no curve, no methods that price. ``currency``/``index_name`` let the pricer
    resolve the discount & projection curve by name.
    """

    receive_leg: CommonInstrument
    pay_leg: CommonInstrument
    currency: str
    index_name: str


def Swap(
    *,
    notional: float,
    rate_index: str,
    fixed_rate: float,
    tenor: str = "10Y",
    as_of: Date,
    currency: str = "USD",
    registry: ConventionRegistry = default_registry,
    **overrides,
) -> ResolvedSwap:
    """Build a vanilla fixed-vs-compounded(-OIS) swap from minimal trader input."""
    conv = resolve_conventions(currency, rate_index, registry)
    effective = roll_spot(as_of, conv.spot_lag, conv.calendar)
    maturity = effective + Term.from_str(tenor)

    receive_fixed = notional >= 0
    notl = abs(float(notional))
    fixed_dc = conv.fixed_day_count_method or conv.day_count_method

    fixed_leg = CommonInstrument(
        effective=effective, maturity=maturity, currency=currency, notional=notl,
        payment_frequency=conv.payment_frequency, day_count_method=fixed_dc,
        business_day_convention=conv.business_day_convention, roll_convention=conv.roll_convention,
        pay_calendar=conv.calendar, coupon_type=CouponType.Fixed, coupon_rate=fixed_rate,
        **overrides,
    )
    float_leg = CommonInstrument(
        effective=effective, maturity=maturity, currency=currency, notional=notl,
        payment_frequency=conv.payment_frequency, day_count_method=conv.day_count_method,
        business_day_convention=conv.business_day_convention, roll_convention=conv.roll_convention,
        pay_calendar=conv.calendar, reset_frequency=conv.reset_frequency,
        coupon_type=CouponType.GeometricAveraged,  # SOFR compounds in arrears
        rate_index=f"{currency} {rate_index}", spread=0.0, rate_calendar=conv.calendar,
        **overrides,
    )

    # receive fixed: receive_leg = fixed; pay fixed (negative notional): receive_leg = float
    receive_leg, pay_leg = (fixed_leg, float_leg) if receive_fixed else (float_leg, fixed_leg)
    return ResolvedSwap(receive_leg=receive_leg, pay_leg=pay_leg, currency=currency, index_name=rate_index)


__all__ = ["ResolvedSwap", "Swap"]
