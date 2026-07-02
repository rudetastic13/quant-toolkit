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

from finance.dates import Date, DayCountMethod, Term, add_term
from finance.instruments.common_instrument import CommonInstrument
from finance.instruments.enums import CouponType
from finance.instruments.priceable import Priceable
from finance.instruments.resolution.resolver import STDCSA, resolve_conventions, roll_spot
from finance.pricing.conventions import ConventionRegistry, default_registry


@dataclass
class ResolvedSwap(Priceable):
    """Two fully-specified legs (``CommonInstrument`` each).

    Pure data plus the ``Priceable`` functor: ``swap(market)`` prices standalone through
    the same compile→reprice machinery the portfolio path uses (compiled once, cached).
    ``currency``/``index_name`` let the pricer resolve the *projection* curve by name;
    ``funding_id`` (default ``STDCSA``) resolves the *discount* curve, so the two can
    differ for a real basis trade.

    Iterating a ``ResolvedSwap`` yields its legs in pricing order (receive then pay) — the
    same order the pricer compiles them and ``KernelResult.leg_pv`` is laid out in.
    """

    receive_leg: CommonInstrument
    pay_leg: CommonInstrument
    currency: str
    index_name: str
    funding_id: str = STDCSA

    def __iter__(self):
        """Yield legs in pricing/compile order (receive, pay)."""
        yield self.receive_leg
        yield self.pay_leg


def Swap(
    *,
    notional: float,
    rate_index: str,
    fixed_rate: float,
    tenor: str = "10Y",
    as_of: Date,
    currency: str = "USD",
    funding_id: str = STDCSA,
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
        fixing_type=conv.fixing_type,
        coupon_type=CouponType.GeometricAveraged,  # SOFR compounds in arrears
        rate_index=f"{currency} {rate_index}", spread=0.0, rate_calendar=conv.calendar,
        **overrides,
    )

    # receive fixed: receive_leg = fixed; pay fixed (negative notional): receive_leg = float
    receive_leg, pay_leg = (fixed_leg, float_leg) if receive_fixed else (float_leg, fixed_leg)
    return ResolvedSwap(
        receive_leg=receive_leg, pay_leg=pay_leg,
        currency=currency, index_name=rate_index, funding_id=funding_id,
    )


# ---------------------------------------------------------------------------
# Money-market instruments (deposits / FRAs) — short-end calibration helpers.
# Single-period contracts: there is no schedule builder downstream to adjust the
# end date, so the builder BDC-adjusts the maturity itself.
# ---------------------------------------------------------------------------


@dataclass
class ResolvedDeposit:
    """A cash deposit: lend ``notional`` over ``[effective, maturity]`` at simple ``rate``.

    Pure data.  ``index_name``/``currency`` resolve the forecast curve; ``funding_id`` the
    discount curve (a no-op for the single-curve case where they coincide).
    """

    effective: Date
    maturity: Date
    rate: float
    day_count_method: DayCountMethod
    currency: str
    index_name: str
    funding_id: str = STDCSA
    notional: float = 1.0


@dataclass
class ResolvedFra:
    """A forward rate agreement over ``[effective, maturity]``.

    Pure data: ``coupon_rate`` is the agreed forward rate, ``rate_index`` the projected
    index label, ``index_name``/``currency``/``funding_id`` resolve the curves.
    """

    effective: Date
    maturity: Date
    coupon_rate: float
    day_count_method: DayCountMethod
    rate_index: str
    currency: str
    index_name: str
    funding_id: str = STDCSA
    notional: float = 1.0


def Deposit(
    *,
    rate: float,
    tenor: str,
    as_of: Date,
    rate_index: str = "SOFR",
    currency: str = "USD",
    funding_id: str = STDCSA,
    notional: float = 1.0,
    registry: ConventionRegistry = default_registry,
) -> ResolvedDeposit:
    """Build a spot-starting cash deposit from minimal trader input."""
    conv = resolve_conventions(currency, rate_index, registry)
    effective = roll_spot(as_of, conv.spot_lag, conv.calendar)
    maturity = Date.from_numpy(
        add_term(effective.to_numpy(), Term.from_str(tenor), conv.business_day_convention, conv.calendar)
    )
    if maturity.to_numpy() <= effective.to_numpy():
        raise ValueError(f"deposit maturity {maturity} must be after effective {effective}")
    return ResolvedDeposit(
        effective=effective, maturity=maturity, rate=float(rate),
        day_count_method=conv.day_count_method, currency=currency,
        index_name=rate_index, funding_id=funding_id, notional=float(notional),
    )


def Fra(
    *,
    rate: float,
    start: str,
    end: str,
    as_of: Date,
    rate_index: str = "SOFR",
    currency: str = "USD",
    funding_id: str = STDCSA,
    notional: float = 1.0,
    registry: ConventionRegistry = default_registry,
) -> ResolvedFra:
    """Build a forward rate agreement, e.g. a 3x6 is ``start='3M', end='6M'``."""
    conv = resolve_conventions(currency, rate_index, registry)
    spot = roll_spot(as_of, conv.spot_lag, conv.calendar)
    bdc, cal = conv.business_day_convention, conv.calendar
    effective = Date.from_numpy(add_term(spot.to_numpy(), Term.from_str(start), bdc, cal))
    maturity = Date.from_numpy(add_term(spot.to_numpy(), Term.from_str(end), bdc, cal))
    if maturity.to_numpy() <= effective.to_numpy():
        raise ValueError(f"FRA end {end} must be after start {start}")
    return ResolvedFra(
        effective=effective, maturity=maturity, coupon_rate=float(rate),
        day_count_method=conv.day_count_method, rate_index=f"{currency} {rate_index}",
        currency=currency, index_name=rate_index, funding_id=funding_id, notional=float(notional),
    )


__all__ = ["ResolvedSwap", "Swap", "ResolvedDeposit", "Deposit", "ResolvedFra", "Fra"]
