"""Trader-facing instrument dataclasses with convention-resolving constructors.

Two tiers, both self-documenting:

- **Convention path** — classmethods like :meth:`Swap.fixed_float_swap`: minimal trader
  input, the :class:`MarketConventions` bundle fills the rest.  Every overridable field is
  an explicit keyword argument (``None`` = "take the convention"), routed to the leg it
  belongs to — no ``**kwargs`` guessing.
- **Direct construction** — build the ``CommonInstrument`` legs yourself and construct the
  dataclass; the full surface is articulated by the dataclass definitions.

The dataclasses are pure data (no curve, no market state) and validate on construction:
contracts round-trip through serialization without touching the convention registry.
"""
from __future__ import annotations

from dataclasses import dataclass

from finance.dates import Date, DayCountMethod, Frequency, Term, add_term
from finance.instruments.common_instrument import CommonInstrument
from finance.instruments.enums import CouponType
from finance.instruments.priceable import Priceable
from finance.instruments.resolution.resolver import STDCSA, resolve_conventions, roll_spot
from finance.conventions import ConventionRegistry, default_registry


def _use(override, convention):
    """Explicit override wins; ``None`` means "take the convention"."""
    return override if override is not None else convention


@dataclass(kw_only=True)
class Swap(Priceable):
    """Two fully-specified legs (``CommonInstrument`` each).

    Pure data plus the ``Priceable`` functor: ``swap(market)`` prices standalone through
    the same compile→reprice machinery the portfolio path uses (compiled once, cached).
    ``currency``/``index_name`` let the pricer resolve the *projection* curve by name;
    ``funding_id`` (default ``STDCSA``) resolves the *discount* curve, so the two can
    differ for a real basis trade.

    Iterating a ``Swap`` yields its legs in pricing order (receive then pay) — the
    same order the pricer compiles them and ``KernelResult.leg_pv`` is laid out in.
    """

    receive_leg: CommonInstrument
    pay_leg: CommonInstrument
    currency: str
    index_name: str
    funding_id: str = STDCSA

    def __post_init__(self):
        for name, leg in (("receive_leg", self.receive_leg), ("pay_leg", self.pay_leg)):
            if leg.currency != self.currency:
                raise ValueError(
                    f"{name} currency ({leg.currency}) does not match swap currency ({self.currency})"
                )

    def __iter__(self):
        """Yield legs in pricing/compile order (receive, pay)."""
        yield self.receive_leg
        yield self.pay_leg

    @classmethod
    def fixed_float_swap(
        cls,
        *,
        notional: float,
        rate_index: str,
        fixed_rate: float,
        as_of: Date,
        tenor: str = "10Y",
        currency: str = "USD",
        funding_id: str = STDCSA,
        # trade-level overrides — None means "take the convention"
        fixed_frequency: Frequency | None = None,
        float_frequency: Frequency | None = None,
        fixed_day_count: DayCountMethod | None = None,
        float_day_count: DayCountMethod | None = None,
        payment_delay: Term | None = None,
        # float-leg shaping — routed to the float leg only
        spread: float = 0.0,
        cap: float | None = None,
        floor: float | None = None,
        index_floor: float | None = None,
        rate_lookback: Term | None = None,
        rate_lockout: Term | None = None,
        registry: ConventionRegistry = default_registry,
    ) -> Swap:
        """Build a standard fixed-vs-float swap from minimal trader input.

        Sign rides on the notional — positive = receive fixed, negative = pay fixed.
        Conventions come from the ``(currency, rate_index)`` bundle; any keyword above
        overrides its convention.  For a leg structure the conventions can't express,
        construct the legs directly and call ``Swap(...)``.
        """
        conv = resolve_conventions(currency, rate_index, registry)
        idx, swp = conv.index, conv.swap
        fixed_conv, float_conv = swp.fixed_leg, swp.float_leg

        effective = roll_spot(as_of, swp.spot_lag, swp.spot_calendar)
        maturity = effective + Term.from_str(tenor)

        receive_fixed = notional >= 0
        notl = abs(float(notional))

        fixed_leg = CommonInstrument(
            effective=effective, maturity=maturity, currency=currency, notional=notl,
            payment_frequency=_use(fixed_frequency, fixed_conv.payment_frequency),
            day_count_method=_use(fixed_day_count, fixed_conv.day_count_method),
            business_day_convention=fixed_conv.business_day_convention,
            roll_convention=fixed_conv.roll_convention,
            pay_calendar=fixed_conv.pay_calendar,
            payment_delay=_use(payment_delay, fixed_conv.payment_delay),
            coupon_type=CouponType.Fixed, coupon_rate=fixed_rate,
        )
        float_leg = CommonInstrument(
            effective=effective, maturity=maturity, currency=currency, notional=notl,
            payment_frequency=_use(float_frequency, float_conv.payment_frequency),
            day_count_method=_use(float_day_count, conv.float_leg_day_count),
            business_day_convention=float_conv.business_day_convention,
            roll_convention=float_conv.roll_convention,
            pay_calendar=float_conv.pay_calendar,
            payment_delay=_use(payment_delay, float_conv.payment_delay),
            reset_frequency=float_conv.reset_frequency,
            fixing_type=idx.fixing_type,
            coupon_type=float_conv.coupon_type,
            rate_index=idx.label,
            rate_calendar=idx.fixing_calendar,
            spread=spread, cap=cap, floor=floor, index_floor=index_floor,
            rate_lookback=_use(rate_lookback, float_conv.rate_lookback),
            rate_lockout=_use(rate_lockout, float_conv.rate_lockout),
        )

        # receive fixed: receive_leg = fixed; pay fixed (negative notional): receive_leg = float
        receive_leg, pay_leg = (fixed_leg, float_leg) if receive_fixed else (float_leg, fixed_leg)
        return cls(
            receive_leg=receive_leg, pay_leg=pay_leg,
            currency=currency, index_name=rate_index, funding_id=funding_id,
        )


# ---------------------------------------------------------------------------
# Money-market instruments (deposits / FRAs) — short-end calibration helpers.
# Single-period contracts: there is no schedule builder downstream to adjust the
# end date, so the constructor BDC-adjusts the maturity itself.
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class Deposit:
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

    def __post_init__(self):
        if self.maturity <= self.effective:
            raise ValueError(
                f"deposit maturity {self.maturity} must be after effective {self.effective}"
            )

    @classmethod
    def spot_deposit(
        cls,
        *,
        rate: float,
        tenor: str,
        as_of: Date,
        rate_index: str = "SOFR",
        currency: str = "USD",
        funding_id: str = STDCSA,
        notional: float = 1.0,
        day_count: DayCountMethod | None = None,
        registry: ConventionRegistry = default_registry,
    ) -> Deposit:
        """Build a spot-starting cash deposit from minimal trader input."""
        conv = resolve_conventions(currency, rate_index, registry)
        dep = conv.deposit
        effective = roll_spot(as_of, dep.spot_lag, dep.calendar)
        maturity = Date.from_numpy(
            add_term(effective.to_numpy(), Term.from_str(tenor), dep.business_day_convention, dep.calendar)
        )
        return cls(
            effective=effective, maturity=maturity, rate=float(rate),
            day_count_method=_use(day_count, conv.deposit_day_count), currency=currency,
            index_name=rate_index, funding_id=funding_id, notional=float(notional),
        )


@dataclass(kw_only=True)
class Fra:
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

    def __post_init__(self):
        if self.maturity <= self.effective:
            raise ValueError(
                f"FRA maturity {self.maturity} must be after effective {self.effective}"
            )

    @classmethod
    def forward_starting(
        cls,
        *,
        rate: float,
        start: str,
        end: str,
        as_of: Date,
        rate_index: str = "SOFR",
        currency: str = "USD",
        funding_id: str = STDCSA,
        notional: float = 1.0,
        day_count: DayCountMethod | None = None,
        registry: ConventionRegistry = default_registry,
    ) -> Fra:
        """Build a forward rate agreement, e.g. a 3x6 is ``start='3M', end='6M'``."""
        conv = resolve_conventions(currency, rate_index, registry)
        fra = conv.fra
        spot = roll_spot(as_of, fra.spot_lag, fra.calendar)
        bdc, cal = fra.business_day_convention, fra.calendar
        effective = Date.from_numpy(add_term(spot.to_numpy(), Term.from_str(start), bdc, cal))
        maturity = Date.from_numpy(add_term(spot.to_numpy(), Term.from_str(end), bdc, cal))
        return cls(
            effective=effective, maturity=maturity, coupon_rate=float(rate),
            day_count_method=_use(day_count, conv.fra_day_count),
            rate_index=conv.index.label,
            currency=currency, index_name=rate_index, funding_id=funding_id,
            notional=float(notional),
        )


__all__ = ["Swap", "Deposit", "Fra"]
