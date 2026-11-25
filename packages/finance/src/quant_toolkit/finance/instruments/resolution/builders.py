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

from quant_toolkit.common.object import serializable
from quant_toolkit.finance.dates import BDC, Date, DayCountMethod, Frequency, Term, add_term
from quant_toolkit.finance.dates.enums.fixing_type import FixingType
from quant_toolkit.finance.instruments.common_instrument import CommonInstrument
from quant_toolkit.finance.instruments.enums import CouponType, PrincipalExchange
from quant_toolkit.finance.instruments.priceable import Priceable
from quant_toolkit.finance.instruments.resolution.resolver import STDCSA, curve_name, resolve_effective, roll_spot
from quant_toolkit.finance.conventions import ConventionRegistry, default_registry


def _use(override, convention):
    """Explicit override wins; ``None`` means "take the convention"."""
    return override if override is not None else convention


@serializable("Swap")
@dataclass(kw_only=True)
class Swap(Priceable):
    """Two fully-specified legs (``CommonInstrument`` each).

    Pure data plus the ``Priceable`` functor: ``swap(market)`` prices standalone through
    the same compile→reprice machinery the portfolio path uses (compiled once, cached).
    ``currency``/``index_name`` let the pricer resolve the *projection* curve
    (``USD.SOFR.1D``); ``funding_id`` (default ``STDCSA``) names the *discount* curve
    (``USD.STDCSA``), so the two can differ for a real basis trade.

    Iterating a ``Swap`` yields its legs in pricing order (receive then pay) — the
    same order the pricer compiles them and ``KernelResult.leg_pv`` is laid out in;
    ``legs`` pairs each with its sign.
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

    @property
    def legs(self) -> tuple[tuple[CommonInstrument, float], ...]:
        return ((self.receive_leg, 1.0), (self.pay_leg, -1.0))

    @classmethod
    def fixed_float_swap(
        cls,
        *,
        notional: float,
        rate_index: str,
        fixed_rate: float,
        effective: Date | Term | None = None,
        maturity: Date | Term | None = None,
        as_of: Date | None = None,
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
        reset_frequency: Frequency | None = None,
        fixing_calendar: str | None = None,
        fixing_type: FixingType | None = None,
        float_coupon_type: CouponType | None = None,
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
        ``rate_index`` is the projection curve key (``"USD.SOFR.1D"``, ``"USD.CME-TERM-SOFR.3M"``);
        conventions come from that family's bundle and any keyword above overrides its
        convention.  For a leg structure the conventions can't express, construct the legs
        directly and call ``Swap(...)``.

        Dating: ``effective`` is a ``Date``, a ``Term`` offset from ``as_of`` (``Term.from_str("2BD")``,
        rolled Following on the spot calendar), or ``None`` for the convention's spot date.
        ``maturity`` is a ``Date``, a ``Term`` offset from the effective date, or ``None`` for
        ``tenor``.  ``as_of`` is only read when ``effective`` is not a ``Date`` and defaults to today.
        """
        key = curve_name(currency, rate_index)
        conv = registry.get(key)
        idx, swp = registry.index(key), conv.swap
        fixed_conv, float_conv = swp.fixed_leg, swp.float_leg

        effective = resolve_effective(effective, as_of, swp.spot_lag, swp.spot_calendar)
        if maturity is None:
            maturity = Term.from_str(tenor)
        if not isinstance(maturity, Date):
            maturity = effective + maturity
        if maturity <= effective:
            raise ValueError(f"maturity {maturity} is not after effective {effective}")

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
            payment_frequency=_use(float_frequency, conv.float_leg_payment_frequency(idx)),
            day_count_method=_use(float_day_count, conv.float_leg_day_count),
            business_day_convention=float_conv.business_day_convention,
            roll_convention=float_conv.roll_convention,
            pay_calendar=float_conv.pay_calendar,
            payment_delay=_use(payment_delay, float_conv.payment_delay),
            reset_frequency=_use(reset_frequency, conv.float_leg_reset_frequency(idx)),
            fixing_type=_use(fixing_type, idx.family.fixing_type),
            coupon_type=_use(float_coupon_type, float_conv.coupon_type),
            rate_index=key,
            rate_calendar=_use(fixing_calendar, idx.family.fixing_calendar),
            spread=spread, cap=cap, floor=floor, index_floor=index_floor,
            rate_lookback=_use(rate_lookback, float_conv.rate_lookback),
            rate_lockout=_use(rate_lockout, float_conv.rate_lockout),
        )

        # receive fixed: receive_leg = fixed; pay fixed (negative notional): receive_leg = float
        receive_leg, pay_leg = (fixed_leg, float_leg) if receive_fixed else (float_leg, fixed_leg)
        return cls(
            receive_leg=receive_leg, pay_leg=pay_leg,
            currency=currency, index_name=key, funding_id=funding_id,
        )


# ---------------------------------------------------------------------------
# Money-market instruments (deposits / FRAs): one-period legs.
# The constructor BDC-adjusts the maturity itself, so the legs carry their dates as
# contractual (``BDC.NoAdjustment``) and price through the same lowering as a swap.
# ---------------------------------------------------------------------------


@serializable("Deposit")
@dataclass(kw_only=True)
class Deposit(Priceable):
    """A cash deposit: lend ``notional`` over ``[effective, maturity]`` at simple ``rate``.

    Pure data.  ``index_name``/``currency`` resolve the forecast curve; ``funding_id`` the
    discount curve (a no-op for the single-curve case where they coincide).  As a contract it
    is one fixed leg with principal out on the effective date and back, plus interest, on
    maturity — so ``deposit(market).pv`` is zero exactly when ``rate`` is the curve's simple
    rate over the window.
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

    @property
    def legs(self) -> tuple[tuple[CommonInstrument, float], ...]:
        leg = CommonInstrument(
            effective=self.effective, maturity=self.maturity, currency=self.currency,
            notional=self.notional, payment_frequency=Frequency.Once,
            day_count_method=self.day_count_method, business_day_convention=BDC.NoAdjustment,
            coupon_type=CouponType.Fixed, coupon_rate=self.rate,
            principal_exchange=PrincipalExchange.Both,
        )
        return ((leg, 1.0),)

    @classmethod
    def spot_deposit(
        cls,
        *,
        rate: float,
        tenor: str | None = None,
        as_of: Date | None = None,
        effective: Date | Term | None = None,
        maturity: Date | Term | None = None,
        rate_index: str = "USD.SOFR.1D",
        currency: str = "USD",
        funding_id: str = STDCSA,
        notional: float = 1.0,
        day_count: DayCountMethod | None = None,
        registry: ConventionRegistry = default_registry,
    ) -> Deposit:
        """Build a cash deposit from minimal trader input (spot-starting unless ``effective`` says otherwise).

        ``effective`` is a ``Date``, a ``Term`` from ``as_of`` or ``None`` for spot; ``maturity`` is a
        ``Date``, a ``Term`` from the effective date (rolled on the deposit convention) or ``None``
        for ``tenor``.
        """
        key = curve_name(currency, rate_index)
        conv = registry.get(key)
        dep = conv.deposit
        effective = resolve_effective(effective, as_of, dep.spot_lag, dep.calendar)
        if maturity is None:
            if tenor is None:
                raise ValueError("a deposit needs a maturity or a tenor")
            maturity = Term.from_str(tenor)
        if not isinstance(maturity, Date):
            maturity = Date.from_numpy(
                add_term(effective.to_numpy(), maturity, dep.business_day_convention, dep.calendar)
            )
        return cls(
            effective=effective, maturity=maturity, rate=float(rate),
            day_count_method=_use(day_count, conv.deposit_day_count), currency=currency,
            index_name=key, funding_id=funding_id, notional=float(notional),
        )


@serializable("Fra")
@dataclass(kw_only=True)
class Fra(Priceable):
    """A forward rate agreement over ``[effective, maturity]``.

    Pure data: ``coupon_rate`` is the agreed forward rate, ``rate_index`` the projected
    index's key (``'USD.SOFR.1D'``), ``index_name``/``currency``/``funding_id`` resolve the curves.  As a contract
    it is a one-period fixed leg against a one-period floating leg settled at maturity; as
    for a swap, positive ``notional`` receives fixed, so ``fra(market).pv`` is zero exactly
    when ``coupon_rate`` is the curve's forward.
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

    @property
    def legs(self) -> tuple[tuple[CommonInstrument, float], ...]:
        common = dict(
            effective=self.effective, maturity=self.maturity, currency=self.currency,
            notional=abs(self.notional), payment_frequency=Frequency.Once,
            day_count_method=self.day_count_method, business_day_convention=BDC.NoAdjustment,
        )
        fixed = CommonInstrument(coupon_type=CouponType.Fixed, coupon_rate=self.coupon_rate, **common)
        floating = CommonInstrument(coupon_type=CouponType.Floating, rate_index=self.rate_index, **common)
        receive_fixed = self.notional >= 0
        return ((fixed, 1.0), (floating, -1.0)) if receive_fixed else ((floating, 1.0), (fixed, -1.0))

    @classmethod
    def forward_starting(
        cls,
        *,
        rate: float,
        start: str | None = None,
        end: str | None = None,
        as_of: Date | None = None,
        effective: Date | Term | None = None,
        maturity: Date | Term | None = None,
        rate_index: str = "USD.SOFR.1D",
        currency: str = "USD",
        funding_id: str = STDCSA,
        notional: float = 1.0,
        day_count: DayCountMethod | None = None,
        registry: ConventionRegistry = default_registry,
    ) -> Fra:
        """Build a forward rate agreement, e.g. a 3x6 is ``start='3M', end='6M'`` (both from spot).

        Alternatively date it directly: ``effective`` is a ``Date`` or a ``Term`` from ``as_of``,
        ``maturity`` a ``Date`` or a ``Term`` from the effective date; terms roll on the FRA convention.
        """
        key = curve_name(currency, rate_index)
        conv = registry.get(key)
        registry.index(key)  # the family must publish the tenor
        fra = conv.fra
        bdc, cal = fra.business_day_convention, fra.calendar
        if effective is None or maturity is None:
            if (effective is None and start is None) or (maturity is None and end is None):
                raise ValueError("an FRA needs effective/maturity, or start/end terms from spot")
            spot = roll_spot(as_of if as_of is not None else Date.today(), fra.spot_lag, fra.calendar)
        if effective is None:
            effective = Date.from_numpy(add_term(spot.to_numpy(), Term.from_str(start), bdc, cal))
        else:
            effective = resolve_effective(effective, as_of, fra.spot_lag, cal)
        if maturity is None:
            maturity = Date.from_numpy(add_term(spot.to_numpy(), Term.from_str(end), bdc, cal))
        elif not isinstance(maturity, Date):
            maturity = Date.from_numpy(add_term(effective.to_numpy(), maturity, bdc, cal))
        return cls(
            effective=effective, maturity=maturity, coupon_rate=float(rate),
            day_count_method=_use(day_count, conv.fra_day_count),
            rate_index=key,
            currency=currency, index_name=key, funding_id=funding_id,
            notional=float(notional),
        )

