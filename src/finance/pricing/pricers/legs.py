"""LegPricer — the one lowering from leg-bearing instruments to a compiled program.

Any instrument that exposes ``legs`` (``(CommonInstrument, sign)`` pairs), ``currency``,
``index_name`` and ``funding_id`` prices here: swaps, deposits, FRAs, futures underlyings.
Impure orchestration only — build each leg's payment schedule, resolve curve names, pack
``LegSpec``s and compile.  The market is passed to ``price`` (never stored on the
instrument), so the same compiled program reprices across scenarios cheaply.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from finance.conventions import CurveKey
from finance.dates.term import TermType
from finance.instruments.common_instrument import CommonInstrument
from finance.instruments.priceable import pricer_registry
from finance.instruments.resolution.resolver import curve_name, funding_curve_name
from finance.instruments.schedules.payment_schedule import build_payment_schedule
from finance.pricing.kernels import LegSpec, StaticNotional, compile_portfolio
from finance.pricing.pricers.base import PricingProgram
from finance.pricing.results import PricingResult
from finance.pricing.types import Backend


@runtime_checkable
class LegInstrument(Protocol):
    """A contract made of signed legs plus the labels that resolve its curves."""

    currency: str
    index_name: str
    funding_id: str

    @property
    def legs(self) -> tuple[tuple[CommonInstrument, float], ...]: ...


def _business_days(term) -> int:
    """Business-day count from a Term lag (0 if unset / not business-day denominated)."""
    if term is None or term.term_type != TermType.BusinessDays:
        return 0
    return int(term.term_length)


def projection_curve_name(leg: CommonInstrument, default: str) -> str:
    """The leg's projection curve: its ``rate_index`` key (``'USD.SOFR.1D'``) when set, else ``default``.

    Lets a basis swap's legs project off different indices without any product-level field.

    Raises
    ------
    ValueError
        If the label is not a projection key (``CCY.INDEX.TENOR``).
    """
    label = leg.rate_index
    if not label:
        return default
    try:
        key = CurveKey.parse(label)
    except ValueError as exc:
        raise ValueError(f"leg rate_index {label!r} is not a curve key: {exc}") from None
    if key.is_discount:
        raise ValueError(f"leg rate_index {label!r} is a discount key; a floating leg projects off 'CCY.INDEX.TENOR'")
    return str(key)


def leg_spec(
    leg: CommonInstrument, sign: float, discount_curve: str, projection_curve: str, instrument: int
) -> LegSpec:
    """Lower one leg: its schedules are the only thing read."""
    schedule = build_payment_schedule(
        effective=leg.effective, maturity=leg.maturity, frequency=leg.payment_frequency,
        day_count_method=leg.day_count_method, bdc=leg.business_day_convention,
        calendar=leg.pay_calendar, roll=leg.roll_convention, direction=leg.direction,
        payment_delay=leg.payment_delay,
        first_regular=leg.first_regular_accrual_date, last_regular=leg.last_regular_accrual_date,
        reset_frequency=leg.reset_frequency, fixing_type=leg.fixing_type,
        build_observations=leg.needs_observation_grid,
        observation_calendar=leg.rate_calendar,
        rate_lookback=_business_days(leg.rate_lookback), rate_lockout=_business_days(leg.rate_lockout),
        lookback_style=leg.lookback_style,
    )
    return LegSpec(
        schedule=schedule, coupon=leg.coupon_schedule, sign=sign, discount_curve=discount_curve,
        projection_curve=projection_curve if leg.is_floating else None,
        notional=StaticNotional(leg.notional), instrument=instrument,
        principal_exchange=leg.principal_exchange,
    )


class LegPricer:
    """Compiles leg-bearing instruments to columnar inputs and prices them against a MarketContext."""

    default_backend = Backend.Numpy

    def compile(self, instruments: Sequence[LegInstrument], *, backend: Backend | None = None) -> PricingProgram:
        legs: list[LegSpec] = []
        for index, instrument in enumerate(instruments):
            discount = funding_curve_name(instrument.currency, instrument.funding_id)
            projection = curve_name(instrument.currency, instrument.index_name)
            for leg, sign in instrument.legs:
                legs.append(leg_spec(leg, sign, discount, projection_curve_name(leg, projection), index))
        return PricingProgram(
            inputs=compile_portfolio(legs),
            backend=self.default_backend if backend is None else Backend(backend),
        )

    def price(
        self, instruments: Sequence[LegInstrument], market, *, backend: Backend | None = None
    ) -> PricingResult:
        selected = self.default_backend if backend is None else Backend(backend)
        if selected not in (Backend.Numpy, Backend.Numba):
            raise NotImplementedError(f"backend {selected.name} is not implemented by {type(self).__name__}")
        return self.compile(instruments, backend=selected).price(market)


# ``Priceable`` dispatch: every leg-bearing contract type prices through the same lowering.
for _name in ("Swap", "Deposit", "Fra"):
    pricer_registry.register(_name, LegPricer, overwrite=True)

