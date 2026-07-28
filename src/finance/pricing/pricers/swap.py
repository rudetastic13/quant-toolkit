"""SwapPricer — turns Swaps into compiled columns and prices them.

Impure orchestration only: it builds each leg's schedule + coupon, assigns the pay/receive
sign, resolves curve names, and compiles.  The market is passed to ``price`` (never stored
on the instrument), so the same compiled portfolio reprices across scenarios cheaply.
"""
from __future__ import annotations

from common.registry import register_with
from finance.dates.term import TermType
from finance.instruments.common_instrument import CommonInstrument
from finance.instruments.enums import CouponType
from finance.instruments.priceable import pricer_registry
from finance.instruments.resolution import Swap, curve_name, funding_curve_name
from finance.instruments.schedules.coupon_schedule import (
    CouponSchedule,
    FixedCouponEvent,
    FloatingCouponEvent,
    AveragedCouponEvent,
    CompoundedCouponEvent,
)
from finance.instruments.schedules.payment_schedule import build_payment_schedule
from finance.pricing.kernels import LegSpec, StaticNotional, compile_portfolio
from finance.pricing.pricers.base import PricingProgram
from finance.pricing.results import PricingResult
from finance.pricing.types import Backend


def _bd(term) -> int:
    """Business-day count from a Term lag (0 if unset / not business-day denominated)."""
    if term is None or term.term_type != TermType.BusinessDays:
        return 0
    return int(term.term_length)


def _event(leg: CommonInstrument):
    ct = leg.coupon_type
    if ct in (CouponType.Fixed, CouponType.Zero):
        return FixedCouponEvent(start_date=leg.effective, coupon_rate=leg.coupon_rate)
    common = dict(
        start_date=leg.effective, rate_index=leg.rate_index or "undefined",
        spread=leg.spread or 0.0, index_floor=leg.index_floor, cap=leg.cap, floor=leg.floor,
    )
    if ct == CouponType.Floating:
        return FloatingCouponEvent(**common)
    if ct == CouponType.ArithmeticAveraged:
        return AveragedCouponEvent(**common)
    return CompoundedCouponEvent(**common)


@register_with(pricer_registry, "Swap", overwrite=True)
class SwapPricer:
    """Compiles Swaps to columnar inputs and prices them against a MarketContext.

    Registered in ``pricer_registry`` so ``Swap.__call__`` (the ``Priceable``
    functor) can dispatch here for standalone pricing.
    """

    default_backend = Backend.Numpy

    def compile(self, swaps: list[Swap], *, backend: Backend | None = None) -> PricingProgram:
        legs: list[LegSpec] = []
        for inst, swap in enumerate(swaps):
            proj = curve_name(swap.currency, swap.index_name)
            disc = funding_curve_name(swap.currency, swap.funding_id)
            legs.append(self._leg_spec(swap.receive_leg, +1.0, disc, proj, inst))
            legs.append(self._leg_spec(swap.pay_leg, -1.0, disc, proj, inst))
        return PricingProgram(
            inputs=compile_portfolio(legs),
            backend=self.default_backend if backend is None else Backend(backend),
        )

    def price(self, swaps: list[Swap], market, *, backend: Backend = Backend.Numpy) -> PricingResult:
        backend = Backend(backend)
        if backend not in (Backend.Numpy, Backend.Numba):
            raise NotImplementedError(f"backend {backend.name} is not implemented by SwapPricer")
        return self.compile(swaps, backend=backend).price(market)

    def _leg_spec(
        self, leg: CommonInstrument, sign: float, discount_curve: str, projection_curve: str, inst: int
    ) -> LegSpec:
        ps = build_payment_schedule(
            effective=leg.effective, maturity=leg.maturity, frequency=leg.payment_frequency,
            day_count_method=leg.day_count_method, bdc=leg.business_day_convention,
            calendar=leg.pay_calendar, roll=leg.roll_convention,
            payment_delay=leg.payment_delay,
            reset_frequency=leg.reset_frequency, fixing_type=leg.fixing_type,
            build_observations=leg.coupon_type.needs_observation_grid,
            observation_calendar=leg.rate_calendar,
            rate_lookback=_bd(leg.rate_lookback), rate_lockout=_bd(leg.rate_lockout),
        )
        coupon = CouponSchedule(events=[_event(leg)])
        return LegSpec(
            schedule=ps, coupon=coupon, sign=sign, discount_curve=discount_curve,
            projection_curve=projection_curve if leg.coupon_type.is_floating else None,
            notional=StaticNotional(leg.notional), instrument=inst,
        )


__all__ = ["SwapPricer"]
