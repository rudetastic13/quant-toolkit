"""SOFR futures pricing through the shared compounded/averaged rate engine."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finance.dates import period_fractions
from finance.instruments.resolution.futures import SofrFuture
from finance.instruments.resolution.resolver import curve_name, funding_curve_name
from finance.instruments.schedules.coupon_schedule import AveragedCouponEvent, CompoundedCouponEvent, CouponSchedule
from finance.instruments.schedules.observation import build_observation_grid
from finance.instruments.schedules.payment_schedule import PaymentSchedule
from finance.pricing.kernels import LegSpec, StaticNotional, compile_portfolio
from finance.pricing.pricers.base import PricingProgram
from finance.pricing.types import Backend, RateKind


@dataclass(frozen=True)
class FuturesPricingResult:
    """Model rate/price and variation-margin P&L for a futures strip."""

    model_rate: np.ndarray
    model_price: np.ndarray
    pnl: np.ndarray


@dataclass(frozen=True)
class FuturesRiskResult:
    """Per-contract sensitivities to non-origin zero-rate pillars."""

    rate_gradient: np.ndarray
    price_gradient: np.ndarray
    pnl_gradient: np.ndarray


def _future_schedule(future: SofrFuture) -> PaymentSchedule:
    starts = np.array([future.ref_start.to_numpy()], dtype="datetime64[D]")
    ends = np.array([future.ref_end.to_numpy()], dtype="datetime64[D]")
    grid = build_observation_grid(
        accrual_starts=starts,
        accrual_ends=ends,
        calendar="no_holidays",
        day_count=future.day_count_method,
    )
    return PaymentSchedule(
        accrual_starts=starts,
        accrual_ends=ends,
        payment_dates=ends.copy(),
        period_fracs=period_fractions(future.day_count_method, starts, ends),
        reset_starts=None,
        reset_ends=None,
        obs_read_starts=grid.read_starts,
        obs_read_ends=grid.read_ends,
        obs_weights=grid.weights,
        obs_offsets=grid.offsets,
    )


class FuturesProgram:
    """Compiled strip whose one flow per instrument yields its settlement rate."""

    def __init__(self, futures: list[SofrFuture], backend: Backend):
        if not futures:
            raise ValueError("FuturesPricer requires at least one future")
        self.futures = tuple(futures)
        legs = []
        for inst, future in enumerate(futures):
            event_type = CompoundedCouponEvent if future.rate_kind == RateKind.Compounded else AveragedCouponEvent
            event = event_type(start_date=future.ref_start)
            legs.append(
                LegSpec(
                    schedule=_future_schedule(future),
                    coupon=CouponSchedule(events=[event]),
                    sign=1.0,
                    discount_curve=funding_curve_name(future.currency, future.funding_id),
                    projection_curve=curve_name(future.currency, future.index_name),
                    notional=StaticNotional(1.0),
                    instrument=inst,
                )
            )
        self.program = PricingProgram(compile_portfolio(legs), backend=backend)

    def reprice(self, market) -> FuturesPricingResult:
        raw = self.program.reprice(market)
        rate = raw.rate + np.array([future.convexity for future in self.futures])
        price = 100.0 * (1.0 - rate)
        pnl = np.full(rate.shape, np.nan, dtype=np.float64)
        for i, future in enumerate(self.futures):
            if future.entry_price is not None:
                pnl[i] = (
                    (price[i] - future.entry_price)
                    / 0.01
                    * future.point_value
                    * future.contracts
                )
        return FuturesPricingResult(model_rate=rate, model_price=price, pnl=pnl)

    def risk(self, market, curve: str) -> FuturesRiskResult:
        """Analytic rate, price, and P&L gradients from the Numba projection adjoint.

        Always runs on the Numba adjoint regardless of the reprice backend (numpy has no
        analytic adjoint).
        """
        numba_program = self.program.prepare(market, backend=Backend.Numba)
        adjoint = numba_program.value_and_grad(market)
        flow_grad = adjoint.cashflow_gradient(curve, role="index")
        ki = self.program.inputs
        discount = np.empty(ki.n_flows, dtype=np.float64)
        for ci, name in enumerate(ki.curve_names):
            mask = ki.discount_curve == ci
            discount[mask] = market.yield_curve(name).discount_factor(ki.pay_dates[mask])
        rate_grad = flow_grad / (ki.period_frac * discount)[:, None]
        price_grad = -100.0 * rate_grad
        pnl_grad = np.empty_like(price_grad)
        for i, future in enumerate(self.futures):
            pnl_grad[i] = price_grad[i] / 0.01 * future.point_value * future.contracts
        return FuturesRiskResult(rate_grad, price_grad, pnl_grad)


class FuturesPricer:
    # Numpy is the always-available reference (numba is optional and LogLinearDF-only);
    # pass Backend.Numba for the fused hot path.  risk() uses the Numba adjoint either way.
    default_backend = Backend.Numpy

    def compile(self, futures: list[SofrFuture], *, backend: Backend | None = None) -> FuturesProgram:
        selected = self.default_backend if backend is None else Backend(backend)
        if selected not in (Backend.Numpy, Backend.Numba):
            raise NotImplementedError(f"backend {selected.name} is not implemented by FuturesPricer")
        return FuturesProgram(futures, selected)

    def price(
        self,
        futures: list[SofrFuture],
        market,
        *,
        backend: Backend | None = None,
    ) -> FuturesPricingResult:
        return self.compile(futures, backend=backend).reprice(market)


__all__ = ["FuturesPricer", "FuturesProgram", "FuturesPricingResult", "FuturesRiskResult"]
