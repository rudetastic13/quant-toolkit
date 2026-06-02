"""PricingProgram — the compile-once / reprice-many primitive.

``PricingProgram`` wraps the columnar ``KernelInputs`` produced by the compiler: instruments
are *compiled* into a program once, then *run* (repriced) against any market.  A scenario or
curve shock is a single ``reprice(market)`` over the *same* compiled arrays — no instrument
rebuild.  This is the structural fix for the legacy approach that rebuilt the whole trade per
shocked curve.  The future Sensitivities layer bumps the market and calls ``reprice`` here.

It is intentionally NOT called "Portfolio" — it has no positions, P&L, or book identity;
that word is reserved for a future book abstraction that would sit on top of this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finance.pricing.kernels.compiler import reprice
from finance.pricing.kernels.inputs import KernelInputs
from finance.pricing.results import CashflowReport, PricingResult


@dataclass
class PricingProgram:
    """Immutable compiled form; reprice across markets without recompiling."""

    inputs: KernelInputs

    def reprice(self, market):
        """Raw kernel result (instrument/leg/flow PVs + realized rates)."""
        return reprice(self.inputs, market)

    def price(self, market, *, with_cashflows: bool = True) -> PricingResult:
        kr = self.reprice(market)
        cashflows = None
        if with_cashflows:
            ki = self.inputs
            leg_of_flow = np.repeat(
                np.arange(ki.n_legs), np.diff(np.append(ki.leg_offsets, ki.n_flows))
            )
            # df recovered from flow_pv = notional*rate*frac*df*sign (guard tiny cash)
            cash = ki.notional * kr.rate * ki.period_frac
            with np.errstate(divide="ignore", invalid="ignore"):
                df = np.where(cash * ki.sign != 0.0, kr.flow_pv / (cash * ki.sign), np.nan)
            cashflows = CashflowReport(
                pay_dates=ki.pay_dates, leg=leg_of_flow, notional=ki.notional,
                rate=kr.rate, period_frac=ki.period_frac, df=df, sign=ki.sign, flow_pv=kr.flow_pv,
            )
        return PricingResult(instrument_pv=kr.instrument_pv, leg_pv=kr.leg_pv, cashflows=cashflows)


__all__ = ["PricingProgram"]
