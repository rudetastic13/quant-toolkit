"""PricingProgram — the compile-once / reprice-many primitive.

``PricingProgram`` wraps the columnar ``KernelInputs`` produced by the compiler: instruments
are *compiled* into a program once, then *run* (repriced) against any market.  A scenario or
curve shock is a single ``reprice(market)`` over the *same* compiled arrays — no instrument
rebuild.  This is the structural fix for the legacy approach that rebuilt the whole trade per
shocked curve.  The future Sensitivities layer bumps the market and calls ``reprice`` here.

It is intentionally NOT called "Portfolio" — it has no positions, P&L, or book identity;
that word is reserved for the book abstraction designed in docs/portfolio_and_scenarios.md,
which sits on top of this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finance.pricing.engines import program_class
from finance.pricing.kernels.inputs import KernelInputs
from finance.pricing.results import CashflowReport, PricingResult
from finance.pricing.types import Backend


@dataclass
class PricingProgram:
    """Compiled form with lazily cached backend preparation; reprice without rebuilding."""

    inputs: KernelInputs
    backend: Backend = Backend.Numpy
    _prepared: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def prepare(self, market, *, backend: Backend | None = None):
        """The engine program for ``backend``, prepared against ``market`` and cached per backend.

        Preparation is keyed to the market's curve geometry (origin, pillars, schemes): scenario and
        bump markets share it, and a market with a different geometry rebuilds it.  The engine is
        looked up in ``program_registry``, so a new backend needs no change here.
        """
        selected = self.backend if backend is None else Backend(backend)
        prepared = self._prepared.get(selected)
        if prepared is None or not prepared.compatible(market):
            prepared = program_class(selected)(self.inputs, market)
            self._prepared[selected] = prepared
        return prepared

    def reprice(self, market, *, backend: Backend | None = None):
        """Raw kernel result (instrument/leg/flow PVs + realized rates)."""
        return self.prepare(market, backend=backend).reprice(market)

    def price(
        self,
        market,
        *,
        with_cashflows: bool = True,
        backend: Backend | None = None,
    ) -> PricingResult:
        kr = self.reprice(market, backend=backend)
        cashflows = None
        if with_cashflows:
            ki = self.inputs
            leg_of_flow = np.repeat(
                np.arange(ki.n_legs), np.diff(np.append(ki.leg_offsets, ki.n_flows))
            )
            if kr.df is None:
                # Compatibility fallback for an engine that has not supplied payment DFs.
                cash = ki.notional * kr.rate * ki.period_frac
                with np.errstate(divide="ignore", invalid="ignore"):
                    df = np.where(cash * ki.sign != 0.0, kr.flow_pv / (cash * ki.sign), np.nan)
            else:
                df = kr.df
            cashflows = CashflowReport(
                pay_dates=ki.pay_dates, leg=leg_of_flow, notional=ki.notional,
                rate=kr.rate, period_frac=ki.period_frac, df=df, sign=ki.sign, flow_pv=kr.flow_pv,
            )
        return PricingResult(instrument_pv=kr.instrument_pv, leg_pv=kr.leg_pv, cashflows=cashflows)


__all__ = ["PricingProgram"]
