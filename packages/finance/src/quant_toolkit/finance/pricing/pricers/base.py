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
from typing import ClassVar, Sequence

import numpy as np

from quant_toolkit.finance.pricing.engines import program_class
from quant_toolkit.finance.pricing.kernels.inputs import KernelInputs
from quant_toolkit.finance.pricing.pricers.protocol import (
    PillarGradient,
    declare_measures,
    measure,
    on_demand,
    resolve_requests,
)
from quant_toolkit.finance.pricing.results import CashflowReport, PricingResult
from quant_toolkit.finance.pricing.types import Backend


@dataclass
class PricingProgram:
    """Compiled form with lazily cached backend preparation; reprice without rebuilding."""

    inputs: KernelInputs
    backend: Backend = Backend.Numpy
    _prepared: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    measures: ClassVar[frozenset[str]] = declare_measures(PricingResult.fields)

    @property
    def n_positions(self) -> int:
        return int(self.inputs.n_instruments)

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
        requests: Sequence[str] | None = None,
        *,
        backend: Backend | None = None,
    ) -> PricingResult:
        """PV per instrument and per leg, with the cashflow table when asked for.

        Parameters
        ----------
        market : MarketContext
        requests : sequence of str, optional
            ``None`` is ``pv`` + ``leg_pvs`` + ``cashflows``.  ``pv`` and ``leg_pvs`` come from the
            one kernel pass and are always populated; ``cashflows`` is assembled only when
            requested.  ``zero_ladder``, ``index_ladder`` and ``funding_ladder`` (a
            :class:`PillarRisk` each: total, projection-only, discounting-only) run the Numba
            adjoint and are computed only when named.  A measure only another product answers
            (``"vega"``) is skipped and reads ``None`` from ``result.get``.
        backend : Backend, optional

        Raises
        ------
        ValueError
            If a request is a name no program answers.
        """
        asked = resolve_requests(requests, self.measures, PricingResult.fields)
        kr = self.reprice(market, backend=backend)
        cashflows = None
        if "cashflows" in asked:
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
        return PricingResult(
            instrument_pv=kr.instrument_pv, leg_pv=kr.leg_pv, cashflows=cashflows,
            extras=on_demand(self, market, asked),
        )

    def pillar_gradient(self, market) -> PillarGradient:
        """Total (projection + discounting) ``d PV / d z`` from the Numba adjoint."""
        numba = self.prepare(market, backend=Backend.Numba)
        result = numba.value_and_grad(market)
        return PillarGradient(numba.layout, result.instrument_total_gradient, result.primal.instrument_pv)


def _adjoint(program: PricingProgram, market, cache: dict):
    """The Numba adjoint result, run once per ``price`` call."""
    if "adjoint" not in cache:
        cache["adjoint"] = program.prepare(market, backend=Backend.Numba).value_and_grad(market)
    return cache["adjoint"]


def _ladder(program: PricingProgram, market, cache: dict, attribute: str):
    from quant_toolkit.finance.pricing.risk.pillar import PillarRisk

    result = _adjoint(program, market, cache)
    return PillarRisk.from_layout(result.layout, getattr(result, attribute), result.primal.instrument_pv)


@measure(PricingProgram, "zero_ladder")
def _zero_ladder(program, market, cache):
    """Total ``d PV / d z`` per instrument (projection + discounting)."""
    return _ladder(program, market, cache, "instrument_total_gradient")


@measure(PricingProgram, "index_ladder")
def _index_ladder(program, market, cache):
    """``d PV / d z`` through projection only."""
    return _ladder(program, market, cache, "instrument_index_gradient")


@measure(PricingProgram, "funding_ladder")
def _funding_ladder(program, market, cache):
    """``d PV / d z`` through discounting only."""
    return _ladder(program, market, cache, "instrument_funding_gradient")


__all__ = ["PricingProgram"]
