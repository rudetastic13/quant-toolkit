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
    MeasureContext,
    PillarGradient,
    declare_measures,
    dv01_from,
    gamma_1bp_from,
    gamma_ladder_from,
    measure,
    on_demand,
    resolve_requests,
)
from quant_toolkit.finance.pricing.results import CashflowReport, PricingResult
from quant_toolkit.finance.pricing.types import Backend, RateKind


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
        **settings,
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
            adjoint and are computed only when named.  ``pv01``, ``annuity`` and ``par_coupon``
            (one number per instrument, from its fixed coupon flows) are also on request, as
            is ``dv01`` (PV change for +1bp on every zero pillar of one curve), with
            ``gamma_1bp`` (the change in that ``dv01`` for a +1bp parallel move) and
            ``gamma_ladder`` (the same per pillar, a :class:`PillarRisk`).  A measure only another product answers
            (``"vega"``) is skipped and reads ``None`` from ``result.get``.
        backend : Backend, optional
        **settings
            ``curve`` : the curve ``dv01`` reports against.  Default: each instrument's own
            projection curve (its discount curve if it projects off nothing; ``nan`` if its
            legs project off several).
            ``gamma_step`` : the bump behind ``gamma_1bp`` / ``gamma_ladder``, zero-rate units
            (default ``1e-5``).
            ``template`` : a ``RiskTemplate`` (bound or not).  The ladders come back as
            ``QuoteRisk`` and ``dv01`` as +1bp on the template's quotes on the curve.

        Raises
        ------
        ValueError
            If a request is a name no program answers.
        TypeError
            If a setting is not recognised.
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
                is_principal=ki.is_principal,
            )
        return PricingResult(
            instrument_pv=kr.instrument_pv, leg_pv=kr.leg_pv, cashflows=cashflows,
            extras=on_demand(self, market, asked, {"kernel": kr}, settings),
        )

    def pillar_gradient(self, market) -> PillarGradient:
        """Total (projection + discounting) ``d PV / d z`` from the Numba adjoint."""
        numba = self.prepare(market, backend=Backend.Numba)
        result = numba.value_and_grad(market)
        return PillarGradient(numba.layout, result.instrument_total_gradient, result.primal.instrument_pv)


def _adjoint(program: PricingProgram, market, context: MeasureContext):
    """The Numba adjoint result, run once per ``price`` call."""
    cache = context.cache
    if "adjoint" not in cache:
        cache["adjoint"] = program.prepare(market, backend=Backend.Numba).value_and_grad(market)
    return cache["adjoint"]


def _ladder(program: PricingProgram, market, context: MeasureContext, attribute: str):
    from quant_toolkit.finance.pricing.risk.pillar import PillarRisk

    result = _adjoint(program, market, context)
    pillar = PillarRisk.from_layout(result.layout, getattr(result, attribute), result.primal.instrument_pv)
    return context.ladder(market, pillar)


@measure(PricingProgram, "dv01")
def _dv01(program, market, context):
    """PV change for +1bp on every zero pillar of one curve; see :func:`dv01_from`."""
    result = _adjoint(program, market, context)
    gradient = PillarGradient(result.layout, result.instrument_total_gradient, result.primal.instrument_pv)
    return dv01_from(gradient, program.inputs, market, context)


@measure(PricingProgram, "zero_ladder")
def _zero_ladder(program, market, context):
    """Total ``d PV / d z`` per instrument (projection + discounting)."""
    return _ladder(program, market, context, "instrument_total_gradient")


@measure(PricingProgram, "index_ladder")
def _index_ladder(program, market, context):
    """``d PV / d z`` through projection only."""
    return _ladder(program, market, context, "instrument_index_gradient")


@measure(PricingProgram, "funding_ladder")
def _funding_ladder(program, market, context):
    """``d PV / d z`` through discounting only."""
    return _ladder(program, market, context, "instrument_funding_gradient")


def _fixed_coupons(program: PricingProgram, market, context: MeasureContext):
    """Signed annuity and PV of the fixed coupon flows, per instrument.

    Returns
    -------
    annuity : ndarray of float64, shape (n_instruments,)
        ``sum(sign * notional * period_frac * df)`` over fixed coupon flows: ``d PV / d coupon``.
        Principal flows are excluded (they are fixed flows too; ``is_principal`` tells them apart).
    coupon_pv : ndarray of float64, shape (n_instruments,)
        PV of those same flows at their contract coupons.
    pv : ndarray of float64, shape (n_instruments,)
    """
    cache = context.cache
    if "fixed_coupons" not in cache:
        ki = program.inputs
        kr = cache["kernel"] if "kernel" in cache else program.reprice(market)
        if kr.df is None:
            raise NotImplementedError("pv01 / annuity / par_coupon need payment discount factors from the engine")
        leg_of_flow = np.repeat(np.arange(ki.n_legs), np.diff(np.append(ki.leg_offsets, ki.n_flows)))
        instrument = ki.leg_instrument[leg_of_flow]
        coupon = (ki.rate_kind == RateKind.Fixed) & ~ki.is_principal
        n = ki.n_instruments
        annuity = np.bincount(
            instrument[coupon], (ki.sign * ki.notional * ki.period_frac * kr.df)[coupon], minlength=n
        )
        coupon_pv = np.bincount(instrument[coupon], kr.flow_pv[coupon], minlength=n)
        cache["fixed_coupons"] = annuity, coupon_pv, kr.instrument_pv
    return cache["fixed_coupons"]


@measure(PricingProgram, "gamma_1bp")
def _gamma_1bp(program, market, context):
    """Change in ``dv01`` for a +1bp parallel move of the same curve; see :func:`gamma_1bp_from`."""
    return gamma_1bp_from(program, program.inputs, market, context)


@measure(PricingProgram, "gamma_ladder")
def _gamma_ladder(program, market, context):
    """Per-pillar change in 01 for that move; see :func:`gamma_ladder_from`."""
    return gamma_ladder_from(program, program.inputs, market, context)


@measure(PricingProgram, "pv01")
def _pv01(program, market, context):
    """PV change for +1bp on the fixed coupon, signed; 0 for an instrument with no fixed coupon."""
    return _fixed_coupons(program, market, context)[0] * 1e-4


@measure(PricingProgram, "annuity")
def _annuity(program, market, context):
    """PV of 1.0 of fixed coupon, unsigned; 0 for an instrument with no fixed coupon."""
    return np.abs(_fixed_coupons(program, market, context)[0])


@measure(PricingProgram, "par_coupon")
def _par_coupon(program, market, context):
    """The flat fixed coupon that makes PV zero; ``nan`` where there is no fixed coupon to solve for."""
    annuity, coupon_pv, pv = _fixed_coupons(program, market, context)
    out = np.full(annuity.shape, np.nan)
    np.divide(-(pv - coupon_pv), annuity, out=out, where=annuity != 0.0)
    return out


__all__ = ["PricingProgram"]
