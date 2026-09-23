"""European Black/Bachelier swaption pricer with analytic option and curve greeks."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from finance.instruments.resolution.swaption import Swaption, SwaptionModel
from finance.instruments.resolution.resolver import curve_name
from finance.markets.vols import VolUnits
from finance.pricing.engines.numpy.option import OptionGreeks
from finance.pricing.pricers.base import PricingProgram
from finance.pricing.pricers.swap import SwapPricer
from finance.pricing.types import Backend

# Quoting convention each model consumes; enforced against VolSurface.units at lookup.
_SURFACE_UNITS = {
    SwaptionModel.Black: VolUnits.Lognormal,
    SwaptionModel.Bachelier: VolUnits.Normal,
}


@dataclass(frozen=True)
class SwaptionPricingResult:
    """Reprice output; ``greeks`` are monetary — every unit (per-annuity) greek scaled by ``annuity``."""

    pv: np.ndarray
    forward: np.ndarray
    annuity: np.ndarray
    volatility: np.ndarray
    greeks: OptionGreeks


@dataclass(frozen=True)
class SwaptionRiskResult:
    curve_gradient: np.ndarray
    dv01: np.ndarray


def _unit_fixed_swap(swap):
    receive = swap.receive_leg
    pay = swap.pay_leg
    if receive.coupon_type.is_floating:
        pay = dataclasses.replace(pay, coupon_rate=1.0)
    else:
        receive = dataclasses.replace(receive, coupon_rate=1.0)
    return dataclasses.replace(swap, receive_leg=receive, pay_leg=pay)


class SwaptionProgram:
    """Compiled underlying swaps, repriced without rebuilding option geometry."""

    def __init__(self, swaptions: list[Swaption], backend: Backend):
        if not swaptions:
            raise ValueError("SwaptionPricer requires at least one swaption")
        self.swaptions = tuple(swaptions)
        unit_swaps = [_unit_fixed_swap(option.underlying) for option in swaptions]
        self.program: PricingProgram = SwapPricer().compile(unit_swaps, backend=backend)
        fixed_idx = []
        float_idx = []
        for inst, option in enumerate(swaptions):
            for local, leg in enumerate(option.underlying):
                (float_idx if leg.coupon_type.is_floating else fixed_idx).append(2 * inst + local)
        self.fixed_leg = np.asarray(fixed_idx, dtype=np.int64)
        self.float_leg = np.asarray(float_idx, dtype=np.int64)

    def _underlying(self, market):
        raw = self.program.reprice(market)
        fixed_pv = raw.leg_pv[self.fixed_leg]
        float_pv = raw.leg_pv[self.float_leg]
        forward = -float_pv / fixed_pv
        annuity = np.abs(fixed_pv)
        return raw, fixed_pv, float_pv, forward, annuity

    def _option_inputs(self, market, forward):
        if market.vols is None:
            raise ValueError("swaption pricing requires MarketContext.vols")
        expiry = np.empty(len(self.swaptions), dtype=np.float64)
        tenor = np.empty_like(expiry)
        strike = np.empty_like(expiry)
        vol = np.empty_like(expiry)
        as_of = np.datetime64(market.as_of_date.to_str(), "D")
        for i, option in enumerate(self.swaptions):
            fixed_leg = next(leg for leg in option.underlying if not leg.coupon_type.is_floating)
            expiry[i] = max((option.expiry.to_numpy() - as_of).astype(int) / 365.0, 0.0)
            tenor[i] = (fixed_leg.maturity.to_numpy() - fixed_leg.effective.to_numpy()).astype(int) / 365.0
            strike[i] = float(fixed_leg.coupon_rate)
            name = option.vol_name or curve_name(option.underlying.currency, option.underlying.index_name)
            surface = market.vols.resolve(name)
            required = _SURFACE_UNITS[option.model]
            if surface.units != required:
                raise ValueError(
                    f"swaption model {option.model.name} requires a {required.name} vol surface; "
                    f"'{name}' is quoted {surface.units.name}"
                )
            vol[i] = float(surface.vol(expiry[i], tenor[i], strike[i], forward[i]))
        return strike, expiry, vol

    def _unit_greeks(self, forward, strike, expiry, vol) -> OptionGreeks:
        if self.program.backend == Backend.Numba:
            from finance.pricing.engines.numba import bachelier_greeks, black_greeks
        else:
            from finance.pricing.engines.numpy.option import bachelier_greeks, black_greeks

        arrays = [np.empty_like(forward) for _ in range(6)]
        for model, fn in (
            (SwaptionModel.Black, black_greeks),
            (SwaptionModel.Bachelier, bachelier_greeks),
        ):
            mask = np.array([option.model == model for option in self.swaptions])
            for is_call in (False, True):
                selected = mask & np.array([option.payer == is_call for option in self.swaptions])
                if not selected.any():
                    continue
                greeks = fn(
                    forward[selected], strike[selected], expiry[selected], vol[selected], is_call=is_call
                )
                for target, source in zip(arrays, dataclasses.astuple(greeks)):
                    target[selected] = source
        return OptionGreeks(*arrays)

    def reprice(self, market) -> SwaptionPricingResult:
        _, _, _, forward, annuity = self._underlying(market)
        strike, expiry, vol = self._option_inputs(market, forward)
        unit = self._unit_greeks(forward, strike, expiry, vol)
        monetary = OptionGreeks(*(annuity * value for value in dataclasses.astuple(unit)))
        return SwaptionPricingResult(
            pv=monetary.value,
            forward=forward,
            annuity=annuity,
            volatility=vol,
            greeks=monetary,
        )

    def risk(self, market, curve: str, *, bp: float = 1e-4) -> SwaptionRiskResult:
        """Chain option delta through the underlying forward and annuity adjoints.

        Always runs on the Numba adjoint regardless of the reprice backend (numpy has no
        analytic adjoint), and holds the quoted vol fixed under curve moves (sticky
        strike) — there is no dvol/dforward cross term.
        """
        numba_program = self.program.prepare(market, backend=Backend.Numba)
        adjoint = numba_program.value_and_grad(market)
        flow_gradient = adjoint.cashflow_gradient(curve, role="total")
        leg_gradient = np.add.reduceat(flow_gradient, self.program.inputs.leg_offsets, axis=0)
        fixed_gradient = leg_gradient[self.fixed_leg]
        float_gradient = leg_gradient[self.float_leg]
        fixed_pv = adjoint.primal.leg_pv[self.fixed_leg]
        float_pv = adjoint.primal.leg_pv[self.float_leg]
        forward = -float_pv / fixed_pv
        annuity = np.abs(fixed_pv)
        d_forward = -(
            float_gradient * fixed_pv[:, None] - float_pv[:, None] * fixed_gradient
        ) / (fixed_pv * fixed_pv)[:, None]
        d_annuity = np.sign(fixed_pv)[:, None] * fixed_gradient

        strike, expiry, vol = self._option_inputs(market, forward)
        unit = self._unit_greeks(forward, strike, expiry, vol)
        curve_gradient = unit.value[:, None] * d_annuity + annuity[:, None] * unit.delta[:, None] * d_forward
        return SwaptionRiskResult(curve_gradient=curve_gradient, dv01=curve_gradient * bp)


class SwaptionPricer:
    # Numpy is the always-available reference (numba is optional and LogLinearDF-only);
    # pass Backend.Numba for the fused hot path.  risk() uses the Numba adjoint either way.
    default_backend = Backend.Numpy

    def compile(self, swaptions: list[Swaption], *, backend: Backend | None = None) -> SwaptionProgram:
        selected = self.default_backend if backend is None else Backend(backend)
        if selected not in (Backend.Numpy, Backend.Numba):
            raise NotImplementedError(f"backend {selected.name} is not implemented by SwaptionPricer")
        return SwaptionProgram(swaptions, selected)

    def price(self, swaptions: list[Swaption], market, *, backend: Backend | None = None):
        return self.compile(swaptions, backend=backend).reprice(market)


__all__ = ["SwaptionPricer", "SwaptionProgram", "SwaptionPricingResult", "SwaptionRiskResult"]
