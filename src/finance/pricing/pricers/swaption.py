"""European Black/Bachelier swaption pricer with analytic option and curve greeks."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from finance.instruments.priceable import pricer_registry
from finance.instruments.resolution.swaption import Swaption
from finance.instruments.resolution.resolver import curve_name
from finance.markets.vols import VolUnits
from finance.pricing.engines.numpy.option import OptionGreeks
from finance.pricing.pricers.base import PricingProgram
from finance.pricing.pricers.swap import SwapPricer
from finance.pricing.types import Backend

# The option model follows the quoting units of the cube an option resolves.
_MODEL = {VolUnits.Lognormal: "black_greeks", VolUnits.Normal: "bachelier_greeks"}


@dataclass(frozen=True)
class SwaptionPricingResult:
    """Reprice output; ``greeks`` are monetary — every unit (per-annuity) greek scaled by ``annuity``.

    ``pv`` and ``greeks`` carry the position's sign (a written option is negative);
    ``forward``, ``annuity`` and ``volatility`` describe the underlying and do not.
    """

    pv: np.ndarray
    forward: np.ndarray
    annuity: np.ndarray
    volatility: np.ndarray
    greeks: OptionGreeks


@dataclass(frozen=True)
class SwaptionRiskResult:
    """Curve and vol risk per option.

    ``vol_gradient[name]`` is ``d pv / d params`` of the cube bound under ``name``, shape
    ``(n_options, n_params)``; an option priced off another cube has a zero row.
    """

    curve_gradient: np.ndarray
    dv01: np.ndarray
    vol_gradient: dict[str, np.ndarray]


def _unit_fixed_swap(swap):
    receive = swap.receive_leg
    pay = swap.pay_leg
    if receive.is_floating:
        pay = pay.replace(coupon_rate=1.0)
    else:
        receive = receive.replace(coupon_rate=1.0)
    return dataclasses.replace(swap, receive_leg=receive, pay_leg=pay)


class SwaptionProgram:
    """Compiled underlying swaps, repriced without rebuilding option geometry."""

    def __init__(self, swaptions: list[Swaption], backend: Backend):
        if not swaptions:
            raise ValueError("SwaptionPricer requires at least one swaption")
        self.swaptions = tuple(swaptions)
        self.strike = np.array([option.strike for option in swaptions], dtype=np.float64)
        self.payer = np.array([option.payer for option in swaptions])
        self.position = np.array([1.0 if option.long else -1.0 for option in swaptions])
        unit_swaps = [_unit_fixed_swap(option.underlying) for option in swaptions]
        self.program: PricingProgram = SwapPricer().compile(unit_swaps, backend=backend)
        # Program legs are laid out in instrument order, each underlying's legs in its own
        # order — classify by walking them, never by assuming two legs per swap.
        fixed_idx = []
        float_idx = []
        leg_index = 0
        for option in swaptions:
            for leg, _sign in option.underlying.legs:
                (float_idx if leg.is_floating else fixed_idx).append(leg_index)
                leg_index += 1
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
        """Expiry, tenor and vol per option, plus the ``(name, cube, mask)`` groups the vols came from."""
        if market.vols is None:
            raise ValueError("swaption pricing requires MarketContext.vols")
        expiry = np.empty(len(self.swaptions), dtype=np.float64)
        tenor = np.empty_like(expiry)
        vol = np.empty_like(expiry)
        names = []
        as_of = np.datetime64(market.origin, "D")
        for i, option in enumerate(self.swaptions):
            fixed_leg = option.fixed_leg
            expiry[i] = max((option.expiry.to_numpy() - as_of).astype(int) / 365.0, 0.0)
            tenor[i] = (fixed_leg.maturity.to_numpy() - fixed_leg.effective.to_numpy()).astype(int) / 365.0
            names.append(option.vol_name or curve_name(option.underlying.currency, option.underlying.index_name))
        groups = []
        for name in dict.fromkeys(names):
            cube = market.vols.resolve(name)
            mask = np.array([n == name for n in names])
            vol[mask] = cube.vol(expiry[mask], tenor[mask], self.strike[mask], forward[mask])
            groups.append((name, cube, mask))
        return expiry, tenor, vol, groups

    def _unit_greeks(self, forward, expiry, vol, groups) -> OptionGreeks:
        """Per-annuity greeks of a long option, each group through the model its cube's units name."""
        if self.program.backend == Backend.Numba:
            from finance.pricing.engines import numba as kernels
        else:
            from finance.pricing.engines.numpy import option as kernels

        arrays = [np.empty_like(forward) for _ in range(6)]
        for _name, cube, mask in groups:
            fn = getattr(kernels, _MODEL[cube.units])
            for is_call in (False, True):
                selected = mask & (self.payer == is_call)
                if not selected.any():
                    continue
                greeks = fn(
                    forward[selected], self.strike[selected], expiry[selected], vol[selected], is_call=is_call
                )
                for target, source in zip(arrays, dataclasses.astuple(greeks)):
                    target[selected] = source
        return OptionGreeks(*arrays)

    def reprice(self, market) -> SwaptionPricingResult:
        _, _, _, forward, annuity = self._underlying(market)
        expiry, _, vol, groups = self._option_inputs(market, forward)
        unit = self._unit_greeks(forward, expiry, vol, groups)
        monetary = OptionGreeks(*(self.position * annuity * value for value in dataclasses.astuple(unit)))
        return SwaptionPricingResult(
            pv=monetary.value,
            forward=forward,
            annuity=annuity,
            volatility=vol,
            greeks=monetary,
        )

    def price(self, market, *, with_cashflows: bool = True) -> SwaptionPricingResult:
        """``Priceable`` entry point; an option has no cashflow report, so the flag is unused."""
        return self.reprice(market)

    def risk(self, market, curve: str | None = None, *, bp: float = 1e-4) -> SwaptionRiskResult:
        """Chain option delta through the underlying forward and annuity adjoints.

        Over the block ``curve`` reports under, or over every parameter block of the prepared
        program when ``curve`` is ``None``.  Always runs on the Numba adjoint regardless of the
        reprice backend (numpy has no analytic adjoint), and holds the quoted vol fixed under
        curve moves (sticky strike) — there is no dvol/dforward cross term.  Vol risk is
        monetary vega spread onto each cube's parameters by ``ModelVolCube.vol_gradient``.
        """
        numba_program = self.program.prepare(market, backend=Backend.Numba)
        adjoint = numba_program.value_and_grad(market)
        if curve is None:
            flow_gradient = adjoint.cashflow_index_gradient + adjoint.cashflow_funding_gradient
        else:
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

        expiry, tenor, vol, groups = self._option_inputs(market, forward)
        unit = self._unit_greeks(forward, expiry, vol, groups)
        curve_gradient = self.position[:, None] * (
            unit.value[:, None] * d_annuity + annuity[:, None] * unit.delta[:, None] * d_forward
        )
        vega = self.position * annuity * unit.vega
        vol_gradient = {}
        for name, cube, mask in groups:
            gradient = np.zeros((len(self.swaptions), cube.params.size))
            gradient[mask] = vega[mask, None] * cube.vol_gradient(
                expiry[mask], tenor[mask], self.strike[mask], forward[mask]
            )
            vol_gradient[name] = gradient
        return SwaptionRiskResult(curve_gradient=curve_gradient, dv01=curve_gradient * bp, vol_gradient=vol_gradient)


class SwaptionPricer:
    # Numpy is the always-available reference (numba is optional); pass Backend.Numba for
    # the fused hot path.  risk() uses the Numba adjoint either way.
    default_backend = Backend.Numpy

    def compile(self, swaptions: list[Swaption], *, backend: Backend | None = None) -> SwaptionProgram:
        selected = self.default_backend if backend is None else Backend(backend)
        if selected not in (Backend.Numpy, Backend.Numba):
            raise NotImplementedError(f"backend {selected.name} is not implemented by SwaptionPricer")
        return SwaptionProgram(swaptions, selected)

    def price(self, swaptions: list[Swaption], market, *, backend: Backend | None = None):
        return self.compile(swaptions, backend=backend).reprice(market)


pricer_registry.register("Swaption", SwaptionPricer, overwrite=True)

__all__ = ["SwaptionPricer", "SwaptionProgram", "SwaptionPricingResult", "SwaptionRiskResult"]
