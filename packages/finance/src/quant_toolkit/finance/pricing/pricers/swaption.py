"""European Black/Bachelier swaption pricer with analytic option and curve greeks."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

import numpy as np

from quant_toolkit.common.registry import register_with
from quant_toolkit.finance.instruments.priceable import compile_instruments, pricer_registry
from quant_toolkit.finance.instruments.resolution.swaption import Swaption
from quant_toolkit.finance.instruments.resolution.resolver import curve_name
from quant_toolkit.finance.markets.vols import VolUnits
from quant_toolkit.finance.pricing.engines import kernel, select_backend
from quant_toolkit.finance.pricing.engines.numpy.option import OptionGreeks
from quant_toolkit.finance.pricing.pricers.base import PricingProgram
from quant_toolkit.finance.pricing.pricers.protocol import (
    PillarGradient,
    declare_measures,
    dv01_from,
    measure,
    on_demand,
    pillar_measure,
    resolve_requests,
)
from quant_toolkit.finance.pricing.results import measures_dict, read_measure
from quant_toolkit.finance.pricing.types import KERNEL_BACHELIER, KERNEL_BLACK, Backend

# The option model follows the cube: its quoting convention names the kernel in ``kernel_registry``.
OPTION_KERNEL = {
    VolUnits.Lognormal: KERNEL_BLACK,
    VolUnits.Normal: KERNEL_BACHELIER,
}


@dataclass(frozen=True)
class SwaptionPricingResult:
    """Reprice output; ``greeks`` are monetary — every unit (per-annuity) greek scaled by ``annuity``
    and by the option's ``position`` (so a sold option has negative PV and vega)."""

    pv: np.ndarray
    forward: np.ndarray
    annuity: np.ndarray
    volatility: np.ndarray
    greeks: OptionGreeks
    extras: dict[str, Any] = field(default_factory=dict)   # on-demand measures, by request name

    fields: ClassVar[dict[str, str]] = {
        "pv": "pv", "forward": "forward", "annuity": "annuity", "volatility": "volatility", "greeks": "greeks",
        "delta": "greeks.delta", "gamma": "greeks.gamma", "vega": "greeks.vega",
        "vanna": "greeks.vanna", "volga": "greeks.volga",
    }

    def get(self, measure: str):
        """Measure by request name; ``None`` if a swaption does not answer it."""
        return read_measure(self, self.fields, measure)

    def as_dict(self, measures=None) -> dict[str, Any]:
        """``{request name: value}``; see :func:`measures_dict`."""
        return measures_dict(self, self.fields, measures)

    @property
    def value(self) -> np.ndarray:
        """PV per option: the number risk differentiates."""
        return self.pv


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

    measures = declare_measures(SwaptionPricingResult.fields)

    def __init__(self, swaptions: list[Swaption], backend: Backend):
        if not swaptions:
            raise ValueError("SwaptionPricer requires at least one swaption")
        self.swaptions = tuple(swaptions)
        unit_swaps = [_unit_fixed_swap(option.underlying) for option in swaptions]
        self.program: PricingProgram = compile_instruments(unit_swaps, backend=backend)
        # Program legs are laid out in instrument order, each underlying's legs in its own
        # order — classify by walking them, never by assuming two legs per swap.
        fixed_idx = []
        float_idx = []
        leg_index = 0
        for option in swaptions:
            for leg, _sign in option.underlying.legs:
                (float_idx if leg.is_floating else fixed_idx).append(leg_index)
                leg_index += 1
        self.position = np.array([option.position for option in swaptions], dtype=np.float64)
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
        """Strike, expiry, tenor and vol per option, plus the ``(name, cube, mask)`` groups the vols came from."""
        if market.vols is None:
            raise ValueError("swaption pricing requires MarketContext.vols")
        expiry = np.empty(len(self.swaptions), dtype=np.float64)
        tenor = np.empty_like(expiry)
        strike = np.empty_like(expiry)
        vol = np.empty_like(expiry)
        names = []
        as_of = np.datetime64(market.origin, "D")
        for i, option in enumerate(self.swaptions):
            fixed_leg = next(leg for leg in option.underlying if not leg.is_floating)
            expiry[i] = max((option.expiry.to_numpy() - as_of).astype(int) / 365.0, 0.0)
            tenor[i] = (fixed_leg.maturity.to_numpy() - fixed_leg.effective.to_numpy()).astype(int) / 365.0
            strike[i] = float(fixed_leg.coupon_rate)
            names.append(option.vol_name or curve_name(option.underlying.currency, option.underlying.index_name))
        groups = []
        for name in dict.fromkeys(names):
            cube = market.vols.resolve(name)
            mask = np.array([n == name for n in names])
            vol[mask] = cube.vol(expiry[mask], tenor[mask], strike[mask], forward[mask])
            groups.append((name, cube, mask))
        return strike, expiry, tenor, vol, groups

    def _unit_greeks(self, forward, strike, expiry, vol, groups) -> OptionGreeks:
        """Per-annuity value and greeks, each cube's options through the kernel its units name."""
        backend = self.program.backend
        arrays = [np.empty_like(forward) for _ in range(6)]
        payer = np.array([option.payer for option in self.swaptions])
        for _name, cube, mask in groups:
            kernel_id = OPTION_KERNEL[cube.units]
            for is_call in (False, True):
                selected = mask & (payer == is_call)
                if not selected.any():
                    continue
                greeks = kernel(
                    (kernel_id, backend),
                    forward[selected], strike[selected], expiry[selected], vol[selected], is_call=is_call
                )
                for target, source in zip(arrays, dataclasses.astuple(greeks)):
                    target[selected] = source
        return OptionGreeks(*arrays)

    @property
    def n_positions(self) -> int:
        return len(self.swaptions)

    def price(self, market, requests: Sequence[str] | None = None, **settings) -> SwaptionPricingResult:
        """PV, forward, annuity, vol and monetary greeks per option.

        ``requests`` is checked against the known vocabulary; the kernel returns value and
        greeks in one pass, so every field is always populated.  ``zero_ladder`` (curve delta at
        sticky strike as a :class:`PillarRisk`) runs the Numba adjoint and is computed only when
        named.
        """
        asked = resolve_requests(requests, self.measures, SwaptionPricingResult.fields)
        _, _, _, forward, annuity = self._underlying(market)
        strike, expiry, _, vol, groups = self._option_inputs(market, forward)
        unit = self._unit_greeks(forward, strike, expiry, vol, groups)
        monetary = OptionGreeks(*(self.position * annuity * value for value in dataclasses.astuple(unit)))
        return SwaptionPricingResult(
            pv=monetary.value,
            forward=forward,
            annuity=annuity,
            volatility=vol,
            greeks=monetary,
            extras=on_demand(self, market, asked, settings=settings),
        )

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

        strike, expiry, tenor, vol, groups = self._option_inputs(market, forward)
        unit = self._unit_greeks(forward, strike, expiry, vol, groups)
        curve_gradient = self.position[:, None] * (
            unit.value[:, None] * d_annuity + annuity[:, None] * unit.delta[:, None] * d_forward
        )
        vega = self.position * annuity * unit.vega
        vol_gradient = {}
        for name, cube, mask in groups:
            gradient = np.zeros((len(self.swaptions), cube.params.size))
            gradient[mask] = vega[mask, None] * cube.vol_gradient(expiry[mask], tenor[mask], strike[mask], forward[mask])
            vol_gradient[name] = gradient
        return SwaptionRiskResult(curve_gradient=curve_gradient, dv01=curve_gradient * bp, vol_gradient=vol_gradient)

    def pillar_gradient(self, market) -> PillarGradient:
        """Curve delta at sticky strike over every curve parameter."""
        numba = self.program.prepare(market, backend=Backend.Numba)
        return PillarGradient(numba.layout, self.risk(market).curve_gradient, self.price(market, ["pv"]).value)


measure(SwaptionProgram, "zero_ladder")(pillar_measure)


@measure(SwaptionProgram, "dv01")
def _dv01(program, market, context):
    """Change in PV at sticky strike for +1bp on every zero pillar of one curve; see :func:`dv01_from`."""
    return dv01_from(program.pillar_gradient(market), program.program.inputs, context)


@register_with(pricer_registry, Swaption, overwrite=True)
class SwaptionPricer:
    # Numpy is the always-available reference (numba is optional); pass Backend.Numba for
    # the fused hot path.  risk() uses the Numba adjoint either way.
    default_backend = Backend.Numpy

    def compile(self, swaptions: list[Swaption], *, backend: Backend | None = None) -> SwaptionProgram:
        return SwaptionProgram(swaptions, select_backend(backend, self.default_backend))

    def price(self, swaptions: list[Swaption], market, *, backend: Backend | None = None):
        return self.compile(swaptions, backend=backend).price(market)


__all__ = ["SwaptionPricer", "SwaptionProgram", "SwaptionPricingResult", "SwaptionRiskResult"]
