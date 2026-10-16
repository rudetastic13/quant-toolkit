"""template_jacobian — ``∂quote/∂z`` of calibration instruments over every curve they touch.

A :class:`~finance.pricing.calibration.instruments.CalibrationInstrument` carries one quoted
measure (a par rate, a simple rate, a futures rate) and knows how to compute it off a market.
This module differentiates that measure with respect to the market's curve parameters, laid out
by :class:`~finance.pricing.engines.layout.ParamLayout` (one block per owning key), and returns
the rows as a :class:`~finance.pricing.risk.pillar.PillarRisk` whose "positions" are the
instruments and whose ``pv`` is their implied quote.

Rows are batched by kind: every swap helper is lowered into one leg program and differentiated
in one adjoint sweep (then the par-rate quotient rule per instrument), futures helpers into one
futures program, deposits and FRAs through their curve's exact affine map.  Rows come back in
the order the helpers were given.

:func:`calibration_jacobian` is the calibrator's view of the same routine: the block of the key
being calibrated, which is exactly the parameters it owns.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from finance.dates import period_fractions
from finance.pricing.calibration.instruments import DepositHelper, FraHelper, FuturesHelper, SwapHelper
from finance.pricing.engines.layout import ParamLayout
from finance.pricing.risk.pillar import PillarRisk
from finance.pricing.types import Backend

FloatArray = np.ndarray


def _label(helper) -> str:
    return f"{helper.curve} {np.datetime_as_string(np.datetime64(helper.pillar_date, 'D'))}"


def _scatter(row: FloatArray, source: ParamLayout, target: ParamLayout, into: FloatArray) -> None:
    """Add ``row`` (over ``source``'s blocks) into ``into`` (over ``target``'s blocks), block by key."""
    for key in source.blocks:
        src = source.block_slices[key]
        dst = target.block_slices[key]
        if dst.stop - dst.start != src.stop - src.start:
            raise ValueError(
                f"block '{key}' has {src.stop - src.start} columns in one layout and {dst.stop - dst.start} in another"
            )
        into[dst] += row[src]


def template_jacobian(
    helpers: Sequence, market, *, curves: Sequence[str] | None = None, backend: Backend = Backend.Numba
) -> PillarRisk:
    """``∂implied_k/∂z`` for every helper, over every curve block the helpers touch.

    Parameters
    ----------
    helpers : sequence of CalibrationInstrument
        Rows, in this order.
    market : MarketContext
    curves : sequence of str, optional
        Owning keys to lay the columns out over, in this order.  Every block a helper touches
        must be among them; blocks no helper touches are zero columns.  Default: the touched
        blocks in order of first appearance.
    backend : Backend
        Only ``Numba`` has the adjoint.

    Returns
    -------
    PillarRisk
        ``values`` is ``J_T`` (K, P); ``pv`` is each helper's implied quote; ``labels`` name the
        rows (``"USD.SOFR.ON 2028-06-01"`` by default).
    """
    from finance.pricing.engines.numba.program import NumbaProgram
    from finance.pricing.pricers import FuturesPricer, SwapPricer

    if Backend(backend) != Backend.Numba:
        raise NotImplementedError(
            f"template_jacobian needs the Numba adjoint; backend {Backend(backend).name} has none"
        )
    helpers = list(helpers)
    if not helpers:
        raise ValueError("template_jacobian needs at least one helper")

    swaps = [(k, h) for k, h in enumerate(helpers) if isinstance(h, SwapHelper)]
    futures = [(k, h) for k, h in enumerate(helpers) if isinstance(h, FuturesHelper)]
    money = [(k, h) for k, h in enumerate(helpers) if isinstance(h, (DepositHelper, FraHelper))]
    others = [h for h in helpers if not isinstance(h, (SwapHelper, FuturesHelper, DepositHelper, FraHelper))]
    if others:
        raise NotImplementedError(f"no template Jacobian for helper type {type(others[0]).__name__}")

    swap_numba = futures_program = None
    names: list[str] = [h.curve for h in helpers]
    if swaps:
        inputs = SwapPricer().compile([h.swap for _, h in swaps]).inputs
        swap_numba = NumbaProgram(inputs, market)
        names += list(inputs.curve_names)
    if futures:
        futures_program = FuturesPricer().compile([h.future for _, h in futures])
        names += list(futures_program.program.inputs.curve_names)
    if curves is not None:
        names += list(curves)
    layout = ParamLayout.build(market, dict.fromkeys(names))
    if curves is not None:
        missing = [b for b in layout.blocks if b not in curves]
        if missing:
            raise ValueError(
                f"helpers touch curve blocks {missing} that are not in the template's curves {list(curves)}"
            )

    rows = np.zeros((len(helpers), layout.n_params), dtype=np.float64)
    implied = np.empty(len(helpers), dtype=np.float64)

    if swaps:
        result = swap_numba.value_and_grad(market)
        ki = swap_numba.inputs
        flow_gradient = result.cashflow_index_gradient + result.cashflow_funding_gradient
        leg_gradient = np.add.reduceat(flow_gradient, ki.leg_offsets, axis=0)
        leg_pv = result.primal.leg_pv
        for i, (k, h) in enumerate(swaps):
            legs = np.nonzero(ki.leg_instrument == i)[0]
            fixed, floating = legs[list(h._fixed_idx)], legs[list(h._float_idx)]
            fixed_pv, float_pv = float(leg_pv[fixed].sum()), float(leg_pv[floating].sum())
            fixed_g, float_g = leg_gradient[fixed].sum(axis=0), leg_gradient[floating].sum(axis=0)
            row = -(float_g * fixed_pv - float_pv * fixed_g) / (fixed_pv * fixed_pv)
            _scatter(row, swap_numba.layout, layout, rows[k])
            implied[k] = -float_pv / fixed_pv

    if futures:
        risk = futures_program.risk(market)
        numba = futures_program.program.prepare(market, backend=Backend.Numba)
        rates = futures_program.program.reprice(market).rate
        for i, (k, h) in enumerate(futures):
            _scatter(risk.rate_gradient[i], numba.layout, layout, rows[k])
            implied[k] = float(rates[i])

    for k, h in money:
        inst = getattr(h, "deposit", None) or getattr(h, "fra", None)
        curve = market.yield_curve(h.curve)
        window = np.array([inst.effective.to_numpy(), inst.maturity.to_numpy()], dtype="datetime64[D]")
        tau = float(period_fractions(inst.day_count_method, window[:1], window[1:])[0])
        offset, jac = curve.zero_curve.log_df_affine(curve.to_x(window))
        log_df = offset + jac @ curve.zero_curve.params
        ratio = float(np.exp(log_df[0] - log_df[1]))
        rows[k, layout.column_maps[h.curve]] += ratio / tau * (jac[0] - jac[1])
        implied[k] = (ratio - 1.0) / tau

    out = PillarRisk.from_layout(layout, rows, implied, labels=[_label(h) for h in helpers])
    if curves is not None:
        return out.aligned(tuple(curves), {c: layout.pillar_dates(c) for c in curves})
    return out


def calibration_jacobian(helpers: Sequence, target_name: str, market, curve=None) -> FloatArray:
    """``∂implied/∂z`` over the parameters ``target_name`` owns: the calibrator's Jacobian.

    ``curve`` is accepted for backward compatibility and ignored.
    """
    jac = template_jacobian(helpers, market)
    block = ParamLayout.build(market, [target_name]).block_of(target_name)
    return np.ascontiguousarray(jac.block(block))
