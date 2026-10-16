"""Analytic calibration Jacobian using the Numba pricing adjoint."""
from __future__ import annotations

import numpy as np

from finance.dates import period_fractions
from finance.pricing.calibration.instruments import DepositHelper, FraHelper, SwapHelper
from finance.pricing.engines.layout import ParamLayout
from finance.pricing.engines.numba.program import NumbaProgram


def _money_market_gradient(helper, curve, own: np.ndarray) -> np.ndarray:
    """``d simple_rate / dz`` over ``[effective, maturity]`` for the solved columns ``own`` of ``curve``'s parameters."""
    inst = getattr(helper, "deposit", None) or getattr(helper, "fra", None)
    window = np.array([inst.effective.to_numpy(), inst.maturity.to_numpy()], dtype="datetime64[D]")
    tau = float(period_fractions(inst.day_count_method, window[:1], window[1:])[0])
    offset, jac = curve.zero_curve.log_df_affine(curve.to_x(window))
    log_df = offset + jac @ curve.zero_curve.params
    ratio = float(np.exp(log_df[0] - log_df[1]))
    return ratio / tau * (jac[0, own] - jac[1, own])


def _swap_gradient(helper: SwapHelper, target_name: str, market) -> np.ndarray:
    program = NumbaProgram(helper._program.inputs, market)
    result = program.value_and_grad(market)
    flow_gradient = result.cashflow_gradient(target_name, role="total")
    leg_gradient = np.add.reduceat(flow_gradient, helper._program.inputs.leg_offsets, axis=0)
    leg_pv = result.primal.leg_pv
    fixed_pv = float(sum(leg_pv[i] for i in helper._fixed_idx))
    float_pv = float(sum(leg_pv[i] for i in helper._float_idx))
    fixed_gradient = sum((leg_gradient[i] for i in helper._fixed_idx), start=np.zeros(flow_gradient.shape[1]))
    float_gradient = sum((leg_gradient[i] for i in helper._float_idx), start=np.zeros(flow_gradient.shape[1]))
    return -(float_gradient * fixed_pv - float_pv * fixed_gradient) / (fixed_pv * fixed_pv)


def calibration_jacobian(helpers, target_name: str, market, curve) -> np.ndarray:
    """Return ``d(implied quote) / dz`` over the parameters ``target_name`` owns, without JAX tracing."""
    layout = ParamLayout.build(market, [target_name])
    block = layout.block_slice(target_name)
    column_map = layout.column_maps[target_name]
    own = np.nonzero((column_map >= block.start) & (column_map < block.stop))[0]
    rows = []
    for helper in helpers:
        if isinstance(helper, (DepositHelper, FraHelper)):
            rows.append(_money_market_gradient(helper, curve, own))
        elif isinstance(helper, SwapHelper):
            rows.append(_swap_gradient(helper, target_name, market))
        else:  # futures and other helpers can expose their own analytic gradient later
            gradient = getattr(helper, "numba_gradient", None)
            if gradient is None:
                raise NotImplementedError(
                    f"no Numba calibration adjoint for helper type {type(helper).__name__}"
                )
            rows.append(np.asarray(gradient(market), dtype=np.float64))
    return np.vstack(rows)


__all__ = ["calibration_jacobian"]
