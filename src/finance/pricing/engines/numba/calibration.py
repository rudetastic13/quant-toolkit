"""Analytic calibration Jacobian using the Numba pricing adjoint."""
from __future__ import annotations

import numpy as np

from finance.dates import period_fractions
from finance.pricing.calibration.instruments import DepositHelper, FraHelper, SwapHelper
from finance.pricing.engines.numba.program import NumbaProgram


def _logdf_gradient(curve, date: np.datetime64) -> tuple[float, np.ndarray]:
    """Log-linear DF and ``d ln(DF) / dz`` at one date (non-origin pillars only).

    ``curve`` is the calibrated ``YieldCurve`` (origin + pure ``ZeroCurve``).
    """
    x = curve.zero_curve.x
    z = curve.zero_curve.node_zero_rates
    q = float(curve.to_x(np.array([date], dtype="datetime64[D]"))[0])
    grad = np.zeros(z.size, dtype=np.float64)
    if q <= x[0]:
        return 1.0, grad
    if q >= x[-1]:
        left, right = x.size - 2, x.size - 1
        w_right = 1.0 + (q - x[right]) / (x[right] - x[left])
    else:
        right = int(np.searchsorted(x, q, side="right"))
        left = right - 1
        w_right = (q - x[left]) / (x[right] - x[left])
    w_left = 1.0 - w_right
    log_left = 0.0 if left == 0 else -z[left - 1] * (x[left] / 365.0)
    log_right = -z[right - 1] * (x[right] / 365.0)
    if left > 0:
        grad[left - 1] = -(x[left] / 365.0) * w_left
    grad[right - 1] += -(x[right] / 365.0) * w_right
    return float(np.exp(w_left * log_left + w_right * log_right)), grad


def _money_market_gradient(helper, curve) -> np.ndarray:
    inst = getattr(helper, "deposit", None) or getattr(helper, "fra", None)
    start = np.datetime64(inst.effective.to_numpy(), "D")
    end = np.datetime64(inst.maturity.to_numpy(), "D")
    tau = float(period_fractions(inst.day_count_method, np.array([start]), np.array([end]))[0])
    df_start, grad_start = _logdf_gradient(curve, start)
    df_end, grad_end = _logdf_gradient(curve, end)
    ratio = df_start / df_end
    return ratio / tau * (grad_start - grad_end)


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
    """Return ``d(implied quote) / dz`` without JAX tracing or shape compilation."""
    rows = []
    for helper in helpers:
        if isinstance(helper, (DepositHelper, FraHelper)):
            rows.append(_money_market_gradient(helper, curve))
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
