"""Numba option kernels (Black / Bachelier with greeks), registered under ``(kernel_id, Backend.Numba)``.

The rate and DCF reductions have no Numba twins: measured, they were 1.1-1.4x (``compounded``) and a
few microseconds (``dcf``) over numpy, and the fused :class:`NumbaProgram` is what makes the Numba
engine fast.  Option kernels stay narrow because pricers call them at exactly this granularity
(forward, strike, expiry, vol in; value and greeks out).  Numba specializes on dtype/layout, not
array length, so one compiled signature serves every shape.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from quant_toolkit.common.registry import register_with
from quant_toolkit.finance.pricing.engines.registry import kernel_registry
from quant_toolkit.finance.pricing.engines.numpy.option import OptionGreeks
from quant_toolkit.finance.pricing.types import Backend, KERNEL_BACHELIER, KERNEL_BLACK

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _float_1d(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


@njit(cache=True, inline="always")  # pragma: no cover - executed as Numba machine code
def _cdf(x):
    return 0.5 * math.erfc(-x * _INV_SQRT_2)


@njit(cache=True, inline="always")  # pragma: no cover - executed as Numba machine code
def _pdf(x):
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(cache=True)  # pragma: no cover - executed as Numba machine code
def _bachelier_all(forward, strike, expiry, vol, cp):
    n = forward.size
    value = np.empty(n)
    delta = np.empty(n)
    gamma = np.zeros(n)
    vega = np.zeros(n)
    vanna = np.zeros(n)
    volga = np.zeros(n)
    for i in range(n):
        sqrt_t = math.sqrt(expiry[i])
        std = vol[i] * sqrt_t
        intrinsic = max(cp * (forward[i] - strike[i]), 0.0)
        if std <= 0.0:
            value[i] = intrinsic
            delta[i] = cp if cp * (forward[i] - strike[i]) > 0.0 else 0.0
            continue
        d = (forward[i] - strike[i]) / std
        pdf = _pdf(d)
        value[i] = cp * (forward[i] - strike[i]) * _cdf(cp * d) + std * pdf
        delta[i] = cp * _cdf(cp * d)
        gamma[i] = pdf / std
        vega[i] = sqrt_t * pdf
        vanna[i] = -d * pdf / vol[i]
        volga[i] = sqrt_t * d * d * pdf / vol[i]
    return value, delta, gamma, vega, vanna, volga


@njit(cache=True)  # pragma: no cover - executed as Numba machine code
def _black_all(forward, strike, expiry, vol, cp):
    n = forward.size
    value = np.empty(n)
    delta = np.empty(n)
    gamma = np.zeros(n)
    vega = np.zeros(n)
    vanna = np.zeros(n)
    volga = np.zeros(n)
    for i in range(n):
        sqrt_t = math.sqrt(expiry[i])
        std = vol[i] * sqrt_t
        intrinsic = max(cp * (forward[i] - strike[i]), 0.0)
        if std <= 0.0:
            value[i] = intrinsic
            delta[i] = cp if cp * (forward[i] - strike[i]) > 0.0 else 0.0
            continue
        d1 = (math.log(forward[i] / strike[i]) + 0.5 * vol[i] * vol[i] * expiry[i]) / std
        d2 = d1 - std
        pdf = _pdf(d1)
        value[i] = cp * (forward[i] * _cdf(cp * d1) - strike[i] * _cdf(cp * d2))
        delta[i] = cp * _cdf(cp * d1)
        gamma[i] = pdf / (forward[i] * std)
        vega[i] = forward[i] * pdf * sqrt_t
        vanna[i] = pdf * (sqrt_t - d1 / vol[i])
        volga[i] = vega[i] * d1 * d2 / vol[i]
    return value, delta, gamma, vega, vanna, volga


def _option_inputs(forward, strike, expiry, vol):
    arrays = np.broadcast_arrays(forward, strike, expiry, vol)
    out = tuple(_float_1d(np.asarray(a).reshape(-1)) for a in arrays)
    if np.any(out[2] < 0.0) or np.any(out[3] < 0.0):
        raise ValueError("expiry and vol must be non-negative")
    return arrays[0].shape, out


@register_with(kernel_registry, (KERNEL_BACHELIER, Backend.Numba))
def bachelier_greeks(forward, strike, expiry, vol, is_call: bool = True) -> OptionGreeks:
    shape, args = _option_inputs(forward, strike, expiry, vol)
    result = _bachelier_all(*args, 1.0 if is_call else -1.0)
    return OptionGreeks(*(a.reshape(shape) for a in result))


@register_with(kernel_registry, (KERNEL_BLACK, Backend.Numba))
def black_greeks(forward, strike, expiry, vol, is_call: bool = True) -> OptionGreeks:
    shape, args = _option_inputs(forward, strike, expiry, vol)
    if np.any(args[0] <= 0.0) or np.any(args[1] <= 0.0):
        raise ValueError("Black forward and strike must be positive")
    result = _black_all(*args, 1.0 if is_call else -1.0)
    return OptionGreeks(*(a.reshape(shape) for a in result))


__all__ = ["bachelier_greeks", "black_greeks"]
