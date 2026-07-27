"""Small Numba kernels registered against the backend-neutral engine ABI.

The fused :class:`~finance.pricing.engines.numba.program.NumbaProgram` is the production
pricing path.  These narrow kernels keep the public registry complete and are useful when a
caller already has projected rates or cashflows.  Numba specializes them on dtype/layout,
not array length, so one compiled signature serves every portfolio shape.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from common.registry import register_with
from finance.pricing.engines import engine_registry
from finance.pricing.engines.numpy.option import OptionGreeks
from finance.pricing.types import (
    Backend,
    KERNEL_BACHELIER,
    KERNEL_BLACK,
    KERNEL_DCF,
    RATE_AVERAGED,
    RATE_COMPOUNDED,
)

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)


def _float_1d(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _int_1d(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def _check_segments(obs_rate, obs_w, offsets) -> None:
    if obs_rate.shape != obs_w.shape:
        raise ValueError("obs_rate and obs_w must share shape")
    if offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0:
        raise ValueError("offsets must be non-empty, 1-D, and start at zero")
    if np.any(np.diff(offsets) <= 0) or offsets[-1] >= obs_rate.size:
        raise ValueError("offsets must be strictly increasing and inside the observation array")


@njit(cache=True)  # pragma: no cover - executed as Numba machine code
def _dcf(cash, df, sign, offsets):
    out = np.zeros(offsets.size, dtype=np.float64)
    for row in range(offsets.size):
        end = offsets[row + 1] if row + 1 < offsets.size else cash.size
        total = 0.0
        for i in range(offsets[row], end):
            total += cash[i] * df[i] * sign[i]
        out[row] = total
    return out


@register_with(engine_registry, (KERNEL_DCF, Backend.Numba))
def dcf(cash, df, sign, row_offsets):
    """Present value per row using the same inputs as the NumPy DCF kernel."""
    cash = _float_1d(cash)
    df = _float_1d(df)
    sign = _float_1d(sign)
    offsets = _int_1d(row_offsets)
    if not (cash.shape == df.shape == sign.shape):
        raise ValueError("cash, df, sign must share shape")
    if offsets.size and offsets[0] != 0:
        raise ValueError("row_offsets[0] must be 0")
    return _dcf(cash, df, sign, offsets)


@njit(cache=True)  # pragma: no cover - executed as Numba machine code
def _compounded(obs_rate, obs_w, offsets):
    out = np.empty(offsets.size, dtype=np.float64)
    for p in range(offsets.size):
        end = offsets[p + 1] if p + 1 < offsets.size else obs_rate.size
        log_growth = 0.0
        weight = 0.0
        for i in range(offsets[p], end):
            log_growth += math.log1p(obs_rate[i] * obs_w[i])
            weight += obs_w[i]
        out[p] = math.expm1(log_growth) / weight
    return out


@register_with(engine_registry, (RATE_COMPOUNDED, Backend.Numba))
def compounded(obs_rate, obs_w, offsets):
    """Segmented daily compounding, shape-polymorphic over the flat observation grid."""
    obs_rate = _float_1d(obs_rate)
    obs_w = _float_1d(obs_w)
    offsets = _int_1d(offsets)
    _check_segments(obs_rate, obs_w, offsets)
    return _compounded(obs_rate, obs_w, offsets)


@njit(cache=True)  # pragma: no cover - executed as Numba machine code
def _averaged(obs_rate, obs_w, offsets):
    out = np.empty(offsets.size, dtype=np.float64)
    for p in range(offsets.size):
        end = offsets[p + 1] if p + 1 < offsets.size else obs_rate.size
        weighted = 0.0
        weight = 0.0
        for i in range(offsets[p], end):
            weighted += obs_rate[i] * obs_w[i]
            weight += obs_w[i]
        out[p] = weighted / weight
    return out


@register_with(engine_registry, (RATE_AVERAGED, Backend.Numba))
def averaged(obs_rate, obs_w, offsets):
    """Segmented arithmetic averaging over a flat observation grid."""
    obs_rate = _float_1d(obs_rate)
    obs_w = _float_1d(obs_w)
    offsets = _int_1d(offsets)
    _check_segments(obs_rate, obs_w, offsets)
    return _averaged(obs_rate, obs_w, offsets)


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


def bachelier_greeks(forward, strike, expiry, vol, is_call: bool = True) -> OptionGreeks:
    shape, args = _option_inputs(forward, strike, expiry, vol)
    result = _bachelier_all(*args, 1.0 if is_call else -1.0)
    return OptionGreeks(*(a.reshape(shape) for a in result))


def black_greeks(forward, strike, expiry, vol, is_call: bool = True) -> OptionGreeks:
    shape, args = _option_inputs(forward, strike, expiry, vol)
    if np.any(args[0] <= 0.0) or np.any(args[1] <= 0.0):
        raise ValueError("Black forward and strike must be positive")
    result = _black_all(*args, 1.0 if is_call else -1.0)
    return OptionGreeks(*(a.reshape(shape) for a in result))


@register_with(engine_registry, (KERNEL_BACHELIER, Backend.Numba))
def bachelier(forward, strike, expiry, vol, is_call: bool = True):
    return bachelier_greeks(forward, strike, expiry, vol, is_call).value


@register_with(engine_registry, (KERNEL_BLACK, Backend.Numba))
def black(forward, strike, expiry, vol, is_call: bool = True):
    return black_greeks(forward, strike, expiry, vol, is_call).value


__all__ = [
    "dcf",
    "compounded",
    "averaged",
    "bachelier",
    "black",
    "bachelier_greeks",
    "black_greeks",
]
