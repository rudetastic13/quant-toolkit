"""Vectorized Black and Bachelier option kernels.

Values are undiscounted and per unit annuity.  A swaption pricer multiplies the result by
its cash annuity and notional; an option on a future uses its contract multiplier instead.
The kernels broadcast all numeric inputs and use an explicit call/put sign, making them a
small, backend-neutral ABI shared with the Numba implementations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from common.registry import register_with
from finance.pricing.engines import engine_registry
from finance.pricing.types import Backend, KERNEL_BACHELIER, KERNEL_BLACK

FloatArray = np.ndarray
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class OptionGreeks:
    """Value and derivatives with respect to forward and quoted volatility."""

    value: FloatArray
    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    vanna: FloatArray
    volga: FloatArray


def _inputs(forward, strike, expiry, vol):
    return np.broadcast_arrays(
        np.asarray(forward, dtype=np.float64),
        np.asarray(strike, dtype=np.float64),
        np.asarray(expiry, dtype=np.float64),
        np.asarray(vol, dtype=np.float64),
    )


def _normal_pdf(x):
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _phi(is_call: bool) -> float:
    return 1.0 if is_call else -1.0


@register_with(engine_registry, (KERNEL_BACHELIER, Backend.Numpy))
def bachelier(forward: FloatArray, strike: FloatArray, expiry: FloatArray, vol: FloatArray,
              is_call: bool = True) -> FloatArray:
    """Normal-vol (Bachelier) option value per unit annuity."""
    return bachelier_greeks(forward, strike, expiry, vol, is_call).value


@register_with(engine_registry, (KERNEL_BLACK, Backend.Numpy))
def black(forward: FloatArray, strike: FloatArray, expiry: FloatArray, vol: FloatArray,
          is_call: bool = True) -> FloatArray:
    """Lognormal-vol (Black-76) option value per unit annuity."""
    return black_greeks(forward, strike, expiry, vol, is_call).value


def bachelier_greeks(forward, strike, expiry, vol, is_call: bool = True) -> OptionGreeks:
    """Bachelier value, delta, gamma, vega, vanna, and volga.

    At expiry or zero volatility, value becomes intrinsic and delta uses a deterministic
    left/right subgradient (zero exactly at the strike).  Singular second-order greeks are
    reported as zero on that boundary.
    """
    f, k, t, sigma = _inputs(forward, strike, expiry, vol)
    if np.any(t < 0.0) or np.any(sigma < 0.0):
        raise ValueError("expiry and vol must be non-negative")

    cp = _phi(is_call)
    std = sigma * np.sqrt(t)
    live = std > 0.0
    d = np.zeros_like(std)
    np.divide(f - k, std, out=d, where=live)
    pdf = _normal_pdf(d)

    intrinsic = np.maximum(cp * (f - k), 0.0)
    value = np.where(live, cp * (f - k) * ndtr(cp * d) + std * pdf, intrinsic)
    delta_boundary = np.where(cp * (f - k) > 0.0, cp, 0.0)
    delta = np.where(live, cp * ndtr(cp * d), delta_boundary)
    gamma = np.zeros_like(std)
    np.divide(pdf, std, out=gamma, where=live)
    vega = np.where(live, np.sqrt(t) * pdf, 0.0)
    vanna = np.zeros_like(std)
    np.divide(-d * pdf, sigma, out=vanna, where=live)
    volga = np.zeros_like(std)
    np.divide(np.sqrt(t) * d * d * pdf, sigma, out=volga, where=live)
    return OptionGreeks(value, delta, gamma, vega, vanna, volga)


def black_greeks(forward, strike, expiry, vol, is_call: bool = True) -> OptionGreeks:
    """Black-76 value, delta, gamma, vega, vanna, and volga."""
    f, k, t, sigma = _inputs(forward, strike, expiry, vol)
    if np.any(t < 0.0) or np.any(sigma < 0.0):
        raise ValueError("expiry and vol must be non-negative")
    if np.any(f <= 0.0) or np.any(k <= 0.0):
        raise ValueError("Black forward and strike must be positive")

    cp = _phi(is_call)
    sqrt_t = np.sqrt(t)
    std = sigma * sqrt_t
    live = std > 0.0
    d1 = np.zeros_like(std)
    np.divide(np.log(f / k) + 0.5 * sigma * sigma * t, std, out=d1, where=live)
    d2 = d1 - std
    pdf = _normal_pdf(d1)

    intrinsic = np.maximum(cp * (f - k), 0.0)
    value = np.where(live, cp * (f * ndtr(cp * d1) - k * ndtr(cp * d2)), intrinsic)
    delta_boundary = np.where(cp * (f - k) > 0.0, cp, 0.0)
    delta = np.where(live, cp * ndtr(cp * d1), delta_boundary)
    gamma = np.zeros_like(std)
    np.divide(pdf, f * std, out=gamma, where=live)
    vega = np.where(live, f * pdf * sqrt_t, 0.0)
    vanna = np.zeros_like(std)
    np.divide(d1, sigma, out=vanna, where=live)
    vanna = np.where(live, pdf * (sqrt_t - vanna), 0.0)
    volga = np.zeros_like(std)
    np.divide(vega * d1 * d2, sigma, out=volga, where=live)
    return OptionGreeks(value, delta, gamma, vega, vanna, volga)


__all__ = ["OptionGreeks", "bachelier", "black", "bachelier_greeks", "black_greeks"]
