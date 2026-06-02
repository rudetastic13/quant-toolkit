"""Rate kernels — the irreducible hard part: ragged observation -> period reduction.

A kernel is pure, total and narrow: it takes only the contiguous arrays it needs to do
the vectorized-hard thing, and nothing else.  No spread, no caps/floors, no margin, no
optional behaviour flags — those are coupon *shaping* concerns handled by the prep layer
(``finance.coupons.rates``) before/after the call.  Keeping the kernel tight is what makes
it cleanly swappable for a numba/rust implementation under the same registry key.

Both kernels turn a flat ``(M,)`` observation grid (every period's daily fixings
concatenated) into a ``(N,)`` per-period rate using ``np.add.reduceat`` over ``offsets`` —
the jaggedness lives in ``offsets``, not in a Python loop.

    obs_rate : (M,)  per-observation simple index rate
    obs_w    : (M,)  per-observation accrual weight (day-count fraction)
    offsets  : (N,)  start index of each period; strictly increasing, offsets[0] == 0
"""
from __future__ import annotations

import numpy as np

from common.registry import register_with
from finance.pricing.engines import engine_registry
from finance.pricing.types import Backend, RATE_COMPOUNDED, RATE_AVERAGED

FloatArray = np.ndarray


def _check_offsets(offsets: np.ndarray, m: int) -> None:
    """Guard the one contract ``reduceat`` cannot enforce itself.

    Empty or non-increasing segments make ``np.add.reduceat`` return garbage silently,
    so this stays in the kernel — it is a correctness guard, not a behaviour switch.
    """
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("offsets must be a non-empty 1-D array")
    if offsets[0] != 0:
        raise ValueError("offsets[0] must be 0")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("offsets must be strictly increasing (every period needs >=1 observation)")
    if offsets[-1] >= m:
        raise ValueError("last offset must be < number of observations")


@register_with(engine_registry, (RATE_COMPOUNDED, Backend.Numpy))
def compounded(obs_rate: FloatArray, obs_w: FloatArray, offsets: np.ndarray) -> FloatArray:
    """Daily-compounded (OIS) per-period rate.

    ``rate_p = (prod_{i in p}(1 + r_i w_i) - 1) / sum_{i in p} w_i``

    Evaluated in log space for stability:
    ``expm1(reduceat(log1p(r*w), offsets)) / reduceat(w, offsets)``.
    """
    obs_rate = np.asarray(obs_rate, dtype=np.float64)
    obs_w = np.asarray(obs_w, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.intp)
    _check_offsets(offsets, obs_rate.size)

    log_g = np.log1p(obs_rate * obs_w)
    growth = np.expm1(np.add.reduceat(log_g, offsets))
    weight = np.add.reduceat(obs_w, offsets)
    return growth / weight


@register_with(engine_registry, (RATE_AVERAGED, Backend.Numpy))
def averaged(obs_rate: FloatArray, obs_w: FloatArray, offsets: np.ndarray) -> FloatArray:
    """Weighted arithmetic average of daily fixings per period.

    ``rate_p = sum_{i in p}(r_i w_i) / sum_{i in p} w_i``
    """
    obs_rate = np.asarray(obs_rate, dtype=np.float64)
    obs_w = np.asarray(obs_w, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.intp)
    _check_offsets(offsets, obs_rate.size)

    return np.add.reduceat(obs_rate * obs_w, offsets) / np.add.reduceat(obs_w, offsets)


__all__ = ["compounded", "averaged"]
