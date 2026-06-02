"""Coupon rate shaping — the prep layer that feeds the tight rate kernels.

This is the vectorized replacement for the per-period loops in ``coupons.calculators``.
It is the *interface* layer, so carrying the rich coupon parameters (spread, margin
treatment, index_floor, cap, floor) here is appropriate — the kernels stay tight and this
layer prepares their inputs and shapes their outputs.

Order of operations (matches the legacy ``calculators`` semantics):

    1. index_floor   -> applied to each daily fixing      (INPUT prep, per-observation)
    2. spread (incl.) -> added to each fixing before reduction, when margin is Inclusive
    3. kernel         -> compounded/averaged segmented reduction (the hard part)
    4. spread (excl.) -> added to the period rate, when margin is Exclusive
    5. floor, cap     -> applied to the FINAL period coupon (OUTPUT prep)

Cap/floor convention
--------------------
``cap``/``floor`` are applied to the *realized period coupon* (the displayed all-in rate),
NOT on a daily running basis.  If the running compounded rate breaches the level
mid-period but recovers by period end, that intra-period breach is **ignored** — this is
the standard market treatment for vanilla capped/floored RFR coupons.  A genuine
daily-running cap/floor is a different, path-dependent product and would not use these
reductions.  ``index_floor`` is the separate per-fixing flooring that *does* change the
compounding.
"""
from __future__ import annotations

import numpy as np

from finance.instruments.enums import MarginTreatment
from finance.pricing.engines.numpy.rates import compounded, averaged

FloatArray = np.ndarray


def fixed_coupon(coupon_rate: float, n_periods: int) -> FloatArray:
    """Same fixed rate for every period."""
    return np.full(int(n_periods), float(coupon_rate), dtype=np.float64)


def floating_coupon(
    index_rate: FloatArray,
    *,
    spread: float = 0.0,
    index_floor: float | None = None,
    cap: float | None = None,
    floor: float | None = None,
) -> FloatArray:
    """Simple single-fixing (IBOR-style) coupon: index_floor -> +spread -> floor -> cap."""
    out = np.array(index_rate, dtype=np.float64, copy=True)
    if index_floor is not None:
        np.maximum(out, index_floor, out=out)
    out += spread
    if floor is not None:
        np.maximum(out, floor, out=out)
    if cap is not None:
        np.minimum(out, cap, out=out)
    return out


def _shape_reduced(
    rate: FloatArray,
    spread: float,
    margin: MarginTreatment,
    cap: float | None,
    floor: float | None,
) -> FloatArray:
    """Output prep shared by compounded/averaged: exclusive spread, then floor/cap."""
    if spread and margin == MarginTreatment.Exclusive:
        rate = rate + spread
    if floor is not None:
        np.maximum(rate, floor, out=rate)
    if cap is not None:
        np.minimum(rate, cap, out=rate)
    return rate


def _prep_obs(
    obs_rate: FloatArray,
    spread: float,
    margin: MarginTreatment,
    index_floor: float | None,
) -> FloatArray:
    """Input prep shared by compounded/averaged: per-fixing index_floor, inclusive spread."""
    r = np.array(obs_rate, dtype=np.float64, copy=True)
    if index_floor is not None:
        np.maximum(r, index_floor, out=r)
    if spread and margin == MarginTreatment.Inclusive:
        r += spread
    return r


def compounded_coupon(
    obs_rate: FloatArray,
    obs_w: FloatArray,
    offsets: np.ndarray,
    *,
    spread: float = 0.0,
    index_floor: float | None = None,
    cap: float | None = None,
    floor: float | None = None,
    margin: MarginTreatment = MarginTreatment.Inclusive,
) -> FloatArray:
    """Compounded (OIS) coupon with shaping around the tight ``compounded`` kernel."""
    r = _prep_obs(obs_rate, spread, margin, index_floor)
    rate = compounded(r, obs_w, offsets)
    return _shape_reduced(rate, spread, margin, cap, floor)


def averaged_coupon(
    obs_rate: FloatArray,
    obs_w: FloatArray,
    offsets: np.ndarray,
    *,
    spread: float = 0.0,
    index_floor: float | None = None,
    cap: float | None = None,
    floor: float | None = None,
    margin: MarginTreatment = MarginTreatment.Inclusive,
) -> FloatArray:
    """Arithmetic-averaged coupon with shaping around the tight ``averaged`` kernel."""
    r = _prep_obs(obs_rate, spread, margin, index_floor)
    rate = averaged(r, obs_w, offsets)
    return _shape_reduced(rate, spread, margin, cap, floor)


__all__ = [
    "fixed_coupon",
    "floating_coupon",
    "compounded_coupon",
    "averaged_coupon",
]
