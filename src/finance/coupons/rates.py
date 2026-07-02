"""Coupon rate shaping — scalar/None convenience API over ``pricing.kernels.shaping``.

This is the per-leg interface used by the reference coupon calculators (and tests): rich
scalar parameters (``spread``, ``MarginTreatment``, ``index_floor``/``cap``/``floor`` with
``None`` meaning "no bound").  The shaping algebra itself lives in
``finance.pricing.kernels.shaping`` — the single source shared with the numpy and JAX
engines, which consume it in the columnar sentinel encoding (``±inf`` bounds, boolean
margin masks).  This module only converts ``None``/enums to that encoding and delegates;
see ``shaping``'s docstring for the order of operations and the cap/floor convention.
"""
from __future__ import annotations

import numpy as np

from finance.instruments.enums import MarginTreatment
from finance.pricing.engines.numpy.rates import compounded, averaged
from finance.pricing.kernels.inputs import NO_CAP, NO_FLOOR
from finance.pricing.kernels.shaping import prep_obs, shape_float, shape_period

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
    return shape_float(
        np.asarray(index_rate, dtype=np.float64),
        NO_FLOOR if index_floor is None else index_floor,
        spread,
        NO_FLOOR if floor is None else floor,
        NO_CAP if cap is None else cap,
    )


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
    r = _prep(obs_rate, spread, margin, index_floor)
    rate = compounded(r, obs_w, offsets)
    return _shape(rate, spread, margin, cap, floor)


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
    r = _prep(obs_rate, spread, margin, index_floor)
    rate = averaged(r, obs_w, offsets)
    return _shape(rate, spread, margin, cap, floor)


def _prep(obs_rate: FloatArray, spread: float, margin: MarginTreatment, index_floor: float | None) -> FloatArray:
    return prep_obs(
        np.asarray(obs_rate, dtype=np.float64),
        NO_FLOOR if index_floor is None else index_floor,
        spread,
        margin == MarginTreatment.Inclusive,
    )


def _shape(rate: FloatArray, spread: float, margin: MarginTreatment, cap: float | None, floor: float | None) -> FloatArray:
    return shape_period(
        rate,
        spread,
        margin == MarginTreatment.Exclusive,
        NO_FLOOR if floor is None else floor,
        NO_CAP if cap is None else cap,
    )


__all__ = [
    "fixed_coupon",
    "floating_coupon",
    "compounded_coupon",
    "averaged_coupon",
]
