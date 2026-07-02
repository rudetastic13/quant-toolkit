"""Coupon rate shaping — the single source of the index_floor/spread/floor/cap algebra.

Both engines (numpy ``compiler.reprice`` and the JAX ``JaxProgram``) and the reference
coupon calculators shape rates identically; this module is the one implementation.  The
functions are ``xp``-parameterized (pass ``xp=jnp`` on the JAX path) and jit-safe: they use
only ``maximum``/``minimum``/``where`` on arrays, no Python branching on values.

Inputs follow the columnar sentinel encoding of ``KernelInputs`` (branchless shaping):
``cap = +inf`` (``NO_CAP``) means uncapped, ``floor``/``index_floor = -inf`` (``NO_FLOOR``)
means none, and margin treatment arrives as a boolean mask, not an enum.

Order of operations:

    1. index_floor    -> applied to each fixing                (INPUT prep, per-observation)
    2. spread (incl.) -> added to each fixing before reduction, when margin is Inclusive
    3. kernel         -> compounded/averaged segmented reduction (the hard part)
    4. spread (excl.) -> added to the period rate, when margin is Exclusive
    5. floor, cap     -> applied to the FINAL period coupon    (OUTPUT prep)

For a single-fixing float there is no reduction, so the whole chain collapses to
``shape_float``: index_floor -> +spread -> floor -> cap.

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


def shape_float(idx, index_floor, spread, floor, cap, xp=np):
    """Single-fixing (IBOR-style) coupon shaping: index_floor -> +spread -> floor -> cap."""
    idx = xp.maximum(idx, index_floor)
    idx = idx + spread
    idx = xp.maximum(idx, floor)
    return xp.minimum(idx, cap)


def prep_obs(obs_rate, index_floor, spread, incl_mask, xp=np):
    """Input prep for compounded/averaged: per-fixing index_floor, then inclusive spread."""
    r = xp.maximum(obs_rate, index_floor)
    return r + xp.where(incl_mask, spread, 0.0)


def shape_period(period_rate, spread, excl_mask, floor, cap, xp=np):
    """Output prep for compounded/averaged: exclusive spread, then floor/cap on the final coupon."""
    r = period_rate + xp.where(excl_mask, spread, 0.0)
    r = xp.maximum(r, floor)
    return xp.minimum(r, cap)


__all__ = ["shape_float", "prep_obs", "shape_period"]
