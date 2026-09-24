"""JAX hook for curve calibration — the exact Jacobian ``J = ∂implied/∂z``.

The calibrator solves ``implied(z) = quote`` for the pillar zero-rates ``z``.  ``J_ij =
∂impliedᵢ/∂zⱼ`` is the sensitivity of each calibration instrument's *quoted measure* to each
pillar.  The numpy ``GlobalSolver`` finite-differences this matrix to drive its solve; here
we get it **exactly** in one ``jax.jacobian`` pass at the solution.

Why it matters: ``J`` is the change-of-variables between pillar (zero-rate) risk and
market-quote risk.  By the implicit function theorem on ``implied(z*(q)) = q``, ``dz/dq =
J⁻¹``, so an instrument's **partial DV01 to the calibration quotes** is ``(∂V/∂z) · J⁻¹`` —
the standard "zero-rate sensitivity → par-quote sensitivity" transform (Strata's
``CurveParameterSensitivity → MarketQuoteSensitivity``).  See ``risk.autodiff``.

Each helper contributes its measure as a pure ``jnp`` function of ``z`` (single curve — the
target curve is used for both projection and discount):

* deposit / FRA  ->  simple money-market rate ``(DF(eff)/DF(mat) - 1) / τ``  (τ static)
* swap           ->  par rate ``-(Σ float leg PV) / (Σ fixed leg PV)`` via a ``JaxProgram``
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from finance.dates import period_fractions
from finance.markets.curves import YieldCurve
from finance.pricing.calibration.instruments import DepositHelper, FraHelper, SwapHelper
from finance.pricing.engines.jax.curves import curve_geometry, make_df
from finance.pricing.engines.jax.program import JaxProgram


def _off(date_np: np.datetime64, origin: np.int64) -> float:
    return float(np.datetime64(date_np, "D").astype(np.int64) - origin)


def _money_market_fn(helper, df, origin):
    """Closure z -> simple rate for a deposit/FRA helper (τ is static, only DF moves)."""
    inst = getattr(helper, "deposit", None) or getattr(helper, "fra", None)
    eff = np.datetime64(inst.effective.to_numpy(), "D")
    mat = np.datetime64(inst.maturity.to_numpy(), "D")
    tau = float(period_fractions(inst.day_count_method, np.array([eff]), np.array([mat]))[0])
    eff_off, mat_off = _off(eff, origin), _off(mat, origin)

    def implied(z):
        return (df(z, eff_off) / df(z, mat_off) - 1.0) / tau

    return implied


def _swap_fn(helper: SwapHelper, market, curve_name: str):
    """Closure z -> par swap rate, repricing the helper's compiled swap in JAX.

    The free variable ``z`` is the *target* curve being calibrated; every other curve the
    helper references (e.g. the SOFR discount curve of a Fed-Funds basis helper) is held
    fixed at its already-calibrated value from ``market``.
    """
    jp = JaxProgram(helper._program.inputs, market)
    disc0, proj0 = jp.params_from_market(market)
    fixed_idx = jnp.asarray(helper._fixed_idx)
    float_idx = jnp.asarray(helper._float_idx)

    def implied(z):
        proj = {**proj0, curve_name: z}   # target curve projects with the trial z
        disc = {**disc0, curve_name: z}   # ... and discounts with it too if it is the funding curve
        leg_pv = jp.leg_pv(disc, proj)
        return -jnp.sum(leg_pv[float_idx]) / jnp.sum(leg_pv[fixed_idx])

    return implied


def implied_vector_fn(helpers, target_name: str, market, curve: YieldCurve):
    """Build ``z -> (implied measure per helper)`` as one differentiable ``jnp`` function."""
    geom = curve_geometry(curve.zero_curve)
    df = make_df(geom)
    origin = curve.origin.astype(np.int64)

    fns = []
    for h in helpers:
        if isinstance(h, (DepositHelper, FraHelper)):
            fns.append(_money_market_fn(h, df, origin))
        elif isinstance(h, SwapHelper):
            fns.append(_swap_fn(h, market, target_name))
        else:  # pragma: no cover - future helper kinds (bond yield, ...)
            raise NotImplementedError(f"no JAX implied for helper type {type(h).__name__}")

    def implied_vector(z):
        return jnp.stack([f(z) for f in fns])

    return implied_vector, jnp.asarray(geom.z0)


def calibration_jacobian(helpers, target_name: str, market, curve: YieldCurve) -> np.ndarray:
    """Exact ``J_ij = ∂impliedᵢ/∂zⱼ`` at the calibrated curve (shape ``(n_helpers, n_pillars)``)."""
    implied_vector, z0 = implied_vector_fn(helpers, target_name, market, curve)
    return np.asarray(jax.jacobian(implied_vector)(z0))


__all__ = ["implied_vector_fn", "calibration_jacobian"]
