"""Differentiable discount curves for the JAX backend.

A ``ZeroCurve`` is decomposed into two parts:

* **static geometry** — the day-offset grid ``x_nodes`` (origin-inclusive, day 0 pinned) and
  the Act/365 year fractions ``t_nodes`` at the non-origin pillars.  These never carry a
  gradient; they are fixed at compile time.
* **parameters** ``z`` — the continuously-compounded zero rate at each non-origin pillar.
  This is the *exact* vector ``Sensitivities.bumped_curve`` shifts and the calibrator solves
  for (``DF = exp(-z·t)``), so a ``jax.grad`` in ``z`` is directly comparable to the engine's
  key-rate ladder.

``make_logdf`` returns a closure ``logdf(z, x_query)`` that is differentiable in ``z`` and
reproduces ``CurveInterpolator.LogLinearDF`` *including* the engine's flat-forward
extrapolation beyond the last pillar (``jnp.interp`` alone would clamp — wrong forward).
Only the local interpolators the calibrator accepts are modelled here; ``RateLinear`` is a
follow-up (the seam is the ``interpolation`` dispatch below).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax.numpy as jnp

from finance.markets.curves import CurveInterpolator, ZeroCurve

FloatArray = np.ndarray


@dataclass(frozen=True)
class CurveGeometry:
    """Static (gradient-free) geometry of one curve, shared across reprices.

    x_nodes : (N,)  day offsets from origin, origin-inclusive (x_nodes[0] == 0.0).
    t_nodes : (N-1,) Act/365 year fractions at the non-origin pillars.
    z0      : (N-1,) the calibrated zero rates (initial parameter value).
    interpolation : the curve's interpolation method (only LogLinearDF modelled today).
    """

    x_nodes: FloatArray
    t_nodes: FloatArray
    z0: FloatArray
    interpolation: CurveInterpolator


def curve_geometry(curve: ZeroCurve) -> CurveGeometry:
    """Pull the static geometry and the initial zero-rate parameter vector off a ZeroCurve."""
    o = curve.origin.astype(np.int64)
    x_nodes = (curve.node_dates.astype(np.int64) - o).astype(np.float64)  # incl. origin (0.0)
    t_nodes = x_nodes[1:] / 365.0
    z0 = -np.log(curve.node_dfs[1:]) / t_nodes
    return CurveGeometry(x_nodes=x_nodes, t_nodes=t_nodes, z0=z0, interpolation=curve.interpolation)


def make_logdf(geom: CurveGeometry):
    """Return ``logdf(z, x_query) -> ln DF``, differentiable in the pillar zero-rates ``z``.

    Log-linear in ln(DF) (matching ``CurveInterpolator.LogLinearDF``); the origin node is
    pinned at ln DF = 0 and each pillar at ``-zₚ·tₚ``.  Beyond the last pillar the last
    segment's slope is continued (flat-forward extrapolation) — the same thing the numpy
    ``ZeroCurve`` does, so the two engines agree on flows past the terminal pillar.
    """
    if geom.interpolation is not CurveInterpolator.LogLinearDF:
        raise NotImplementedError(
            f"JAX curve only models LogLinearDF today; got {geom.interpolation.name}. "
            "RateLinear is a follow-up (linear in zero rate, not in ln DF)."
        )
    x_nodes = jnp.asarray(geom.x_nodes)
    t_nodes = jnp.asarray(geom.t_nodes)
    x_last = float(geom.x_nodes[-1])
    x_prev = float(geom.x_nodes[-2])
    dx_last = x_last - x_prev

    def logdf(z, x_query):
        logdf_nodes = jnp.concatenate([jnp.zeros(1, dtype=z.dtype), -z * t_nodes])
        interior = jnp.interp(x_query, x_nodes, logdf_nodes)  # clamps beyond the ends
        slope_r = (logdf_nodes[-1] - logdf_nodes[-2]) / dx_last
        right = logdf_nodes[-1] + slope_r * (x_query - x_last)  # flat-forward continuation
        return jnp.where(x_query > x_last, right, interior)

    return logdf


def make_df(geom: CurveGeometry):
    """Return ``df(z, x_query) -> DF`` = ``exp(logdf(z, x_query))``."""
    logdf = make_logdf(geom)

    def df(z, x_query):
        return jnp.exp(logdf(z, x_query))

    return df


__all__ = ["CurveGeometry", "curve_geometry", "make_logdf", "make_df"]
