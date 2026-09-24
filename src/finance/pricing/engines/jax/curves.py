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

from finance.markets.curves import ZeroCurve

FloatArray = np.ndarray


@dataclass(frozen=True)
class CurveGeometry:
    """Static (gradient-free) geometry of one curve, shared across reprices.

    x_nodes : (N,)  day offsets from origin, origin-inclusive (x_nodes[0] == 0.0).
    t_nodes : (N-1,) Act/365 year fractions at the non-origin pillars.
    z0      : (N-1,) the calibrated zero rates (initial parameter value).
    log_linear : whether the curve is log-linear in DF (the only scheme modelled today).
    description : ``repr`` of the source curve, for error messages.
    """

    x_nodes: FloatArray
    t_nodes: FloatArray
    z0: FloatArray
    log_linear: bool
    description: str


def curve_geometry(curve: ZeroCurve) -> CurveGeometry:
    """Pull the static geometry and the initial zero-rate parameter vector off a ZeroCurve."""
    x_nodes = curve.x  # incl. origin (0.0)
    return CurveGeometry(
        x_nodes=x_nodes,
        t_nodes=x_nodes[1:] / 365.0,
        z0=curve.node_zero_rates,
        log_linear=curve.is_log_linear,
        description=repr(curve),
    )


def make_logdf(geom: CurveGeometry):
    """Return ``logdf(z, x_query) -> ln DF``, differentiable in the pillar zero-rates ``z``.

    Log-linear in ln(DF) (matching ``CurveInterpolator.LogLinearDF``); the origin node is
    pinned at ln DF = 0 and each pillar at ``-zₚ·tₚ``.  Beyond the last pillar the last
    segment's slope is continued (flat-forward extrapolation) — the same thing the numpy
    ``ZeroCurve`` does, so the two engines agree on flows past the terminal pillar.
    """
    if not geom.log_linear:
        raise NotImplementedError(
            f"JAX curve only models log-linear DF today; got {geom.description}. "
            "Linear in zero rate is a follow-up (linear in rate, not in ln DF)."
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
