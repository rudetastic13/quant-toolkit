"""Autodiff risk — key rates, gamma, per-cashflow index/funding deltas, and partial DV01s.

The bump-and-reprice ``Sensitivities`` layer costs ``2·P`` repricings for a P-pillar ladder
and an ``O(P²)`` grid for second order.  This layer prices the *same* compiled program through
the JAX backend and reads risk straight off the derivatives:

* ``zero_ladder``      — ∂V/∂z, the per-pillar key-rate ladder (one reverse pass), split into
  ``index_ladder`` (projection-curve risk) and ``funding_ladder`` (discount-curve risk).
* ``cashflow_index_delta`` / ``cashflow_funding_delta`` — the **per-cashflow** decomposition:
  for each flow, its sensitivity to each pillar of the projection vs the funding curve.  A
  fixed flow has zero index delta and pure funding delta; a floating flow carries both.
* ``gamma``            — the full pillar Hessian ∂²V/∂z∂z (cross-convexity the bump layer
  does not produce at all).
* ``partial_dv01``     — risk to the **calibration quotes** (par DV01), via the calibration
  Jacobian: ``(∂V/∂z) · J⁻¹``.

All ladders are returned as ``(n_instruments, P)`` matrices scaled to a +1bp move, matching
``KeyRateLadder.krd``.  Cashflow deltas are ``(n_flows, P)``.
"""
from __future__ import annotations

import numpy as np

import jax

from finance.markets.context import MarketContext
from finance.pricing.engines.jax.program import JaxProgram
from finance.pricing.pricers.base import PricingProgram

BP = 1e-4
FloatArray = np.ndarray


class AutodiffRisk:
    """Gradient-based risk for a compiled ``PricingProgram`` against a ``MarketContext``.

    Geometry is frozen from ``market`` at construction; risk is taken w.r.t. the per-curve
    zero-rate vectors.  ``bp`` scales results to a +1bp move (default 1bp).

    **Everything is JIT-compiled.**  Each derivative transform (jacobian / hessian) is wrapped
    in ``jax.jit`` and cached per ``(kind, curve_name)``, so the XLA program is built once on
    the first call and every subsequent call is a compiled execution — no re-tracing.  This is
    what makes the steady-state cost competitive with (and, on full ladders, far below) the
    numpy bump-and-reprice path; the first call pays a one-off compile.
    """

    def __init__(self, program: PricingProgram, market: MarketContext, bp: float = BP):
        self.jp = JaxProgram(program.inputs, market)
        self.bp = bp
        self.disc0, self.proj0 = self.jp.params_from_market(market)
        self.curve_names = self.jp.curve_names
        self._jit_cache: dict = {}

    # -- jit cache ----------------------------------------------------------------------
    def _cached(self, key, build):
        fn = self._jit_cache.get(key)
        if fn is None:
            fn = build()
            self._jit_cache[key] = fn
        return fn

    # -- first order: instrument ladders ------------------------------------------------
    def index_ladder(self, curve_name: str) -> FloatArray:
        """(n_inst, P) projection-curve (index) key-rate ladder, per +1bp."""
        disc0, proj0 = self.disc0, self.proj0

        def build():
            return jax.jit(jax.jacfwd(
                lambda z: self.jp.instrument_pv(disc0, {**proj0, curve_name: z})
            ))

        jac = self._cached(("index", curve_name), build)(self.proj0[curve_name])
        return np.asarray(jac) * self.bp

    def funding_ladder(self, curve_name: str) -> FloatArray:
        """(n_inst, P) discount-curve (funding) key-rate ladder, per +1bp."""
        disc0, proj0 = self.disc0, self.proj0

        def build():
            return jax.jit(jax.jacfwd(
                lambda z: self.jp.instrument_pv({**disc0, curve_name: z}, proj0)
            ))

        jac = self._cached(("funding", curve_name), build)(self.disc0[curve_name])
        return np.asarray(jac) * self.bp

    def zero_ladder(self, curve_name: str) -> FloatArray:
        """(n_inst, P) total key-rate ladder = index + funding, per +1bp."""
        return self.index_ladder(curve_name) + self.funding_ladder(curve_name)

    # -- first order: per-cashflow decomposition ----------------------------------------
    def cashflow_index_delta(self, curve_name: str) -> FloatArray:
        """(n_flows, P) each cashflow's sensitivity to the projection curve, per +1bp."""
        disc0, proj0 = self.disc0, self.proj0

        def build():
            return jax.jit(jax.jacfwd(
                lambda z: self.jp.flow_pv(disc0, {**proj0, curve_name: z})
            ))

        jac = self._cached(("cf_index", curve_name), build)(self.proj0[curve_name])
        return np.asarray(jac) * self.bp

    def cashflow_funding_delta(self, curve_name: str) -> FloatArray:
        """(n_flows, P) each cashflow's sensitivity to the funding curve, per +1bp."""
        disc0, proj0 = self.disc0, self.proj0

        def build():
            return jax.jit(jax.jacfwd(
                lambda z: self.jp.flow_pv({**disc0, curve_name: z}, proj0)
            ))

        jac = self._cached(("cf_funding", curve_name), build)(self.disc0[curve_name])
        return np.asarray(jac) * self.bp

    # -- second order -------------------------------------------------------------------
    def _total_z_pv(self, curve_name: str):
        """JIT'd scalar ``PV(z)`` where ``z`` drives both projection and discount of a curve."""
        disc0, proj0 = self.disc0, self.proj0

        def build():
            return jax.jit(
                lambda z: self.jp.pv({**disc0, curve_name: z}, {**proj0, curve_name: z})
            )

        return self._cached(("pv", curve_name), build)

    def gamma(self, curve_name: str) -> FloatArray:
        """(P, P) total-PV pillar gamma ∂²V/∂z∂z for a move of the whole curve, per (1bp)².

        The curve drives projection and discount together (a real curve move shifts both),
        so this is the total convexity; ``0.5·Hₚ_q·(1bp)²`` is the per-pillar-pair P&L.
        """
        disc0, proj0 = self.disc0, self.proj0

        def build():
            return jax.jit(jax.hessian(
                lambda z: self.jp.pv({**disc0, curve_name: z}, {**proj0, curve_name: z})
            ))

        H = np.asarray(self._cached(("gamma", curve_name), build)(self.disc0[curve_name]))
        return 0.5 * H * self.bp**2

    # -- market-quote partial DV01 (the calibration-Jacobian transform) -----------------
    def _unit_zero_jacobian(self, curve_name: str) -> FloatArray:
        """(n_inst, P) ∂V/∂z (per unit rate) for a total move of ``curve_name``."""
        disc0, proj0 = self.disc0, self.proj0

        def build():
            return jax.jit(jax.jacfwd(
                lambda z: self.jp.instrument_pv({**disc0, curve_name: z}, {**proj0, curve_name: z})
            ))

        return np.asarray(self._cached(("unit", curve_name), build)(self.disc0[curve_name]))

    def partial_dv01(self, curve_name: str, jacobian: FloatArray) -> FloatArray:
        """(n_inst, n_quotes) risk to each calibration quote, per +1bp quote move.

        ``jacobian`` is the calibration ``J = ∂implied/∂z`` (from
        ``CurveCalibrator.calibrate(..., jacobian=True)``).  By the implicit function theorem
        ``dz/dq = J⁻¹``, so par-quote risk is ``(∂V/∂z) · J⁻¹``.  Row sums recover the
        parallel DV01 (a parallel quote shift ≈ a parallel zero shift for par instruments).
        """
        g = self._unit_zero_jacobian(curve_name)          # (n_inst, P), per unit rate
        return (g @ np.linalg.inv(np.asarray(jacobian))) * self.bp


__all__ = ["AutodiffRisk", "BP"]
