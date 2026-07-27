"""CurveCalibrator — build a curve from calibration instruments + a solver + a target.

This is the "object that calibrates": it wires a list of single-quote calibration
instruments, a solver, and a target curve definition into one residual closure (trial node
values -> per-instrument residuals) and hands that to the solver. The solved node values
are returned as a mathematical ``ZeroCurve``; the input market is never mutated. Callers
explicitly compose that state with index conventions to register a ``YieldCurve``.

Parameterisation is continuously-compounded zero rates at the pillars (``DF = exp(-x·t)``,
``t`` in Act/365 from origin); the first node is pinned at the origin with ``DF = 1``.  The
pillar ladder (dates -> trial rates) is conceptually a line, but ``ZeroCurve``'s
interpolation is the single source of truth, so no separate ``Line`` is materialised.
Helpers reference a single curve name (their forecast curve); a multi-name basis helper is a
multi-curve follow-up that reuses ``MarketContext.with_curve`` the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, YieldCurve, ZeroCurve
from finance.pricing.calibration.instruments import CalibrationInstrument
from finance.pricing.calibration.solvers import Solver, SolverResult
from finance.pricing.types import Backend

FloatArray = np.ndarray
DateArray = np.ndarray

# Interpolators whose nodes are *local* (a node affects only its adjacent intervals), so a
# sequential bootstrap is valid.  Global/spline schemes are excluded.
LOCAL_INTERPOLATORS = frozenset({CurveInterpolator.LogLinearDF, CurveInterpolator.RateLinear})


@dataclass(frozen=True)
class CurveDefinition:
    """Identity and mathematical interpolation of the zero curve being calibrated."""

    currency: str
    index_name: str
    interpolation: CurveInterpolator = CurveInterpolator.LogLinearDF

    @property
    def name(self) -> str:
        return f"{self.currency.upper()}.{self.index_name.upper()}"

    def compose(self, zero_curve: ZeroCurve, market: MarketContext) -> YieldCurve:
        """Build the temporary convention-aware curve needed to price calibration quotes."""
        return YieldCurve.from_registry(
            zero_curve,
            currency=self.currency,
            index_name=self.index_name,
            registry=market.conventions,
        )


@dataclass
class CalibrationResult:
    """The calibrated mathematical curve and solver diagnostics.

    ``zero_curve`` is deliberately mathematical state, not a market registration. The
    caller composes it with ``MarketConventions`` to create a ``YieldCurve``.

    ``jacobian`` (populated only when ``calibrate(..., jacobian=True)``) is the exact
    ``J_ij = ∂impliedᵢ/∂zⱼ`` at the solution, computed by the Numba analytic adjoint by
    default (JAX remains available as an explicit oracle).  It is the
    change-of-variables from pillar (zero-rate) risk to market-quote risk: a partial DV01 to
    the calibration quotes is ``(∂V/∂z) · J⁻¹`` (see ``finance.pricing.risk.autodiff``).
    """

    zero_curve: ZeroCurve
    solver_result: SolverResult
    pillar_dates: DateArray
    residuals: FloatArray
    jacobian: FloatArray | None = None


@dataclass
class CurveCalibrator:
    """Calibrate one curve from a set of single-quote instruments against a base market."""

    instruments: Sequence[CalibrationInstrument]
    solver: Solver
    target: CurveDefinition

    def calibrate(
        self,
        market: MarketContext,
        *,
        jacobian: bool = False,
        jacobian_backend: Backend = Backend.Numba,
    ) -> CalibrationResult:
        """Solve for the target curve.

        ``jacobian=True`` additionally captures ``∂implied/∂z`` at the solution.  The default
        Numba path uses analytic curve weights and the hand-written swap adjoint; pass
        ``jacobian_backend=Backend.Jax`` only for validation against traced autodiff.
        """
        jacobian_backend = Backend(jacobian_backend)
        helpers = list(self.instruments)
        if not helpers:
            raise ValueError("CurveCalibrator requires at least one instrument")

        for h in helpers:
            if h.curve != self.target.name:
                raise ValueError(
                    f"instrument targets curve '{h.curve}' but the calibration target is "
                    f"'{self.target.name}'"
                )

        helpers.sort(key=lambda h: h.pillar_date)
        pillars = np.array([h.pillar_date for h in helpers], dtype="datetime64[D]")
        if np.any(pillars[1:] <= pillars[:-1]):
            raise ValueError(
                "calibration instruments must have strictly increasing pillar dates "
                "(one node per instrument)"
            )

        origin = np.datetime64(market.as_of_date.to_str(), "D")
        if pillars[0] <= origin:
            raise ValueError(f"all pillar dates must be after the curve origin {origin}")

        if self.solver.requires_local_interpolation and self.target.interpolation not in LOCAL_INTERPOLATORS:
            local = ", ".join(i.name for i in LOCAL_INTERPOLATORS)
            raise ValueError(
                f"{type(self.solver).__name__} requires a local interpolator ({local}); "
                f"got {self.target.interpolation.name}"
            )

        # Both exact-Jacobian curve models currently implement local LogLinearDF weights.
        if jacobian and self.target.interpolation is not CurveInterpolator.LogLinearDF:
            raise NotImplementedError(
                f"calibrate(jacobian=True) requires CurveInterpolator.LogLinearDF (the "
                f"adjoint curve model does not support {self.target.interpolation.name} yet); "
                "calibrate without jacobian and use the numpy bump path for risk instead"
            )

        node_dates = np.concatenate([[origin], pillars]).astype("datetime64[D]")
        t = (pillars.astype(np.int64) - origin.astype(np.int64)) / 365.0

        def build_curve(x: FloatArray) -> ZeroCurve:
            dfs = np.concatenate([[1.0], np.exp(-np.asarray(x, dtype=np.float64) * t)])
            return ZeroCurve(node_dates, dfs, self.target.interpolation)

        def residual_fn(x: FloatArray) -> FloatArray:
            zero_curve = build_curve(x)
            trial = market.with_curve(self.target.compose(zero_curve, market))
            return np.array([h.residual(trial) for h in helpers], dtype=np.float64)

        x0 = np.array([h.quote.value for h in helpers], dtype=np.float64)
        sr = self.solver.solve(residual_fn, x0)
        if not sr.converged:
            raise RuntimeError(
                f"calibration did not converge: max|residual|={sr.max_abs_residual:.3e} "
                f"({sr.message})"
            )

        curve = build_curve(sr.x)
        out = market.with_curve(self.target.compose(curve, market))

        residuals = np.array([h.residual(out) for h in helpers], dtype=np.float64)

        jac = None
        if jacobian:
            if jacobian_backend == Backend.Numba:
                from finance.pricing.engines.numba.calibration import calibration_jacobian
            elif jacobian_backend == Backend.Jax:
                from finance.pricing.engines.jax.calibration import calibration_jacobian
            else:
                raise NotImplementedError(
                    f"calibration Jacobian backend {jacobian_backend.name} is not implemented"
                )

            jac = calibration_jacobian(helpers, self.target.name, out, curve)

        return CalibrationResult(
            zero_curve=curve, solver_result=sr,
            pillar_dates=pillars, residuals=residuals, jacobian=jac,
        )


__all__ = ["CurveDefinition", "CalibrationResult", "CurveCalibrator", "LOCAL_INTERPOLATORS"]
