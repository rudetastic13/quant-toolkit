"""CurveCalibrator — build a curve from calibration instruments + a solver + a target.

This is the "object that calibrates": it wires a list of single-quote calibration
instruments, a solver, and a target curve definition into one residual closure (trial node
values -> per-instrument residuals) and hands that to the solver.  The solved node values
become a ``ZeroCurve`` bound into a *fresh* ``MarketContext`` (the input market is never
mutated), optionally with a flat vol shim.

Parameterisation is continuously-compounded zero rates at the pillars (``DF = exp(-x·t)``,
``t`` in Act/365 from origin); the first node is pinned at the origin with ``DF = 1``.  The
pillar ladder (dates -> trial rates) is conceptually a line, but ``ZeroCurve``'s
interpolation is the single source of truth, so no separate ``Line`` is materialised.
Helpers reference a single curve name (their forecast curve); a multi-name basis helper is a
multi-curve follow-up that reuses ``MarketContext.with_curve`` the same way.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, ZeroCurve
from finance.markets.vols import FlatVolSurface, VolNamespace
from finance.pricing.calibration.instruments import CalibrationInstrument
from finance.pricing.calibration.solvers import Solver, SolverResult

FloatArray = np.ndarray
DateArray = np.ndarray

# Interpolators whose nodes are *local* (a node affects only its adjacent intervals), so a
# sequential bootstrap is valid.  Global/spline schemes are excluded.
LOCAL_INTERPOLATORS = frozenset({CurveInterpolator.LogLinearDF, CurveInterpolator.RateLinear})


@dataclass(frozen=True)
class CurveDefinition:
    """What to build: a curve name and how it interpolates.  Node dates come from the
    calibration instruments' pillar dates."""

    name: str
    interpolation: CurveInterpolator = CurveInterpolator.LogLinearDF


@dataclass
class CalibrationResult:
    """The calibrated curve plus the market it's bound into and solver diagnostics."""

    curve: ZeroCurve
    market: MarketContext
    solver_result: SolverResult
    pillar_dates: DateArray
    residuals: FloatArray


@dataclass
class CurveCalibrator:
    """Calibrate one curve from a set of single-quote instruments against a base market."""

    instruments: Sequence[CalibrationInstrument]
    solver: Solver
    target: CurveDefinition
    vol_shim: float | None = None

    def calibrate(self, market: MarketContext) -> CalibrationResult:
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

        node_dates = np.concatenate([[origin], pillars]).astype("datetime64[D]")
        t = (pillars.astype(np.int64) - origin.astype(np.int64)) / 365.0

        def build_curve(x: FloatArray) -> ZeroCurve:
            dfs = np.concatenate([[1.0], np.exp(-np.asarray(x, dtype=np.float64) * t)])
            return ZeroCurve(node_dates, dfs, self.target.interpolation)

        def residual_fn(x: FloatArray) -> FloatArray:
            trial = market.with_curve(self.target.name, build_curve(x))
            return np.array([h.residual(trial) for h in helpers], dtype=np.float64)

        x0 = np.array([h.quote.value for h in helpers], dtype=np.float64)
        sr = self.solver.solve(residual_fn, x0)
        if not sr.converged:
            raise RuntimeError(
                f"calibration did not converge: max|residual|={sr.max_abs_residual:.3e} "
                f"({sr.message})"
            )

        curve = build_curve(sr.x)
        out = market.with_curve(self.target.name, curve)
        if self.vol_shim is not None and out.vols is None:
            vns = VolNamespace()
            vns.bind(self.target.name, FlatVolSurface(self.vol_shim))
            out = dataclasses.replace(out, vols=vns)

        residuals = np.array([h.residual(out) for h in helpers], dtype=np.float64)
        return CalibrationResult(
            curve=curve, market=out, solver_result=sr,
            pillar_dates=pillars, residuals=residuals,
        )


__all__ = ["CurveDefinition", "CalibrationResult", "CurveCalibrator", "LOCAL_INTERPOLATORS"]
