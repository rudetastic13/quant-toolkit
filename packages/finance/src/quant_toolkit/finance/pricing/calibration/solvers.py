"""Solvers for curve calibration — pure root-finders over a residual vector.

A solver knows nothing about curves or markets: it takes a residual function ``x ->
residuals`` (both in rate units) plus an initial guess, and returns the ``x`` that zeroes
the residuals.  Two implementations behind one ``Solver`` protocol:

* ``GlobalSolver`` — all nodes at once via scipy ``least_squares`` (finite-difference
  Jacobian, i.e. bump-and-reprice).  Works for any interpolation, including global/spline
  schemes where a node affects the whole curve.
* ``Bootstrapper`` — sequential, one pillar at a time via ``brentq``.  Valid only when the
  interpolation is *local* (residual ``i`` must not depend on node ``j > i``), which the
  calibrator enforces before calling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq, least_squares

FloatArray = np.ndarray
ResidualFn = Callable[[FloatArray], FloatArray]


@dataclass
class SolverResult:
    """Outcome of a solve: the solution, convergence flag, and diagnostics."""

    x: FloatArray
    converged: bool
    iterations: int
    max_abs_residual: float
    message: str = ""


@runtime_checkable
class Solver(Protocol):
    """A root-finder over a residual vector.  ``requires_local_interpolation`` lets the
    calibrator reject an incompatible (global) interpolation up front."""

    requires_local_interpolation: bool

    def solve(self, residual_fn: ResidualFn, x0: FloatArray) -> SolverResult: ...


@dataclass
class GlobalSolver:
    """Solve all nodes simultaneously with scipy ``least_squares`` (Trust Region Reflective).

    The Jacobian is finite-differenced — each column is one bump-and-reprice over the whole
    instrument set — so no analytic derivatives are needed and any interpolation works.
    """

    tol: float = 1e-10        # convergence: max|residual| in rate units at the solution
    xtol: float = 1e-14
    diff_step: float | None = None
    requires_local_interpolation: ClassVar[bool] = False

    def solve(self, residual_fn: ResidualFn, x0: FloatArray) -> SolverResult:
        x0 = np.asarray(x0, dtype=np.float64)
        res = least_squares(
            residual_fn, x0, method="trf",
            xtol=self.xtol, ftol=self.xtol, gtol=self.xtol, diff_step=self.diff_step,
        )
        r = np.abs(np.asarray(residual_fn(res.x), dtype=np.float64))
        max_r = float(r.max()) if r.size else 0.0
        return SolverResult(
            x=np.asarray(res.x, dtype=np.float64),
            converged=bool(max_r < self.tol),
            iterations=int(res.nfev),
            max_abs_residual=max_r,
            message=str(res.message),
        )


@dataclass
class Bootstrapper:
    """Sequential pillar-by-pillar root find via ``brentq``.

    For each pillar ``i`` in order, solve ``residual_fn(x with x[i]=v)[i] == 0`` while
    earlier pillars hold their solved values and later pillars hold ``x0``.  This is correct
    only when the interpolation is *local* — the calibrator enforces that.  The bracket
    starts at ``x0[i] ± bracket_halfwidth`` and doubles until a sign change is found.
    """

    bracket_halfwidth: float = 0.05
    max_bracket_expansions: int = 6
    xtol: float = 1e-14
    tol: float = 1e-10
    requires_local_interpolation: ClassVar[bool] = True

    def solve(self, residual_fn: ResidualFn, x0: FloatArray) -> SolverResult:
        x0 = np.asarray(x0, dtype=np.float64)
        x = x0.copy()
        evals = 0

        for i in range(x.size):
            def g(v: float, i: int = i) -> float:
                nonlocal evals
                evals += 1
                trial = x.copy()
                trial[i] = v
                return float(residual_fn(trial)[i])

            h = self.bracket_halfwidth
            a, b = x0[i] - h, x0[i] + h
            ga, gb = g(a), g(b)
            expansions = 0
            while ga * gb > 0.0:
                if expansions >= self.max_bracket_expansions:
                    raise RuntimeError(
                        f"Bootstrapper: could not bracket a root for pillar {i} "
                        f"(x0={x0[i]:.6g}); residual did not change sign within "
                        f"±{h:.3g}.  Check the quote, or that the interpolation is local."
                    )
                h *= 2.0
                a, b = x0[i] - h, x0[i] + h
                ga, gb = g(a), g(b)
                expansions += 1
            x[i] = brentq(g, a, b, xtol=self.xtol)

        r = np.abs(np.asarray(residual_fn(x), dtype=np.float64))
        max_r = float(r.max()) if r.size else 0.0
        return SolverResult(
            x=x, converged=bool(max_r < self.tol),
            iterations=evals, max_abs_residual=max_r, message="bootstrap complete",
        )


__all__ = ["ResidualFn", "SolverResult", "Solver", "GlobalSolver", "Bootstrapper"]
