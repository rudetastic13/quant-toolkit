"""CurveCalibrator — build a curve from calibration instruments + a solver + a target.

This is the "object that calibrates": it wires a list of single-quote calibration
instruments, a solver, and a target curve definition into one residual closure (trial node
values -> per-instrument residuals) and hands that to the solver. The solved node values
are returned as a mathematical ``ZeroCurve`` plus its origin; the input market is never
mutated. Callers register that state under its key (``market[key] = result.zero_curve``).

Parameterisation is continuously-compounded zero rates at the pillars (``DF = exp(-x·t)``,
``t`` in Act/365 from origin); the first node is pinned at the origin with ``DF = 1``.  The
curve geometry (pillar offsets, space, interpolator) is built once as a template and each
solver iteration only rebinds discount factors through ``ZeroCurve.with_dfs``.
Helpers reference a single curve name (their forecast curve); a multi-name basis helper is a
multi-curve follow-up that reuses ``MarketContext.with_curve`` the same way.  Historical fixings
for the target key are read from the market, where they live.  The target's
``discount_keys`` say which discount keys the curve under calibration also serves
(``USD.STDCSA`` for a SOFR OIS curve); nothing is discounted implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from common.math.interpolation import Flat, Interpolator, Linear
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveSpace, RateExtrapolator, ZeroCurve
from finance.instruments.resolution.resolver import curve_name
from finance.pricing.calibration.instruments import CalibrationInstrument
from finance.pricing.calibration.solvers import Solver, SolverResult
from finance.pricing.types import Backend

FloatArray = np.ndarray
DateArray = np.ndarray

# Named schemes whose nodes are *local* (a node affects only its adjacent intervals), so a
# sequential bootstrap is valid.  Kept for callers that speak in ``CurveInterpolator``;
# ``CurveDefinition.is_local`` is the actual check and also covers interpolator instances.
LOCAL_INTERPOLATORS = frozenset({CurveInterpolator.LogLinearDF, CurveInterpolator.RateLinear})


@dataclass(frozen=True)
class CurveDefinition:
    """Identity and mathematical shape of the zero curve being calibrated.

    ``interpolation`` is either a named :class:`CurveInterpolator` (which carries its own
    space, so ``space`` is ignored) or an interpolator instance combined with ``space``.

    The target is a projection curve, ``CCY.INDEX.TENOR`` (``index_name`` ``'SOFR'`` means
    overnight; ``'CME-TERM-SOFR.3M'`` a term tenor).  ``discount_keys`` are the discount keys
    the curve being calibrated is bound under while its quotes are priced — ``('USD.STDCSA',)``
    for an OIS curve that funds its own swaps.  Leave it empty when the market already holds
    the discount curve (a basis calibration); a market never discounts implicitly.
    """

    currency: str
    index_name: str
    interpolation: CurveInterpolator | Interpolator = CurveInterpolator.LogLinearDF
    space: CurveSpace = CurveSpace.LogDF
    extrapolation: RateExtrapolator = RateExtrapolator.FlatForward
    discount_keys: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return curve_name(self.currency, self.index_name)

    @property
    def scheme(self) -> tuple[CurveSpace, Interpolator]:
        if isinstance(self.interpolation, CurveInterpolator):
            return self.interpolation.resolve()
        return self.space, self.interpolation

    @property
    def scheme_label(self) -> str:
        """How the caller named the scheme: the enum member, or ``interpolator in space``."""
        if isinstance(self.interpolation, CurveInterpolator):
            return self.interpolation.name
        return f"{self.interpolation!r} in {self.space.name} space"

    @property
    def is_local(self) -> bool:
        """Whether a node only moves its adjacent intervals (bootstrap-safe)."""
        return isinstance(self.scheme[1], (Linear, Flat))

    @property
    def is_log_linear(self) -> bool:
        space, interpolator = self.scheme
        return space is CurveSpace.LogDF and interpolator == Linear()

    def zero_curve(self, x: FloatArray, dfs: FloatArray) -> ZeroCurve:
        space, interpolator = self.scheme
        return ZeroCurve(x, dfs, space=space, interpolator=interpolator, extrapolation=self.extrapolation)

    def bind(self, market: MarketContext, curve: ZeroCurve) -> MarketContext:
        """``market`` with ``curve`` under the target key and under every ``discount_keys`` entry.

        The market's fixings for the target key (if any) travel with it, since fixings live on
        the market, not on the curve.
        """
        out = market.with_curve(self.name, curve)
        for key in self.discount_keys:
            out = out.with_curve(key, curve)
        return out


@dataclass
class CalibrationResult:
    """The calibrated mathematical curve, its origin, and solver diagnostics.

    ``zero_curve`` is deliberately mathematical state, not a market registration. The
    caller registers it::

        market["USD.SOFR.ON"] = result.zero_curve

    ``jacobian`` (populated only when ``calibrate(..., jacobian=True)``) is the exact
    ``J_ij = ∂impliedᵢ/∂zⱼ`` at the solution, computed by the Numba analytic adjoint by
    default (JAX remains available as an explicit oracle).  It is the
    change-of-variables from pillar (zero-rate) risk to market-quote risk: a partial DV01 to
    the calibration quotes is ``(∂V/∂z) · J⁻¹`` (see ``finance.pricing.risk.autodiff``).
    """

    zero_curve: ZeroCurve
    origin: np.datetime64
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
        Numba path uses the curve's exact affine map and the hand-written swap adjoint, for
        any interpolation scheme; pass ``jacobian_backend=Backend.Jax`` only for validation
        against traced autodiff (log-linear DF curves only).
        """
        jacobian_backend = Backend(jacobian_backend)
        helpers = list(self.instruments)
        if not helpers:
            raise ValueError("CurveCalibrator requires at least one instrument")

        for h in helpers:
            if h.curve != self.target.name:
                raise ValueError(
                    f"instrument targets curve '{h.curve}' but the calibration target is '{self.target.name}'"
                )

        helpers.sort(key=lambda h: h.pillar_date)
        pillars = np.array([h.pillar_date for h in helpers], dtype="datetime64[D]")
        if np.any(pillars[1:] <= pillars[:-1]):
            raise ValueError(
                "calibration instruments must have strictly increasing pillar dates (one node per instrument)"
            )

        origin = market.origin
        if pillars[0] <= origin:
            raise ValueError(f"all pillar dates must be after the curve origin {origin}")

        if self.solver.requires_local_interpolation and not self.target.is_local:
            local = ", ".join(i.name for i in LOCAL_INTERPOLATORS)
            raise ValueError(
                f"{type(self.solver).__name__} requires a local interpolator (Linear or Flat in "
                f"either space, e.g. {local}); got {self.target.scheme_label}"
            )

        # The Numba adjoint differentiates through any scheme (ZeroCurve.log_df_affine); the
        # JAX oracle still models log-linear DF only.
        if jacobian and jacobian_backend == Backend.Jax and not self.target.is_log_linear:
            raise NotImplementedError(
                "calibrate(jacobian_backend=Backend.Jax) requires CurveInterpolator.LogLinearDF "
                f"(the JAX curve model does not support {self.target.scheme_label}); "
                "use the default Numba Jacobian backend"
            )

        x_pillars = (pillars.astype(np.int64) - origin.astype(np.int64)).astype(np.float64)
        x_nodes = np.concatenate([[0.0], x_pillars])
        t = x_pillars / 365.0
        x0 = np.array([h.quote.value for h in helpers], dtype=np.float64)

        def node_dfs(z: FloatArray) -> FloatArray:
            return np.concatenate([[1.0], np.exp(-np.asarray(z, dtype=np.float64) * t)])

        template = self.target.zero_curve(x_nodes, node_dfs(x0))

        def build_curve(z: FloatArray) -> ZeroCurve:
            return template.with_dfs(node_dfs(z))

        def residual_fn(z: FloatArray) -> FloatArray:
            trial = self.target.bind(market, build_curve(z))
            return np.array([h.residual(trial) for h in helpers], dtype=np.float64)

        sr = self.solver.solve(residual_fn, x0)
        if not sr.converged:
            raise RuntimeError(f"calibration did not converge: max|residual|={sr.max_abs_residual:.3e} ({sr.message})")

        curve = build_curve(sr.x)
        out = self.target.bind(market, curve)
        calibrated = out.yield_curve(self.target.name)

        residuals = np.array([h.residual(out) for h in helpers], dtype=np.float64)

        jac = None
        if jacobian:
            if jacobian_backend == Backend.Numba:
                from finance.pricing.engines.numba.calibration import calibration_jacobian
            elif jacobian_backend == Backend.Jax:
                from finance.pricing.engines.jax.calibration import calibration_jacobian
            else:
                raise NotImplementedError(f"calibration Jacobian backend {jacobian_backend.name} is not implemented")

            jac = calibration_jacobian(helpers, self.target.name, out, calibrated)

        return CalibrationResult(
            zero_curve=curve,
            origin=origin,
            solver_result=sr,
            pillar_dates=pillars,
            residuals=residuals,
            jacobian=jac,
        )
