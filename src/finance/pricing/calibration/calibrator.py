"""CurveCalibrator — solve the parameters a curve key owns from calibration instruments.

This is the "object that calibrates": it wires a list of single-quote calibration
instruments, a solver and a target definition into one residual closure (trial parameters ->
per-instrument residuals) and hands that to the solver.  The input market is never mutated;
the result carries the solved atom and the market with it in place.

Two cases, one loop:

* **A new curve.**  The target key is not in the market.  The calibrator builds a ``ZeroCurve``
  with one pillar per instrument (continuously compounded Act/365 zero rates, ``DF = exp(-z t)``,
  the origin pinned at ``DF = 1``), solves it, and registers it under the key and under every
  ``discount_keys`` entry (``USD.STDCSA`` for a SOFR OIS curve that funds its own swaps).
* **A registered curve.**  The target key is in the market and owns exactly one atom
  (``market.owned(key)``): the calibrator solves that atom's parameters with everything else
  held, through ``market.with_params``.  Registering ``USD.CME-TERM-SOFR.1M = USD.SOFR.1D +
  spread`` and calibrating the 1M key solves the spread alone: the sequential basis
  bootstrap (OIS, then the 1M basis, then 6M over 1M) is this loop run three times.
  :meth:`CurveDefinition.flat_on_pillars` builds the spread atom on the instruments' pillars.

Helpers reference a single curve name (their forecast curve); nothing is discounted
implicitly.  Historical fixings for the target key are read from the market, where they live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from common.math.interpolation import Flat, Interpolator, Linear
from finance.conventions import CurveKey
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveSpace, RateExtrapolator, ZeroCurve, dates_to_x
from finance.pricing.calibration.instruments import CalibrationInstrument
from finance.pricing.calibration.solvers import Solver, SolverResult
from finance.pricing.types import Backend

FloatArray = np.ndarray
DateArray = np.ndarray

# Named schemes whose nodes are *local* (a node affects only its adjacent intervals), so a
# sequential bootstrap is valid.  Kept for callers that speak in ``CurveInterpolator``;
# ``CurveDefinition.is_local`` is the actual check and also covers interpolator instances.
LOCAL_INTERPOLATORS = frozenset({CurveInterpolator.LogLinearDF, CurveInterpolator.RateLinear})


def _is_local(scheme: tuple[CurveSpace, Interpolator]) -> bool:
    return isinstance(scheme[1], (Linear, Flat))


@dataclass(frozen=True)
class CurveDefinition:
    """The key being calibrated and, for a curve built from scratch, its mathematical shape.

    Parameters
    ----------
    key : str
        ``'USD.SOFR.1D'`` (projection), ``'USD.CME-TERM-SOFR.1M'``, ``'USD.UST'`` (discount).
    interpolation : CurveInterpolator or Interpolator
        A named scheme (which carries its own space, so ``space`` is ignored) or an
        interpolator instance combined with ``space``.  Used when the key is new to the
        market; a registered key keeps the scheme of the atom it owns.
    space, extrapolation
        As for ``ZeroCurve``; used when the key is new.
    discount_keys : tuple of str
        Discount keys the calibrated curve object also serves while its quotes are priced
        (``('USD.STDCSA',)`` for an OIS curve that funds its own swaps).  Leave empty when the
        market already holds the discount curve (a basis calibration).
    """

    key: str
    interpolation: CurveInterpolator | Interpolator = CurveInterpolator.LogLinearDF
    space: CurveSpace = CurveSpace.LogDF
    extrapolation: RateExtrapolator = RateExtrapolator.FlatForward
    discount_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(CurveKey.parse(self.key)))
        object.__setattr__(self, "discount_keys", tuple(str(CurveKey.parse(k)) for k in self.discount_keys))

    @property
    def name(self) -> str:
        return self.key

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
        return _is_local(self.scheme)

    @property
    def is_log_linear(self) -> bool:
        space, interpolator = self.scheme
        return space is CurveSpace.LogDF and interpolator == Linear()

    def zero_curve(self, x: FloatArray, dfs: FloatArray) -> ZeroCurve:
        space, interpolator = self.scheme
        return ZeroCurve(x, dfs, space=space, interpolator=interpolator, extrapolation=self.extrapolation)

    def flat_on_pillars(
        self, market: MarketContext, instruments: Sequence[CalibrationInstrument], rate: float = 0.0
    ) -> ZeroCurve:
        """A flat curve (zero rate ``rate`` everywhere) with one pillar per instrument, in this definition's scheme.

        The starting point of a basis calibration: ``market["USD.CME-TERM-SOFR.1M"] =
        market["USD.SOFR.1D"] + definition.flat_on_pillars(market, helpers)`` registers a spread
        atom the calibrator can then solve.
        """
        pillars = _pillars(instruments, market.origin)
        x = np.concatenate([[0.0], dates_to_x(market.origin, pillars)])
        dfs = np.exp(-float(rate) * x / 365.0)
        dfs[0] = 1.0
        return self.zero_curve(x, dfs)

    def bind(self, market: MarketContext, curve: ZeroCurve) -> MarketContext:
        """``market`` with ``curve`` under the target key and under every ``discount_keys`` entry (a new curve)."""
        out = market.with_curve(self.key, curve)
        for key in self.discount_keys:
            out = out.with_curve(key, curve)
        return out


def _pillars(instruments: Sequence[CalibrationInstrument], origin: np.datetime64) -> DateArray:
    """Sorted pillar dates, one per instrument, all after ``origin``."""
    helpers = sorted(instruments, key=lambda h: h.pillar_date)
    pillars = np.array([h.pillar_date for h in helpers], dtype="datetime64[D]")
    if np.any(pillars[1:] <= pillars[:-1]):
        raise ValueError("calibration instruments must have strictly increasing pillar dates (one node per instrument)")
    if pillars.size and pillars[0] <= origin:
        raise ValueError(f"all pillar dates must be after the curve origin {origin}")
    return pillars


@dataclass
class CalibrationResult:
    """The solved curve, the market holding it, and solver diagnostics.

    ``zero_curve`` is the solved atom: the whole curve for a new key, the spread (the owned
    atom) for a registered basis key.  ``market`` is the input market with it in place, frozen;
    nothing in the input was mutated.  ``jacobian`` (only with ``calibrate(..., jacobian=True)``)
    is the exact ``J_ij = ∂impliedᵢ/∂zⱼ`` at the solution over the solved parameters, from the
    Numba analytic adjoint by default (JAX remains an explicit oracle for log-linear atoms): the
    change-of-variables from pillar risk to market-quote risk, ``(∂V/∂z) · J⁻¹`` (see
    ``finance.pricing.risk``).
    """

    zero_curve: ZeroCurve
    origin: np.datetime64
    solver_result: SolverResult
    pillar_dates: DateArray
    residuals: FloatArray
    market: MarketContext
    jacobian: FloatArray | None = None


@dataclass
class CurveCalibrator:
    """Calibrate one key from a set of single-quote instruments against a base market."""

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
        """Solve for the target key.

        ``jacobian=True`` additionally captures ``∂implied/∂z`` at the solution.  The default
        Numba path uses the curve's exact affine map and the hand-written swap adjoint, for
        any interpolation scheme; pass ``jacobian_backend=Backend.Jax`` only for validation
        against traced autodiff (log-linear DF atoms only).
        """
        jacobian_backend = Backend(jacobian_backend)
        helpers = list(self.instruments)
        if not helpers:
            raise ValueError("CurveCalibrator requires at least one instrument")
        key = self.target.key
        for h in helpers:
            if h.curve != key:
                raise ValueError(f"instrument targets curve '{h.curve}' but the calibration target is '{key}'")

        helpers.sort(key=lambda h: h.pillar_date)
        origin = market.origin
        pillars = _pillars(helpers, origin)
        x_pillars = dates_to_x(origin, pillars)

        if key in market:
            atoms = market.owned(key)
            if len(atoms) != 1:
                raise ValueError(
                    f"'{key}' owns {len(atoms)} atoms; a calibration solves exactly one (register the "
                    "curve as base + one spread, or calibrate the components under their own keys)"
                )
            template = atoms[0]
            if not isinstance(template, ZeroCurve) or not np.array_equal(template.x[1:], x_pillars):
                raise ValueError(
                    f"'{key}' owns a curve whose pillars {template.x[1:].tolist()} are not the instruments' "
                    f"{x_pillars.tolist()}; build it with CurveDefinition.flat_on_pillars(market, instruments)"
                )
            scheme = template.scheme
            scheme_label = f"{scheme[1]!r} in {scheme[0].name} space"
            x0 = template.params

            def bind(curve: ZeroCurve) -> MarketContext:
                return market.with_params(key, curve.params)
        else:
            x_nodes = np.concatenate([[0.0], x_pillars])
            t = x_pillars / 365.0
            x0 = np.array([h.quote.value for h in helpers], dtype=np.float64)
            template = self.target.zero_curve(x_nodes, np.concatenate([[1.0], np.exp(-x0 * t)]))
            scheme = self.target.scheme
            scheme_label = self.target.scheme_label

            def bind(curve: ZeroCurve) -> MarketContext:
                return self.target.bind(market, curve)

        if self.solver.requires_local_interpolation and not _is_local(scheme):
            local = ", ".join(i.name for i in LOCAL_INTERPOLATORS)
            raise ValueError(
                f"{type(self.solver).__name__} requires a local interpolator (Linear or Flat in "
                f"either space, e.g. {local}); got {scheme_label}"
            )
        is_log_linear = scheme[0] is CurveSpace.LogDF and scheme[1] == Linear()
        if jacobian and jacobian_backend == Backend.Jax and not is_log_linear:
            raise NotImplementedError(
                "calibrate(jacobian_backend=Backend.Jax) requires CurveInterpolator.LogLinearDF "
                f"(the JAX curve model does not support {scheme_label}); use the default Numba Jacobian backend"
            )

        def residual_fn(z: FloatArray) -> FloatArray:
            trial = bind(template.with_params(z))
            return np.array([h.residual(trial) for h in helpers], dtype=np.float64)

        sr = self.solver.solve(residual_fn, x0)
        if not sr.converged:
            raise RuntimeError(f"calibration did not converge: max|residual|={sr.max_abs_residual:.3e} ({sr.message})")

        out = bind(template.with_params(sr.x))
        curve = out.owned(key)[0]  # the solved atom as the market holds it
        residuals = np.array([h.residual(out) for h in helpers], dtype=np.float64)

        jac = None
        if jacobian:
            if jacobian_backend == Backend.Numba:
                from finance.pricing.risk.jacobian import calibration_jacobian
            elif jacobian_backend == Backend.Jax:
                from finance.pricing.engines.jax.calibration import calibration_jacobian
            else:
                raise NotImplementedError(f"calibration Jacobian backend {jacobian_backend.name} is not implemented")
            jac = calibration_jacobian(helpers, key, out, out.yield_curve(key))  # the JAX oracle still reads the curve

        return CalibrationResult(
            zero_curve=curve,
            origin=origin,
            solver_result=sr,
            pillar_dates=pillars,
            residuals=residuals,
            market=out,
            jacobian=jac,
        )
