"""RiskEngine — one front door for curve risk, dispatching per the engine-selection policy.

docs/engine_selection.md fixes the policy: the risk *method* is determined by what the
program and market support, never chosen ad hoc at the call site.

- ``adjoint`` — numba installed and every program curve is ``LogLinearDF``: exact
  first-order ladders from one fused pass (linear books via :class:`NumbaRisk`, swaptions
  via the chained forward/annuity adjoint in ``SwaptionProgram.risk``), and FD-of-gradient
  gamma.
- ``bump`` — the universal fallback: central-difference bump-and-reprice over the same
  compiled program.  Any interpolation, any backend, no numba.

``method="auto"`` (default) resolves per the above; pass ``"adjoint"`` / ``"bump"`` to pin
one explicitly (``bump`` is how the adjoint is cross-checked in CI).  Futures are not
routed here — their measures are rate/price/P&L, not PV; use ``FuturesProgram.risk``.
The JAX validation oracle stays separate in ``risk/autodiff.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finance.markets.curves import CurveInterpolator
from finance.pricing.pricers.base import PricingProgram
from finance.pricing.pricers.swaption import SwaptionProgram
from finance.pricing.risk.sensitivities import _zero_rates, bumped_curve

FloatArray = np.ndarray
BP = 1e-4


@dataclass(frozen=True)
class CurveRiskReport:
    """Per-pillar ladder: ``ladder[i, p]`` is instrument i's PV change for +1 ``bp`` at pillar p."""

    pillar_dates: np.ndarray   # (P,) datetime64[D]
    pillar_years: FloatArray   # (P,)
    ladder: FloatArray         # (n_instruments, P)
    method: str                # 'adjoint' | 'bump'

    @property
    def dv01(self) -> FloatArray:
        """Sum across pillars ~ parallel DV01 (key-rate additivity)."""
        return self.ladder.sum(axis=1)


class RiskEngine:
    """Curve risk for a compiled PV-measured program (linear book or swaptions)."""

    def __init__(self, program, market, *, bp: float = BP, method: str = "auto"):
        if not isinstance(program, (PricingProgram, SwaptionProgram)):
            raise TypeError(
                f"RiskEngine handles PV-measured programs, not {type(program).__name__}; "
                "futures expose rate/price/pnl gradients via FuturesProgram.risk"
            )
        self.program = program
        self.market = market
        self.bp = float(bp)
        self.method = self._resolve(method)

    def _inputs(self):
        inner = self.program.program if isinstance(self.program, SwaptionProgram) else self.program
        return inner.inputs

    def _resolve(self, method: str) -> str:
        if method not in ("auto", "adjoint", "bump"):
            raise ValueError("method must be 'auto', 'adjoint', or 'bump'")
        if method != "auto":
            return method
        from finance.pricing import risk

        if risk.NumbaRisk is None:
            return "bump"
        loglinear = all(
            self.market.zero_curve(name).interpolation is CurveInterpolator.LogLinearDF
            for name in self._inputs().curve_names
        )
        return "adjoint" if loglinear else "bump"

    # -- measures -----------------------------------------------------------------------
    def ladder(self, curve_name: str) -> CurveRiskReport:
        """Key-rate ladder across the curve's non-origin pillars, per instrument."""
        base = self.market.zero_curve(curve_name)
        t, _ = _zero_rates(base)
        live = np.nonzero(t > 0)[0]
        if self.method == "adjoint":
            grid = self._adjoint_ladder(curve_name)
        else:
            grid = self._bump_ladder(curve_name, base, live)
        return CurveRiskReport(
            pillar_dates=base.node_dates[live], pillar_years=t[live], ladder=grid, method=self.method
        )

    def dv01(self, curve_name: str) -> FloatArray:
        """Parallel DV01 per instrument (PV change for a +1bp parallel zero-rate move)."""
        if self.method == "adjoint":
            return self.ladder(curve_name).dv01
        base = self.market.zero_curve(curve_name)
        up = self._pv(self._with(curve_name, bumped_curve(base, self.bp)))
        dn = self._pv(self._with(curve_name, bumped_curve(base, -self.bp)))
        return (up - dn) / 2.0

    def gamma(self, curve_name: str, *, step: float = 1e-5) -> FloatArray:
        """Pillar gamma (``0.5 * H * bp**2``) by finite difference of the exact gradient."""
        if self.method != "adjoint":
            raise ValueError(
                "gamma requires the analytic adjoint (FD of the exact gradient); bump gamma "
                "second-differences price and cancels — see docs/engine_selection.md"
            )
        if isinstance(self.program, SwaptionProgram):
            raise NotImplementedError("swaption curve gamma (vol/rate cross terms) is not implemented")
        from finance.pricing.risk.adjoint import NumbaRisk

        return NumbaRisk(self.program, self.market, self.bp).gamma(curve_name, step=step)

    # -- plumbing -----------------------------------------------------------------------
    def _pv(self, market) -> FloatArray:
        result = self.program.reprice(market)
        return result.pv if isinstance(self.program, SwaptionProgram) else result.instrument_pv

    def _with(self, curve_name: str, curve):
        return self.market.with_curve(self.market.yield_curve(curve_name).with_zero_curve(curve))

    def _adjoint_ladder(self, curve_name: str) -> FloatArray:
        if isinstance(self.program, SwaptionProgram):
            return self.program.risk(self.market, curve_name, bp=self.bp).dv01
        from finance.pricing.risk.adjoint import NumbaRisk

        return NumbaRisk(self.program, self.market, self.bp).zero_ladder(curve_name)

    def _bump_ladder(self, curve_name: str, base, live) -> FloatArray:
        if isinstance(self.program, SwaptionProgram):
            n = len(self.program.swaptions)
        else:
            n = self._inputs().n_instruments
        grid = np.zeros((n, live.size), dtype=np.float64)
        for j, pillar in enumerate(live):
            up = self._pv(self._with(curve_name, bumped_curve(base, self.bp, pillar=int(pillar))))
            dn = self._pv(self._with(curve_name, bumped_curve(base, -self.bp, pillar=int(pillar))))
            grid[:, j] = (up - dn) / 2.0
        return grid


__all__ = ["RiskEngine", "CurveRiskReport"]
