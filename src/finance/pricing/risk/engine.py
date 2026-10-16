"""RiskEngine — one front door for curve risk, dispatching per the engine-selection policy.

docs/engine_selection.md fixes the policy: the risk *method* is determined by what the
program and market support, never chosen ad hoc at the call site.

- ``adjoint`` — numba installed: exact first-order ladders from one fused pass (linear
  books via :class:`NumbaRisk`, swaptions via the chained forward/annuity adjoint in
  ``SwaptionProgram.risk``), and FD-of-gradient gamma.  Any interpolation scheme — the
  curve hands the kernel its exact affine map (``ZeroCurve.log_df_affine``).
- ``bump`` — the universal fallback: central-difference bump-and-reprice over the same
  compiled program.  Any backend, no numba.

``method="auto"`` (default) resolves per the above; pass ``"adjoint"`` / ``"bump"`` to pin
one explicitly (``bump`` is how the adjoint is cross-checked in CI).  Futures are not
routed here — their measures are rate/price/P&L, not PV; use ``FuturesProgram.risk`` or
``pillar_risk`` directly.  The JAX validation oracle stays separate in ``risk/autodiff.py``.

With a :class:`~finance.pricing.risk.template.RiskTemplate` the engine also reports
``quote_risk``: pillar risk re-expressed as risk to the template's quotes.  The template is
bound to the market by object identity and rebound automatically when the market changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finance.pricing.engines.layout import ParamLayout
from finance.pricing.pricers.base import PricingProgram
from finance.pricing.pricers.swaption import SwaptionProgram
from finance.pricing.risk.pillar import PillarRisk, pillar_risk
from finance.pricing.risk.template import BoundRiskTemplate, QuoteRisk, RiskTemplate

FloatArray = np.ndarray
BP = 1e-4


@dataclass(frozen=True)
class CurveRiskReport:
    """Per-pillar ladder: ``ladder[i, p]`` is instrument i's PV change for +1 ``bp`` at pillar p."""

    pillar_dates: np.ndarray  # (P,) datetime64[D]
    pillar_years: FloatArray  # (P,)
    ladder: FloatArray  # (n_instruments, P)
    method: str  # 'adjoint' | 'bump'

    @property
    def dv01(self) -> FloatArray:
        """Sum across pillars ~ parallel DV01 (key-rate additivity)."""
        return self.ladder.sum(axis=1)


class RiskEngine:
    """Curve risk for a compiled PV-measured program (linear book or swaptions).

    Parameters
    ----------
    program : PricingProgram or SwaptionProgram
    market : MarketContext
    template : RiskTemplate, optional
        The default template for :meth:`quote_risk`.
    bp : float
        Bump size for the bump method and the ladder scaling.
    method : {"auto", "adjoint", "bump"}
    """

    def __init__(self, program, market, *, template: RiskTemplate | None = None, bp: float = BP, method: str = "auto"):
        if not isinstance(program, (PricingProgram, SwaptionProgram)):
            raise TypeError(
                f"RiskEngine handles PV-measured programs, not {type(program).__name__}; "
                "futures expose rate/price/pnl gradients via FuturesProgram.risk"
            )
        self.program = program
        self.market = market
        self.template = template
        self.bp = float(bp)
        self.method = self._resolve(method)
        self._bound: BoundRiskTemplate | None = None

    def with_market(self, market) -> RiskEngine:
        """The same program, template and method against another market; the template rebinds if the curves moved."""
        out = RiskEngine(self.program, market, template=self.template, bp=self.bp, method=self.method)
        out._bound = self._bound
        return out

    # -- template ------------------------------------------------------------------------
    def pillar_risk(self, *, labels=None) -> PillarRisk:
        """``∂PV/∂z`` per curve block from the analytic adjoint (the only thing a template consumes)."""
        return pillar_risk(self.program, self.market, labels=labels)

    def bound(self, template: RiskTemplate | None = None) -> BoundRiskTemplate:
        """The template bound to this engine's market, rebound only when the market's curves moved."""
        template = self.template if template is None else template
        if template is None:
            raise ValueError("RiskEngine needs a template: pass template= here or at construction")
        if self._bound is None or self._bound.template is not template or not self._bound.matches(self.market):
            self._bound = template.bind(self.market)
        return self._bound

    def quote_risk(self, template: RiskTemplate | None = None, *, labels=None) -> QuoteRisk:
        """Risk to the template's quotes, with the unexplained pillar risk and reconciliation lines."""
        return self.bound(template).quote_risk(self.pillar_risk(labels=labels))

    def _inputs(self):
        inner = self.program.program if isinstance(self.program, SwaptionProgram) else self.program
        return inner.inputs

    def _resolve(self, method: str) -> str:
        if method not in ("auto", "adjoint", "bump"):
            raise ValueError("method must be 'auto', 'adjoint', or 'bump'")
        if method != "auto":
            return method
        from finance.pricing import risk

        return "bump" if risk.NumbaRisk is None else "adjoint"

    # -- measures -----------------------------------------------------------------------
    def _block(self, curve_name: str) -> tuple[ParamLayout, str]:
        layout = ParamLayout.build(self.market, [curve_name])
        return layout, layout.block_of(curve_name)

    def ladder(self, curve_name: str) -> CurveRiskReport:
        """Key-rate ladder across the pillars of the block ``curve_name`` reports under, per instrument."""
        layout, block = self._block(curve_name)
        if self.method == "adjoint":
            grid = self._adjoint_ladder(curve_name)
        else:
            grid = self._bump_ladder(block, layout.block_slice(block).stop - layout.block_slice(block).start)
        return CurveRiskReport(
            pillar_dates=layout.pillar_dates(block),
            pillar_years=layout.pillar_years(block),
            ladder=grid,
            method=self.method,
        )

    def dv01(self, curve_name: str) -> FloatArray:
        """Parallel DV01 per instrument (PV change for a +1bp move of the whole block)."""
        if self.method == "adjoint":
            return self.ladder(curve_name).dv01
        _, block = self._block(curve_name)
        up = self._pv(self.market.bump(block, self.bp))
        dn = self._pv(self.market.bump(block, -self.bp))
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

    def _adjoint_ladder(self, curve_name: str) -> FloatArray:
        if isinstance(self.program, SwaptionProgram):
            return self.program.risk(self.market, curve_name, bp=self.bp).dv01
        from finance.pricing.risk.adjoint import NumbaRisk

        return NumbaRisk(self.program, self.market, self.bp).zero_ladder(curve_name)

    def _bump_ladder(self, block: str, n_params: int) -> FloatArray:
        if isinstance(self.program, SwaptionProgram):
            n = len(self.program.swaptions)
        else:
            n = self._inputs().n_instruments
        grid = np.zeros((n, n_params), dtype=np.float64)
        for j in range(n_params):
            shift = np.zeros(n_params)
            shift[j] = self.bp
            up = self._pv(self.market.bump(block, shift))
            dn = self._pv(self.market.bump(block, -shift))
            grid[:, j] = (up - dn) / 2.0
        return grid


__all__ = ["RiskEngine", "CurveRiskReport"]
