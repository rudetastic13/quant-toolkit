"""Shape-polymorphic risk from the Numba hand-written pricing adjoint."""
from __future__ import annotations

import numpy as np

from finance.pricing.types import Backend

FloatArray = np.ndarray
BP = 1e-4


class NumbaRisk:
    """Exact first-order curve risk and FD-of-gradient second-order risk.

    First order is analytic: one fused pricing/adjoint call returns index, funding, total,
    and per-cashflow ladders.  Gamma centrally differences that exact gradient, avoiding the
    cancellation inherent in second-differencing price.  No method traces or JIT-compiles
    based on portfolio shape.
    """

    def __init__(self, program, market, bp: float = BP):
        self.program = program.prepare(market, backend=Backend.Numba)
        self.market = market
        self.bp = float(bp)
        self.z0 = self.program.params_from_market(market)
        self.curve_names = self.program.curve_names
        self._base = None

    @property
    def base(self):
        if self._base is None:
            self._base = self.program.value_and_grad(self.market)
        return self._base

    def index_ladder(self, curve_name: str) -> FloatArray:
        """Instrument projection-curve ladder, expressed as P&L for a +``bp`` move."""
        return self.base.instrument_gradient(curve_name, role="index") * self.bp

    def funding_ladder(self, curve_name: str) -> FloatArray:
        """Instrument discount-curve ladder, expressed as P&L for a +``bp`` move."""
        return self.base.instrument_gradient(curve_name, role="funding") * self.bp

    def zero_ladder(self, curve_name: str) -> FloatArray:
        """Total instrument key-rate ladder (projection + funding)."""
        return self.base.instrument_gradient(curve_name, role="total") * self.bp

    def cashflow_index_delta(self, curve_name: str) -> FloatArray:
        return self.base.cashflow_gradient(curve_name, role="index") * self.bp

    def cashflow_funding_delta(self, curve_name: str) -> FloatArray:
        return self.base.cashflow_gradient(curve_name, role="funding") * self.bp

    def cashflow_zero_delta(self, curve_name: str) -> FloatArray:
        return self.base.cashflow_gradient(curve_name, role="total") * self.bp

    def unit_zero_jacobian(self, curve_name: str) -> FloatArray:
        """Instrument ``dPV/dz`` without basis-point scaling."""
        return self.base.instrument_gradient(curve_name, role="total")

    def gamma(self, curve_name: str, *, step: float = 1e-5) -> FloatArray:
        """Total-book pillar gamma, scaled as ``0.5 * H * bp**2``.

        ``step`` is in absolute zero-rate units and controls only the one central finite
        difference applied to the exact analytic gradient.
        """
        if step <= 0.0:
            raise ValueError("gamma step must be positive")
        sl = self.base.curve_slice(curve_name)
        start, stop = sl.start, sl.stop
        p_count = stop - start
        hessian = np.empty((p_count, p_count), dtype=np.float64)
        for col in range(p_count):
            up = self.z0.copy()
            down = self.z0.copy()
            up[start + col] += step
            down[start + col] -= step
            grad_up = self.program.value_and_grad_params(up, self.market).instrument_gradient(
                curve_name, role="total"
            ).sum(axis=0)
            grad_down = self.program.value_and_grad_params(down, self.market).instrument_gradient(
                curve_name, role="total"
            ).sum(axis=0)
            hessian[:, col] = (grad_up - grad_down) / (2.0 * step)
        hessian = 0.5 * (hessian + hessian.T)
        return 0.5 * hessian * self.bp**2

    def partial_dv01(self, curve_name: str, jacobian: FloatArray) -> FloatArray:
        """Risk in a calibration/risk-map quote basis, allowing rectangular maps."""
        g = self.unit_zero_jacobian(curve_name)
        return (g @ np.linalg.pinv(np.asarray(jacobian, dtype=np.float64))) * self.bp


__all__ = ["NumbaRisk"]
