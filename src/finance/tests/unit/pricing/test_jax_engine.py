"""JAX engine: PV parity with the numpy kernels, and autodiff ladders vs bump-and-reprice.

The JaxProgram consumes the *same* compiled KernelInputs as ``compiler.reprice`` — these
tests pin the two engines to each other (PV) and the gradient ladders to the numpy bump
path (risk), on a book that exercises fixed, shaped float, and compounded legs.
"""
import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Swap, curve_name
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, ZeroCurve
from finance.pricing.pricers import SwapPricer

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from finance.pricing.engines.jax.program import JaxProgram  # noqa: E402
from finance.pricing.risk.autodiff import AutodiffRisk  # noqa: E402
from finance.pricing.risk.sensitivities import Sensitivities  # noqa: E402

CURVE = curve_name("USD", "SOFR")


def _market(as_of: Date) -> MarketContext:
    origin = np.datetime64(as_of.to_str(), "D")
    nd = np.array(
        [origin + np.timedelta64(d, "D") for d in (0, 182, 365, 730, 1825, 3650)],
        dtype="datetime64[D]",
    )
    t = (nd.astype(np.int64) - origin.astype(np.int64)) / 365.0
    z = np.interp(t, [0.0, t[-1]], [0.035, 0.048])
    dfs = np.exp(-z * t)
    dfs[0] = 1.0
    ns = CurveNamespace()
    ns.bind(CURVE, ZeroCurve(nd, dfs, CurveInterpolator.LogLinearDF))
    return MarketContext(as_of_date=as_of, curves=ns)


@pytest.mark.risk
class TestJaxEngineParity(UnitTest):
    COVERAGE = [
        "finance.pricing.engines.jax.program",
        "finance.pricing.engines.jax.curves",
        "finance.pricing.risk.autodiff",
    ]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.book = [
            Swap.fixed_float_swap(notional=10e6, rate_index="SOFR", fixed_rate=0.04, tenor="5Y", as_of=self.as_of),
            Swap.fixed_float_swap(notional=-25e6, rate_index="SOFR", fixed_rate=0.045, tenor="7Y", as_of=self.as_of),
            Swap.fixed_float_swap(notional=5e6, rate_index="SOFR", fixed_rate=0.038, tenor="3Y", as_of=self.as_of,
                 index_floor=0.0, cap=0.06, floor=0.005),
        ]
        self.program = SwapPricer().compile(self.book)

    def test_pv_matches_numpy_engine(self):
        numpy_pv = self.program.reprice(self.market).instrument_pv
        jp = JaxProgram(self.program.inputs, self.market)
        disc0, proj0 = jp.params_from_market(self.market)
        jax_pv = np.asarray(jp.instrument_pv(disc0, proj0))
        np.testing.assert_allclose(jax_pv, numpy_pv, rtol=1e-9, atol=1e-4)

    def test_index_plus_funding_ladder_matches_bump_dv01(self):
        risk = AutodiffRisk(self.program, self.market)
        total = risk.index_ladder(CURVE) + risk.funding_ladder(CURVE)
        bump = Sensitivities(self.program, self.market).key_rate_durations(CURVE)
        # bump is finite-difference; autodiff is exact — agree to ~0.5%
        np.testing.assert_allclose(total.sum(axis=1), bump.total, rtol=5e-3)

    def test_fixed_leg_has_zero_index_delta(self):
        risk = AutodiffRisk(self.program, self.market)
        idx_delta = risk.cashflow_index_delta(CURVE)
        rk = self.program.inputs.rate_kind
        fixed_rows = np.abs(idx_delta[rk == 0]).max() if (rk == 0).any() else 0.0
        self.assertLess(fixed_rows, 1e-10)
