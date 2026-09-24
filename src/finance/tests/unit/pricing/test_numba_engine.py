"""Numba fused pricing/adjoint parity and shape-polymorphic compilation."""
import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Swap, curve_name
from finance.markets.context import MarketContext
from finance.markets.curves import CurveNamespace, YieldCurve, ZeroCurve
from finance.pricing.engines.numba import NumbaProgram
from finance.pricing.engines.numba.program import _execute
from finance.pricing.pricers import SwapPricer
from finance.pricing.risk import NumbaRisk
from finance.pricing.risk.sensitivities import Sensitivities
from finance.pricing.types import Backend

pytest.importorskip("numba")

CURVE = curve_name("USD", "SOFR")


def _market(as_of):
    origin = as_of.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 182, 365, 730, 1825, 3650)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.032, 0.035, 0.039, 0.043, 0.046])
    ns = CurveNamespace()
    ns.bind(YieldCurve.build(dates, np.exp(-z * t), currency="USD", index_name="SOFR"))
    return MarketContext(as_of, ns)


def _book(as_of, n):
    tenors = ("2Y", "3Y", "5Y", "7Y")
    return [
        Swap.fixed_float_swap(
            notional=(i + 1) * 1e6,
            rate_index="SOFR",
            fixed_rate=0.035 + i * 0.001,
            tenor=tenors[i % len(tenors)],
            as_of=as_of,
            index_floor=0.0 if i % 2 else None,
        )
        for i in range(n)
    ]


@pytest.mark.risk
class TestNumbaProgram(UnitTest):
    COVERAGE = [
        "finance.pricing.engines.numba.program",
        "finance.pricing.engines.numba.kernels",
        "finance.pricing.risk.adjoint",
    ]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)

    def test_primal_matches_numpy_for_singleton_and_book(self):
        for size in (1, 4):
            program = SwapPricer().compile(_book(self.as_of, size))
            expected = program.reprice(self.market)
            actual = program.reprice(self.market, backend=Backend.Numba)
            np.testing.assert_allclose(actual.instrument_pv, expected.instrument_pv, rtol=1e-11, atol=1e-6)
            np.testing.assert_allclose(actual.flow_pv, expected.flow_pv, rtol=1e-11, atol=1e-7)
            np.testing.assert_allclose(actual.rate, expected.rate, rtol=1e-12, atol=1e-12)

    def test_different_book_shapes_reuse_one_numba_signature(self):
        before = len(_execute.signatures)
        for size in (1, 3):
            program = NumbaProgram(SwapPricer().compile(_book(self.as_of, size)).inputs, self.market)
            program.reprice(self.market)
        after = len(_execute.signatures)
        self.assertLessEqual(after, max(before, 1))

    def test_adjoint_ladder_matches_bump_and_reprice(self):
        program = SwapPricer().compile(_book(self.as_of, 2))
        analytic = NumbaRisk(program, self.market).zero_ladder(CURVE)
        bumped = Sensitivities(program, self.market).key_rate_durations(CURVE).krd
        np.testing.assert_allclose(analytic, bumped, rtol=2e-4, atol=1e-4)

    def test_fixed_cashflows_have_no_projection_delta(self):
        program = SwapPricer().compile(_book(self.as_of, 2))
        delta = NumbaRisk(program, self.market).cashflow_index_delta(CURVE)
        np.testing.assert_array_equal(delta[program.inputs.rate_kind == 0], 0.0)

    def test_fd_of_adjoint_gamma_matches_jax_hessian(self):
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        from finance.pricing.risk.autodiff import AutodiffRisk

        program = SwapPricer().compile(_book(self.as_of, 1))
        actual = NumbaRisk(program, self.market).gamma(CURVE)
        expected = AutodiffRisk(program, self.market).gamma(CURVE)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)

    def test_reprice_rejects_changed_pillar_geometry(self):
        program = NumbaProgram(SwapPricer().compile(_book(self.as_of, 1)).inputs, self.market)
        yc = self.market.yield_curve(CURVE)
        dates = np.insert(yc.node_dates, 2, yc.node_dates[1] + np.timedelta64(30, "D"))
        dfs = np.interp(dates.astype(np.int64), yc.node_dates.astype(np.int64), yc.zero_curve.dfs)
        changed_curve = yc.with_zero_curve(ZeroCurve(yc.to_x(dates), dfs))
        changed = self.market.with_curve(changed_curve)
        with self.assertRaises(ValueError):
            program.reprice(changed)
