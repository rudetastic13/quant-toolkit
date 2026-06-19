"""Tests for the pure adapter logic — the full curve -> swap -> price chain."""
import numpy as np

from common.testing import UnitTest
from quant_toolkit_xl import api, cache


def _curve_handle():
    pillars = ["2026-06-01", "2026-12-01", "2027-06-01", "2031-06-01", "2036-06-01"]
    t = np.array([0.0, 0.5, 1.0, 5.0, 10.0])
    dfs = np.exp(-0.04 * t)  # flat 4% continuously-compounded
    return api.build_curve(pillars, dfs)


class TestCurveApi(UnitTest):
    COVERAGE = ["quant_toolkit_xl.api"]

    def setUp(self):
        cache.clear()

    def test_build_curve_returns_handle(self):
        self.assertTrue(_curve_handle().startswith("Curve::"))

    def test_build_curve_idempotent(self):
        self.assertEqual(_curve_handle(), _curve_handle())

    def test_discount_factor(self):
        ch = _curve_handle()
        df = api.discount_factor(ch, ["2027-06-01", "2031-06-01"])
        np.testing.assert_allclose(df, np.exp(-0.04 * np.array([1.0, 5.0])), atol=1e-9)

    def test_zero_rate(self):
        ch = _curve_handle()
        zr = api.zero_rate(ch, ["2031-06-01"])
        self.assertAlmostEqual(float(zr[0]), 0.04, places=4)

    def test_wrong_handle_kind_raises(self):
        sh = api.make_swap(100e6, "SOFR", 0.041, "5Y", "2026-06-01")
        with self.assertRaises(TypeError):
            api.discount_factor(sh, ["2026-06-01"])


class TestPricingChain(UnitTest):
    COVERAGE = ["quant_toolkit_xl.api"]

    def setUp(self):
        cache.clear()

    def test_full_chain_prices_a_swap(self):
        as_of = "2026-06-01"
        ch = _curve_handle()
        sh = api.make_swap(100e6, "SOFR", 0.041, "5Y", as_of)
        prog = api.compile_program([sh])
        mkt = api.make_market(as_of, "USD", "SOFR", ch)
        pv = api.price(prog, mkt)
        self.assertEqual(pv.shape, (1,))
        self.assertTrue(np.isfinite(pv).all())

    def test_compile_empty_raises(self):
        with self.assertRaises(ValueError):
            api.compile_program([])

    def test_handles_accepts_2d_range(self):
        # swap handles arriving as an Excel column (nested lists) still compile
        as_of = "2026-06-01"
        sh = api.make_swap(50e6, "SOFR", 0.04, "2Y", as_of)
        prog = api.compile_program([[sh]])
        self.assertTrue(prog.startswith("Program::"))
