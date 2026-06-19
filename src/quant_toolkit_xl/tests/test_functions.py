"""Tests for the q* UDF layer — delegation and the #QERR error surface."""
import numpy as np

from common.testing import UnitTest
from quant_toolkit_xl import cache, functions as f


def _curve_dfs():
    pillars = ["2026-06-01", "2027-06-01", "2031-06-01"]
    dfs = np.exp(-0.04 * np.array([0.0, 1.0, 5.0]))
    return pillars, dfs


class TestQFunctions(UnitTest):
    COVERAGE = ["quant_toolkit_xl.functions"]

    def setUp(self):
        cache.clear()

    def test_qbuildcurve_returns_handle(self):
        pillars, dfs = _curve_dfs()
        self.assertTrue(f.qBuildCurve(pillars, dfs).startswith("Curve::"))

    def test_qprice_end_to_end(self):
        pillars, dfs = _curve_dfs()
        as_of = "2026-06-01"
        ch = f.qBuildCurve(pillars, dfs)
        sh = f.qSwap(100e6, "SOFR", 0.041, "5Y", as_of)
        prog = f.qCompileProgram([sh])
        mkt = f.qMarket(as_of, "USD", "SOFR", ch)
        pv = f.qPrice(prog, mkt)
        self.assertIsInstance(pv, np.ndarray)
        self.assertTrue(np.isfinite(pv).all())

    def test_error_becomes_qerr_string(self):
        # bad interpolation name -> readable cell string, not an exception
        pillars, dfs = _curve_dfs()
        out = f.qBuildCurve(pillars, dfs, "NotAnInterpolator")
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("#QERR:"))

    def test_wrong_handle_kind_becomes_qerr_string(self):
        sh = f.qSwap(100e6, "SOFR", 0.041, "5Y", "2026-06-01")
        out = f.qDiscountFactor(sh, ["2026-06-01"])  # swap handle where curve expected
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("#QERR:"))
