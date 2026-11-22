"""Tests for the q* worksheet-function layer — delegation and the #QERR error surface.

xlOil's ``@func`` decorator leaves the functions callable outside Excel, so these run anywhere
xloil installs (Windows); elsewhere they skip.
"""
import importlib.util
import unittest

import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.xl import cache

HAS_XLOIL = importlib.util.find_spec("xloil") is not None


def _curve_dfs():
    pillars = ["2026-06-01", "2027-06-01", "2031-06-01"]
    dfs = np.exp(-0.04 * np.array([0.0, 1.0, 5.0]))
    return pillars, dfs


@unittest.skipUnless(HAS_XLOIL, "xloil not installed (Windows-only)")
class TestQFunctions(UnitTest):

    def setUp(self):
        from quant_toolkit.xl import functions

        self.f = functions
        cache.clear()

    def test_qbuildcurve_returns_handle(self):
        pillars, dfs = _curve_dfs()
        self.assertTrue(self.f.qBuildCurve(pillars, dfs).startswith("Curve::"))

    def test_qprice_end_to_end(self):
        f = self.f
        pillars, dfs = _curve_dfs()
        as_of = "2026-06-01"
        ch = f.qBuildCurve(pillars, dfs)
        sh = f.qSwap(100e6, "USD.SOFR.1D", 0.041, "5Y", as_of)
        prog = f.qCompileProgram([sh])
        mkt = f.qMarket(as_of, "USD", "USD.SOFR.1D", ch)
        pv = f.qPrice(prog, mkt)
        self.assertIsInstance(pv, np.ndarray)
        self.assertEqual(pv.shape, (1, 1))  # a column: spills down in Excel
        self.assertTrue(np.isfinite(pv).all())

    def test_error_becomes_qerr_string(self):
        # bad interpolation name -> readable cell string, not an exception
        pillars, dfs = _curve_dfs()
        out = self.f.qBuildCurve(pillars, dfs, "NotAnInterpolator")
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("#QERR:"))

    def test_wrong_handle_kind_becomes_qerr_string(self):
        sh = self.f.qSwap(100e6, "USD.SOFR.1D", 0.041, "5Y", "2026-06-01")
        out = self.f.qDiscountFactor(sh, ["2026-06-01"])  # swap handle where curve expected
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("#QERR:"))
