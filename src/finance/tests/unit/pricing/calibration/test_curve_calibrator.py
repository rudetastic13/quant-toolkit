"""CurveCalibrator unit tests.

Round-trip: build a synthetic 'true' curve whose nodes sit exactly on the calibration
pillars, generate each instrument's quote from it, then calibrate from an empty market and
recover the true node DFs.  Also: reprice consistency, global-vs-bootstrap agreement,
validation failures, the vol shim, and that the input market is never mutated.
"""
import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, ZeroCurve
from finance.markets.vols import FlatVolSurface
from finance.pricing.calibration import (
    Bootstrapper,
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    Quote,
    deposit_helper,
    fra_helper,
    swap_helper,
)

CURVE = "USD.SOFR"


def _helpers(as_of: Date):
    return [
        deposit_helper(rate=0.0, tenor="1M", as_of=as_of),
        deposit_helper(rate=0.0, tenor="3M", as_of=as_of),
        deposit_helper(rate=0.0, tenor="6M", as_of=as_of),
        fra_helper(rate=0.0, start="6M", end="12M", as_of=as_of),
        swap_helper(rate=0.0, tenor="2Y", as_of=as_of),
        swap_helper(rate=0.0, tenor="3Y", as_of=as_of),
        swap_helper(rate=0.0, tenor="5Y", as_of=as_of),
        swap_helper(rate=0.0, tenor="10Y", as_of=as_of),
    ]


@pytest.mark.calibration
class TestCurveCalibrator(UnitTest):
    COVERAGE = ["finance.pricing.calibration.calibrator", "finance.markets.context"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.origin = np.datetime64("2026-06-01", "D")
        self.helpers = _helpers(self.as_of)

        # True curve: nodes exactly on the calibration pillars so recovery is machine-clean.
        pillars = sorted(h.pillar_date for h in self.helpers)
        nd = np.array([self.origin, *pillars], dtype="datetime64[D]")
        t = (nd.astype(np.int64) - self.origin.astype(np.int64)) / 365.0
        z = np.interp(t, [0.0, t[-1]], [0.03, 0.05])  # upward-sloping 3% -> 5%
        dfs = np.exp(-z * t)
        dfs[0] = 1.0
        self.true_nd = nd
        self.true = ZeroCurve(nd, dfs, CurveInterpolator.LogLinearDF)

        ns = CurveNamespace()
        ns.bind(CURVE, self.true)
        true_mkt = MarketContext(as_of_date=self.as_of, curves=ns)

        # Quotes generated from the true curve.
        for h in self.helpers:
            h.quote = Quote(h.implied(true_mkt), h.quote.kind)

        self.base = MarketContext(as_of_date=self.as_of, curves=CurveNamespace())
        self.defn = CurveDefinition(CURVE, CurveInterpolator.LogLinearDF)

    def test_round_trip_recovers_true_dfs(self):
        res = CurveCalibrator(self.helpers, GlobalSolver(), self.defn).calibrate(self.base)
        # Calibrated nodes coincide with the true nodes → compare DFs directly.
        np.testing.assert_allclose(res.curve.node_dfs, self.true.node_dfs, atol=1e-9)

    def test_residuals_are_zero(self):
        res = CurveCalibrator(self.helpers, GlobalSolver(), self.defn).calibrate(self.base)
        self.assertLess(np.abs(res.residuals).max(), 1e-10)

    def test_global_and_bootstrap_agree(self):
        rg = CurveCalibrator(self.helpers, GlobalSolver(), self.defn).calibrate(self.base)
        rb = CurveCalibrator(self.helpers, Bootstrapper(), self.defn).calibrate(self.base)
        np.testing.assert_allclose(rg.curve.node_dfs, rb.curve.node_dfs, atol=1e-9)

    def test_vol_shim_bound_and_input_untouched(self):
        res = CurveCalibrator(self.helpers, GlobalSolver(), self.defn, vol_shim=0.2).calibrate(self.base)
        self.assertIsInstance(res.market.vols.resolve(CURVE), FlatVolSurface)
        self.assertEqual(res.market.vols.resolve(CURVE).level, 0.2)
        # Input market never mutated: still no curve, still no vols.
        self.assertNotIn(CURVE, self.base.curves)
        self.assertIsNone(self.base.vols)

    def test_bootstrap_rejects_global_interpolator(self):
        defn = CurveDefinition(CURVE, CurveInterpolator.LogCubicDF)
        with self.assertRaises(ValueError):
            CurveCalibrator(self.helpers, Bootstrapper(), defn).calibrate(self.base)

    def test_duplicate_pillars_raise(self):
        dup = [
            swap_helper(rate=0.0, tenor="5Y", as_of=self.as_of),
            swap_helper(rate=0.0, tenor="5Y", as_of=self.as_of),
        ]
        for h in dup:
            h.quote = Quote(0.04, h.quote.kind)
        with self.assertRaises(ValueError):
            CurveCalibrator(dup, GlobalSolver(), self.defn).calibrate(self.base)

    def test_wrong_target_name_raises(self):
        defn = CurveDefinition("USD.NOTSOFR", CurveInterpolator.LogLinearDF)
        with self.assertRaises(ValueError):
            CurveCalibrator(self.helpers, GlobalSolver(), defn).calibrate(self.base)

    def test_empty_instruments_raise(self):
        with self.assertRaises(ValueError):
            CurveCalibrator([], GlobalSolver(), self.defn).calibrate(self.base)


@pytest.mark.calibration
class TestWithCurve(UnitTest):
    COVERAGE = ["finance.markets.context"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        nd = np.array(["2026-06-01", "2028-06-01"], dtype="datetime64[D]")
        self.c1 = ZeroCurve(nd, np.array([1.0, 0.92]))
        self.c2 = ZeroCurve(nd, np.array([1.0, 0.95]))
        ns = CurveNamespace()
        ns.bind(CURVE, self.c1)
        self.mkt = MarketContext(as_of_date=self.as_of, curves=ns, vols=object())

    def test_with_curve_rebinds_and_leaves_original_intact(self):
        out = self.mkt.with_curve(CURVE, self.c2)
        self.assertIs(out.discount(CURVE), self.c2)
        self.assertIs(self.mkt.discount(CURVE), self.c1)  # original unchanged
        # fixings/vols shared by reference
        self.assertIs(out.vols, self.mkt.vols)
        self.assertIs(out.fixings, self.mkt.fixings)

    def test_with_curve_binds_new_name(self):
        out = self.mkt.with_curve("USD.OTHER", self.c2)
        self.assertIs(out.discount("USD.OTHER"), self.c2)
        self.assertNotIn("USD.OTHER", self.mkt.curves)
