"""MarketContext unit tests: discounting, projection, fixings overlay, scenario rebind."""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, ZeroCurve
from finance.markets.paths import FlatPath

CURVE = "USD.SOFR"


def _curve(origin: np.datetime64, level: float = 0.04) -> ZeroCurve:
    nd = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 1825)], dtype="datetime64[D]")
    t = (nd.astype(np.int64) - origin.astype(np.int64)) / 365.0
    dfs = np.exp(-level * t)
    dfs[0] = 1.0
    return ZeroCurve(nd, dfs, CurveInterpolator.LogLinearDF)


class TestMarketContext(UnitTest):
    COVERAGE = ["finance.markets.context"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.origin = np.datetime64("2026-06-01", "D")
        ns = CurveNamespace()
        ns.bind(CURVE, _curve(self.origin))
        self.mkt = MarketContext(as_of_date=self.as_of, curves=ns)

    def test_discount_factor_off_named_curve(self):
        dates = np.array(["2027-06-01"], dtype="datetime64[D]")
        df = self.mkt.discount_factor(CURVE, dates)
        self.assertAlmostEqual(df[0], np.exp(-0.04 * 1.0), places=10)

    def test_project_simple_rates(self):
        starts = np.array(["2026-12-01"], dtype="datetime64[D]")
        ends = np.array(["2027-03-01"], dtype="datetime64[D]")
        got = self.mkt.project(CURVE, starts, ends)[0]
        df_s = self.mkt.discount_factor(CURVE, starts)[0]
        df_e = self.mkt.discount_factor(CURVE, ends)[0]
        tau = (ends[0] - starts[0]).astype(np.int64) / 360.0
        self.assertAlmostEqual(got, (df_s / df_e - 1.0) / tau, places=12)

    def test_project_zero_length_window_is_zero(self):
        d = np.array(["2027-06-01"], dtype="datetime64[D]")
        self.assertEqual(self.mkt.project(CURVE, d, d)[0], 0.0)

    def test_project_overlays_fixings_before_curve(self):
        # a window starting before as_of AND before the curve origin: only the fixings
        # overlay makes this answerable — the curve alone would raise
        mkt = MarketContext(
            as_of_date=self.as_of, curves=self.mkt.curves,
            fixings={CURVE: FlatPath(name=CURVE, as_of_date=self.as_of, value=0.0311)},
        )
        starts = np.array(["2026-03-01", "2026-12-01"], dtype="datetime64[D]")
        ends = np.array(["2026-06-01", "2027-03-01"], dtype="datetime64[D]")
        got = mkt.project(CURVE, starts, ends)
        self.assertAlmostEqual(got[0], 0.0311, places=12)             # realized
        self.assertAlmostEqual(got[1], self.mkt.project(CURVE, starts[1:], ends[1:])[0], places=12)

    def test_with_curve_rebinds_without_mutating_original(self):
        bumped = self.mkt.with_curve(CURVE, _curve(self.origin, level=0.05))
        d = np.array(["2027-06-01"], dtype="datetime64[D]")
        self.assertAlmostEqual(bumped.discount_factor(CURVE, d)[0], np.exp(-0.05), places=10)
        self.assertAlmostEqual(self.mkt.discount_factor(CURVE, d)[0], np.exp(-0.04), places=10)

    def test_with_curve_binds_new_name(self):
        other = self.mkt.with_curve("USD.FEDFUND", _curve(self.origin, level=0.045))
        self.assertIn("USD.FEDFUND", other.curves)
        self.assertIn(CURVE, other.curves)

    def test_forward_rate_delegates_to_curve(self):
        starts = np.array(["2026-12-01"], dtype="datetime64[D]")
        ends = np.array(["2027-06-01"], dtype="datetime64[D]")
        got = self.mkt.forward_rate(CURVE, starts, ends)
        exp = self.mkt.discount(CURVE).forward_rate(starts, ends)
        np.testing.assert_allclose(got, exp)

    def test_convention_helper(self):
        conv = self.mkt.convention("USD", "SOFR")
        self.assertEqual(conv.index.label, "USD SOFR")
        self.assertEqual(conv.swap.spot_calendar, "no_holidays")
