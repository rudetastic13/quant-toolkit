"""YieldCurve: the dates layer over a pure ZeroCurve."""

import numpy as np

from common.math.interpolation import Cubic, Linear, Mixed
from common.testing import UnitTest
from finance.dates import Term
from finance.markets import HistoricalFixings
from finance.markets.curves import CurveSpace, YieldCurve, ZeroCurve, dates_to_x

ORIGIN = np.datetime64("2026-06-01", "D")
NODE_DATES = np.array(["2026-06-01", "2026-12-01", "2027-06-01", "2028-06-01", "2031-06-01"], dtype="datetime64[D]")


def _dfs(level: float = 0.04) -> np.ndarray:
    dfs = np.exp(-level * dates_to_x(ORIGIN, NODE_DATES) / 365.0)
    dfs[0] = 1.0
    return dfs


def _yield_curve(level: float = 0.04, **kwargs) -> YieldCurve:
    return YieldCurve.build(NODE_DATES, _dfs(level), currency="USD", index_name="SOFR", **kwargs)


class TestDatesToX(UnitTest):
    COVERAGE = ["finance.markets.curves.yield_curve"]

    def test_day_offsets_from_origin(self):
        np.testing.assert_array_equal(dates_to_x(ORIGIN, NODE_DATES[:3]), [0.0, 183.0, 365.0])

    def test_accepts_other_datetime_units(self):
        ns = NODE_DATES[:2].astype("datetime64[ns]")
        np.testing.assert_array_equal(dates_to_x(ORIGIN, ns), [0.0, 183.0])


class TestYieldCurveConstruction(UnitTest):
    COVERAGE = ["finance.markets.curves.yield_curve"]

    def test_build_sets_origin_and_pillars(self):
        yc = _yield_curve()
        self.assertEqual(yc.origin, ORIGIN)
        np.testing.assert_array_equal(yc.node_dates, NODE_DATES)
        self.assertEqual(yc.max_date, NODE_DATES[-1])
        np.testing.assert_array_equal(yc.zero_curve.x, dates_to_x(ORIGIN, NODE_DATES))
        self.assertEqual(yc.name, "USD.SOFR")

    def test_build_forwards_scheme(self):
        yc = _yield_curve(space=CurveSpace.ZeroRate, interpolator=Cubic())
        self.assertIs(yc.zero_curve.space, CurveSpace.ZeroRate)
        self.assertEqual(yc.zero_curve.interpolator, Cubic())

    def test_from_registry_composes_origin_and_zero_curve(self):
        zc = ZeroCurve(np.array([0.0, 365.0]), np.array([1.0, 0.96]))
        yc = YieldCurve.from_registry(ORIGIN, zc, currency="USD", index_name="SOFR")
        self.assertIs(yc.zero_curve, zc)
        self.assertEqual(yc.origin, ORIGIN)
        np.testing.assert_array_equal(yc.node_dates, NODE_DATES[[0, 2]])

    def test_with_zero_curve_and_fixings_preserve_origin(self):
        yc = _yield_curve()
        history = HistoricalFixings(np.array(["2026-05-29"], dtype="datetime64[D]"), np.array([0.03]))
        replaced = yc.with_zero_curve(_yield_curve(0.05).zero_curve).with_historical_fixings(history)
        self.assertEqual(replaced.origin, ORIGIN)
        self.assertIs(replaced.conventions, yc.conventions)
        self.assertIs(replaced.historical_fixings, history)
        self.assertAlmostEqual(replaced.zero_rate(NODE_DATES[-1:])[0], 0.05, places=10)


class TestYieldCurveQueries(UnitTest):
    COVERAGE = ["finance.markets.curves.yield_curve"]

    def test_queries_delegate_through_dates(self):
        yc = _yield_curve(0.04)
        dates = np.array(["2027-06-01", "2029-06-01"], dtype="datetime64[D]")
        x = dates_to_x(ORIGIN, dates)
        np.testing.assert_array_equal(yc.discount_factor(dates), yc.zero_curve.discount_factor(x))
        np.testing.assert_array_equal(yc.log_discount_factor(dates), yc.zero_curve.log_discount_factor(x))
        np.testing.assert_array_equal(yc.zero_rate(dates), yc.zero_curve.zero_rate(x))
        np.testing.assert_array_equal(yc.instantaneous_forward(dates), yc.zero_curve.instantaneous_forward(x))
        np.testing.assert_allclose(yc.zero_rate(dates), 0.04)

    def test_pre_origin_date_rejected(self):
        with self.assertRaises(ValueError):
            _yield_curve().discount_factor(np.array(["2026-05-31"], dtype="datetime64[D]"))

    def test_flat_forward_beyond_last_pillar(self):
        yc = _yield_curve(0.04)
        beyond = np.array(["2035-06-01"], dtype="datetime64[D]")
        self.assertAlmostEqual(yc.zero_rate(beyond)[0], 0.04, places=10)


class TestNodeIndex(UnitTest):
    COVERAGE = ["finance.markets.curves.yield_curve"]

    def test_by_term_and_by_date(self):
        yc = _yield_curve()
        self.assertEqual(yc.node_index(Term.from_str("1Y")), 2)
        self.assertEqual(yc.node_index(np.datetime64("2028-06-01")), 3)
        self.assertEqual(yc.node_index(Term.from_str("0D")), 0)

    def test_off_node_raises(self):
        with self.assertRaises(ValueError):
            _yield_curve().node_index(np.datetime64("2027-07-01"))
        with self.assertRaises(ValueError):
            _yield_curve().node_index(Term.from_str("30Y"))

    def test_places_a_mixed_switch(self):
        yc = _yield_curve()
        k = yc.node_index(Term.from_str("2Y"))
        mixed = yc.with_zero_curve(
            ZeroCurve(yc.zero_curve.x, yc.zero_curve.dfs, interpolator=Mixed(Linear(), Cubic(), k))
        )
        np.testing.assert_allclose(mixed.discount_factor(NODE_DATES), yc.zero_curve.dfs, atol=1e-14)
