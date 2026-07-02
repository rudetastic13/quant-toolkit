"""Direct unit tests for ZeroCurve: construction, queries, extrapolation, error paths."""
import numpy as np

from common.testing import UnitTest
from finance.markets.curves import CurveInterpolator, ZeroCurve

ORIGIN = np.datetime64("2026-06-01", "D")


def _curve(interp: CurveInterpolator = CurveInterpolator.LogLinearDF, level: float = 0.04) -> ZeroCurve:
    nd = np.array(["2026-06-01", "2027-06-01", "2031-06-01"], dtype="datetime64[D]")
    t = (nd.astype(np.int64) - nd[0].astype(np.int64)) / 365.0
    dfs = np.exp(-level * t)
    dfs[0] = 1.0
    return ZeroCurve(nd, dfs, interp)


class TestZeroCurveProperties(UnitTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve"]

    def test_interpolation_property_exposes_scheme(self):
        for interp in (CurveInterpolator.LogLinearDF, CurveInterpolator.RateLinear):
            self.assertIs(_curve(interp).interpolation, interp)

    def test_origin_and_max_date(self):
        c = _curve()
        self.assertEqual(c.origin, ORIGIN)
        self.assertEqual(c.max_date, np.datetime64("2031-06-01"))

    def test_node_arrays_are_copies(self):
        c = _curve()
        c.node_dfs[0] = 99.0
        self.assertEqual(c.node_dfs[0], 1.0)


class TestZeroCurveQueries(UnitTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve"]

    def test_df_exact_on_pillars(self):
        c = _curve()
        np.testing.assert_allclose(c.discount_factor(c.node_dates), c.node_dfs, atol=1e-14)

    def test_df_at_origin_is_one(self):
        c = _curve()
        self.assertAlmostEqual(c.discount_factor(np.array([ORIGIN]))[0], 1.0, places=14)

    def test_zero_rate_recovers_flat_level(self):
        c = _curve(level=0.04)
        d = np.array(["2028-06-01"], dtype="datetime64[D]")
        self.assertAlmostEqual(c.rate(d)[0], 0.04, places=10)

    def test_forward_rate_between_pillars(self):
        c = _curve(level=0.04)
        f = c.forward_rate(
            np.array(["2027-06-01"], dtype="datetime64[D]"),
            np.array(["2028-06-01"], dtype="datetime64[D]"),
        )
        self.assertAlmostEqual(f[0], 0.04, places=10)

    def test_flat_forward_extrapolation_beyond_last_pillar(self):
        c = _curve(level=0.04)
        beyond = np.array(["2033-06-01"], dtype="datetime64[D]")
        # flat-forward: the terminal forward continues, so the zero rate stays ~flat
        self.assertAlmostEqual(c.rate(beyond)[0], 0.04, places=6)

    def test_rate_linear_interpolates_zero_rates(self):
        nd = np.array(["2026-06-01", "2027-06-01", "2029-06-01"], dtype="datetime64[D]")
        t = (nd.astype(np.int64) - nd[0].astype(np.int64)) / 365.0
        z = np.array([0.0, 0.03, 0.05])
        dfs = np.exp(-z * t)
        dfs[0] = 1.0
        c = ZeroCurve(nd, dfs, CurveInterpolator.RateLinear)
        mid = np.array(["2028-06-01"], dtype="datetime64[D]")  # halfway between pillars 2 and 3
        self.assertAlmostEqual(c.rate(mid)[0], 0.04, places=3)


class TestZeroCurveValidation(UnitTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve"]

    def test_first_df_must_be_one(self):
        nd = np.array(["2026-06-01", "2027-06-01"], dtype="datetime64[D]")
        with self.assertRaises(ValueError):
            ZeroCurve(nd, np.array([0.99, 0.95]), CurveInterpolator.LogLinearDF)

    def test_unsorted_dates_rejected(self):
        nd = np.array(["2027-06-01", "2026-06-01"], dtype="datetime64[D]")
        with self.assertRaises(ValueError):
            ZeroCurve(nd, np.array([1.0, 0.96]), CurveInterpolator.LogLinearDF)

    def test_too_few_nodes_rejected(self):
        nd = np.array(["2026-06-01"], dtype="datetime64[D]")
        with self.assertRaises(ValueError):
            ZeroCurve(nd, np.array([1.0]), CurveInterpolator.LogLinearDF)

    def test_pre_origin_query_rejected(self):
        c = _curve()
        with self.assertRaises(ValueError):
            c.discount_factor(np.array(["2026-01-01"], dtype="datetime64[D]"))
