"""ZeroCurve on day offsets: construction, queries in both spaces, extrapolation, rebinding."""

import numpy as np

from common.math.interpolation import Cubic, Linear, Mixed, Quadratic
from common.testing import UnitTest
from finance.markets.curves import CurveInterpolator, CurveSpace, RateExtrapolator, ZeroCurve

X = np.array([0.0, 365.0, 1825.0])


def _curve(level: float = 0.04, **kwargs) -> ZeroCurve:
    dfs = np.exp(-level * X / 365.0)
    dfs[0] = 1.0
    return ZeroCurve(X, dfs, **kwargs)


class TestZeroCurveState(UnitTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve"]

    def test_defaults_are_log_linear_flat_forward(self):
        c = _curve()
        self.assertIs(c.space, CurveSpace.LogDF)
        self.assertEqual(c.interpolator, Linear())
        self.assertIs(c.extrapolation, RateExtrapolator.FlatForward)
        self.assertTrue(c.is_log_linear)
        self.assertEqual(c.max_x, 1825.0)

    def test_is_log_linear_requires_both_space_and_scheme(self):
        self.assertFalse(_curve(space=CurveSpace.ZeroRate).is_log_linear)
        self.assertFalse(_curve(interpolator=Cubic()).is_log_linear)

    def test_node_zero_rates_recover_level(self):
        np.testing.assert_allclose(_curve(0.04).node_zero_rates, [0.04, 0.04])

    def test_flat_alias_of_flat_forward(self):
        self.assertIs(RateExtrapolator.Flat, RateExtrapolator.FlatForward)

    def test_repr_names_configuration(self):
        text = repr(_curve(space=CurveSpace.ZeroRate, interpolator=Cubic()))
        self.assertIn("ZeroRate", text)
        self.assertIn("Cubic", text)

    def test_line_exposes_coefficients_for_engines(self):
        c = _curve()
        self.assertEqual(c.line.coefficients.shape, (2, 2))
        np.testing.assert_allclose(c.line.coefficients[0], [-0.04 / 365.0, -0.04 / 365.0])


class TestZeroCurveQueries(UnitTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve"]

    def test_df_exact_on_pillars_in_both_spaces(self):
        for space in CurveSpace:
            for interpolator in (Linear(), Cubic(), Quadratic()):
                with self.subTest(space=space, interpolator=interpolator):
                    c = _curve(space=space, interpolator=interpolator)
                    np.testing.assert_allclose(c.discount_factor(X), c.dfs, atol=1e-14)

    def test_df_at_origin_is_one(self):
        self.assertAlmostEqual(_curve().discount_factor(np.array([0.0]))[0], 1.0, places=14)

    def test_zero_rate_recovers_flat_level(self):
        for space in CurveSpace:
            with self.subTest(space=space):
                c = _curve(0.04, space=space)
                self.assertAlmostEqual(c.zero_rate(np.array([730.0]))[0], 0.04, places=10)

    def test_zero_rate_at_origin_is_the_instantaneous_forward(self):
        for space in CurveSpace:
            with self.subTest(space=space):
                c = _curve(0.04, space=space)
                self.assertAlmostEqual(c.zero_rate(np.array([0.0]))[0], 0.04, places=10)

    def test_log_discount_factor(self):
        self.assertAlmostEqual(_curve(0.04).log_discount_factor(np.array([365.0]))[0], -0.04, places=12)

    def test_instantaneous_forward_of_flat_curve(self):
        for space in CurveSpace:
            with self.subTest(space=space):
                c = _curve(0.04, space=space)
                np.testing.assert_allclose(c.instantaneous_forward(np.array([100.0, 1000.0, 3000.0])), 0.04)

    def test_rate_linear_interpolates_zero_rates(self):
        x = np.array([0.0, 365.0, 1095.0])
        z = np.array([0.0, 0.03, 0.05])
        dfs = np.exp(-z * x / 365.0)
        dfs[0] = 1.0
        c = ZeroCurve(x, dfs, space=CurveSpace.ZeroRate, interpolator=Linear())
        self.assertAlmostEqual(c.zero_rate(np.array([730.0]))[0], 0.04, places=12)

    def test_mixed_switches_scheme_at_node(self):
        x = np.array([0.0, 182.0, 365.0, 730.0, 1825.0, 3650.0])
        z = np.array([0.0, 0.041, 0.039, 0.037, 0.038, 0.040])
        dfs = np.exp(-z * x / 365.0)
        dfs[0] = 1.0
        mixed = ZeroCurve(x, dfs, interpolator=Mixed(Linear(), Cubic(), switch_node=3))
        linear = ZeroCurve(x, dfs, interpolator=Linear())
        cubic = ZeroCurve(x[3:] - x[3], dfs[3:] / dfs[3], interpolator=Cubic())
        short = np.array([90.0, 500.0, 729.0])
        np.testing.assert_allclose(mixed.discount_factor(short), linear.discount_factor(short), rtol=1e-13)
        long = np.array([900.0, 2500.0, 3600.0])
        np.testing.assert_allclose(mixed.discount_factor(long), dfs[3] * cubic.discount_factor(long - x[3]), rtol=1e-13)


class TestZeroCurveExtrapolation(UnitTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve"]

    def test_flat_forward_continues_terminal_forward(self):
        x = np.array([0.0, 365.0, 730.0])
        dfs = np.array([1.0, np.exp(-0.03), np.exp(-0.03 - 0.05)])  # zero rates 3% at 1y, 4% at 2y
        # Log-linear DF: the last interval's forward (5%) continues.  Linear in zero rate:
        # f(T) = r(T) + T r'(T) = 4% + 2 * 1% = 6% continues.
        for space, terminal_forward in ((CurveSpace.LogDF, 0.05), (CurveSpace.ZeroRate, 0.06)):
            with self.subTest(space=space):
                c = ZeroCurve(x, dfs, space=space)
                beyond = np.array([1095.0, 1460.0])
                expected = dfs[-1] * np.exp(-terminal_forward * (beyond - 730.0) / 365.0)
                np.testing.assert_allclose(c.discount_factor(beyond), expected, rtol=1e-12)
                np.testing.assert_allclose(c.instantaneous_forward(beyond), terminal_forward, rtol=1e-12)
                self.assertAlmostEqual(c.instantaneous_forward(np.array([730.0]))[0], terminal_forward, places=12)

    def test_flat_forward_uses_exact_spline_end_slope(self):
        x = np.array([0.0, 365.0, 730.0, 1460.0])
        dfs = np.exp(-np.array([0.0, 0.03, 0.035, 0.045]) * x / 365.0)
        dfs[0] = 1.0
        c = ZeroCurve(x, dfs, interpolator=Cubic())
        end = c.instantaneous_forward(np.array([1460.0]))[0]
        just_beyond = c.instantaneous_forward(np.array([1460.5, 3000.0]))
        np.testing.assert_allclose(just_beyond, end, rtol=1e-12)

    def test_not_allowed_rejects_beyond_last_pillar(self):
        for space in CurveSpace:
            with self.subTest(space=space):
                c = _curve(space=space, extrapolation=RateExtrapolator.NotAllowed)
                np.testing.assert_allclose(c.discount_factor(np.array([1825.0])), [c.dfs[-1]])
                with self.assertRaises(ValueError):
                    c.discount_factor(np.array([1826.0]))

    def test_pre_origin_query_rejected(self):
        c = _curve()
        for method in (c.discount_factor, c.log_discount_factor, c.zero_rate, c.instantaneous_forward):
            with self.assertRaises(ValueError):
                method(np.array([-1.0]))


class TestZeroCurveRebinding(UnitTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve"]

    def test_with_dfs_keeps_configuration(self):
        c = _curve(0.04, space=CurveSpace.ZeroRate, interpolator=Quadratic(), extrapolation=RateExtrapolator.NotAllowed)
        bumped = c.with_dfs(_curve(0.05).dfs)
        self.assertIs(bumped.x, c.x)
        self.assertIs(bumped.space, c.space)
        self.assertEqual(bumped.interpolator, c.interpolator)
        self.assertIs(bumped.extrapolation, c.extrapolation)
        self.assertAlmostEqual(bumped.zero_rate(np.array([730.0]))[0], 0.05, places=10)
        self.assertAlmostEqual(c.zero_rate(np.array([730.0]))[0], 0.04, places=10)

    def test_with_dfs_matches_fresh_construction(self):
        c = _curve(interpolator=Cubic())
        fresh = _curve(0.05, interpolator=Cubic())
        xq = np.linspace(0.0, 2500.0, 50)
        np.testing.assert_allclose(c.with_dfs(fresh.dfs).discount_factor(xq), fresh.discount_factor(xq), rtol=1e-14)

    def test_with_dfs_validates(self):
        c = _curve()
        with self.assertRaises(ValueError):
            c.with_dfs(np.array([0.99, 0.9, 0.8]))
        with self.assertRaises(ValueError):
            c.with_dfs(np.array([1.0, 0.9]))


class TestZeroCurveValidation(UnitTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve"]

    def test_first_df_must_be_one(self):
        with self.assertRaises(ValueError):
            ZeroCurve(np.array([0.0, 365.0]), np.array([0.99, 0.95]))

    def test_origin_must_be_zero(self):
        with self.assertRaises(ValueError):
            ZeroCurve(np.array([1.0, 365.0]), np.array([1.0, 0.95]))

    def test_unsorted_or_duplicate_x_rejected(self):
        with self.assertRaises(ValueError):
            ZeroCurve(np.array([0.0, 365.0, 100.0]), np.array([1.0, 0.96, 0.99]))
        with self.assertRaises(ValueError):
            ZeroCurve(np.array([0.0, 365.0, 365.0]), np.array([1.0, 0.96, 0.96]))

    def test_nonpositive_or_nonfinite_df_rejected(self):
        for bad in (0.0, -0.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                ZeroCurve(np.array([0.0, 365.0]), np.array([1.0, bad]))

    def test_too_few_nodes_rejected(self):
        with self.assertRaises(ValueError):
            ZeroCurve(np.array([0.0]), np.array([1.0]))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ZeroCurve(np.array([0.0, 365.0]), np.array([1.0]))


class TestCurveInterpolatorParseTable(UnitTest):
    COVERAGE = ["finance.markets.curves.types"]

    def test_every_member_resolves(self):
        expected = {
            CurveInterpolator.LogLinearDF: (CurveSpace.LogDF, Linear()),
            CurveInterpolator.LogCubicDF: (CurveSpace.LogDF, Cubic()),
            CurveInterpolator.RateLinear: (CurveSpace.ZeroRate, Linear()),
            CurveInterpolator.RateQuadratic: (CurveSpace.ZeroRate, Quadratic()),
            CurveInterpolator.RateCubic: (CurveSpace.ZeroRate, Cubic()),
        }
        for member in CurveInterpolator:
            self.assertEqual(member.resolve(), expected[member])

    def test_parse_by_name(self):
        self.assertIs(CurveInterpolator["LogLinearDF"], CurveInterpolator.LogLinearDF)
