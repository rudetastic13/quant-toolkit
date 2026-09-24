"""Line1d: node validation, per-side extrapolation, derivative, and with_y rebinding."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from common.math.interpolation import Cubic, Flat, Linear, Quadratic
from common.math.line import Extrapolation, Line1d
from common.testing import UnitTest

X = np.array([0.0, 1.0, 2.0, 3.0])
Y = np.array([0.0, 1.0, 4.0, 9.0])  # y = x^2 at the nodes


def _line(
    interpolator=Linear(),
    left: Extrapolation = Extrapolation.NotAllowed,
    right: Extrapolation = Extrapolation.NotAllowed,
) -> Line1d:
    return Line1d(X, Y, interpolator, left=left, right=right)


class TestLine1dQueries(UnitTest):
    COVERAGE = ["common.math.line"]

    def test_every_scheme_passes_through_nodes(self):
        for interpolator in (Flat(), Linear(), Cubic(), Quadratic()):
            with self.subTest(interpolator=interpolator):
                np.testing.assert_allclose(_line(interpolator)(X), Y, atol=1e-12)

    def test_linear_matches_np_interp(self):
        xq = np.linspace(0.0, 3.0, 31)
        np.testing.assert_allclose(_line()(xq), np.interp(xq, X, Y))

    def test_get_value_is_an_alias_for_call(self):
        line = _line()
        xq = np.array([0.5, 2.5])
        np.testing.assert_array_equal(line.get_value(xq), line(xq))

    def test_default_interpolator_is_linear(self):
        line = Line1d(X, Y)
        np.testing.assert_allclose(line(np.array([1.5])), [2.5])

    def test_coefficients_come_from_the_interpolator(self):
        line = _line()
        self.assertEqual(line.coefficients.shape, (2, 3))
        np.testing.assert_allclose(line.coefficients[0], [1.0, 3.0, 5.0])  # interval slopes


class TestLine1dExtrapolation(UnitTest):
    COVERAGE = ["common.math.line"]

    def test_not_allowed_raises_on_either_side(self):
        line = _line()
        with self.assertRaises(ValueError):
            line(np.array([-0.5]))
        with self.assertRaises(ValueError):
            line(np.array([3.5]))

    def test_error_reports_count_and_furthest_point(self):
        line = _line()
        with self.assertRaisesRegex(ValueError, r"2 query point\(s\) beyond the last node x=3.0 \(furthest: 7.0\)"):
            line(np.array([4.0, 7.0]))
        with self.assertRaisesRegex(ValueError, r"1 query point\(s\) before the first node x=0.0 \(furthest: -2.0\)"):
            line(np.array([-2.0]))

    def test_flat_holds_end_values(self):
        line = _line(left=Extrapolation.Flat, right=Extrapolation.Flat)
        np.testing.assert_allclose(line(np.array([-5.0, 1.5, 10.0])), [0.0, 2.5, 9.0])

    def test_linear_continues_end_slopes(self):
        line = _line(left=Extrapolation.Linear, right=Extrapolation.Linear)
        # first interval slope 1, last interval slope 5
        np.testing.assert_allclose(line(np.array([-1.0, 4.0])), [-1.0, 14.0])

    def test_sides_are_independent(self):
        line = _line(left=Extrapolation.NotAllowed, right=Extrapolation.Flat)
        np.testing.assert_allclose(line(np.array([4.0])), [9.0])
        with self.assertRaises(ValueError):
            line(np.array([-1.0]))

    def test_cubic_linear_extrapolation_uses_spline_end_slope(self):
        line = _line(Cubic(), left=Extrapolation.Linear, right=Extrapolation.Linear)
        spline = CubicSpline(X, Y, bc_type="natural")
        expected_left = Y[0] + spline(0.0, 1) * (-1.0 - 0.0)
        expected_right = Y[-1] + spline(3.0, 1) * (4.0 - 3.0)
        np.testing.assert_allclose(line(np.array([-1.0, 4.0])), [expected_left, expected_right])

    def test_flat_interpolator_with_linear_extrapolation_is_flat(self):
        line = _line(Flat(), left=Extrapolation.Linear, right=Extrapolation.Linear)
        np.testing.assert_allclose(line(np.array([-3.0, 7.0])), [0.0, 9.0])

    def test_output_is_not_a_view_of_the_query(self):
        line = _line()
        xq = np.array([0.5, 1.5])
        out = line(xq)
        out[:] = -1.0
        np.testing.assert_array_equal(xq, [0.5, 1.5])


class TestLine1dDerivative(UnitTest):
    COVERAGE = ["common.math.line"]

    def test_linear_piecewise_slopes(self):
        np.testing.assert_allclose(_line().derivative(np.array([0.5, 1.5, 2.5])), [1.0, 3.0, 5.0])

    def test_flat_extrapolation_has_zero_slope(self):
        line = _line(left=Extrapolation.Flat, right=Extrapolation.Flat)
        np.testing.assert_allclose(line.derivative(np.array([-1.0, 4.0])), [0.0, 0.0])

    def test_linear_extrapolation_keeps_end_slope(self):
        line = _line(left=Extrapolation.Linear, right=Extrapolation.Linear)
        np.testing.assert_allclose(line.derivative(np.array([-1.0, 4.0])), [1.0, 5.0])

    def test_not_allowed_raises(self):
        with self.assertRaises(ValueError):
            _line().derivative(np.array([4.0]))

    def test_cubic_derivative_matches_scipy(self):
        xq = np.linspace(0.0, 3.0, 13)
        np.testing.assert_allclose(_line(Cubic()).derivative(xq), CubicSpline(X, Y, bc_type="natural")(xq, 1))


class TestLine1dRebinding(UnitTest):
    COVERAGE = ["common.math.line"]

    def test_with_y_keeps_geometry_and_configuration(self):
        line = _line(Cubic(), left=Extrapolation.Flat, right=Extrapolation.Linear)
        new = line.with_y(2.0 * Y)
        self.assertIs(new.x, line.x)
        self.assertIs(new.interpolator, line.interpolator)
        self.assertIs(new.left, line.left)
        self.assertIs(new.right, line.right)
        np.testing.assert_allclose(new(X), 2.0 * Y, atol=1e-12)
        np.testing.assert_allclose(new(np.array([1.5])), 2.0 * line(np.array([1.5])))

    def test_with_y_does_not_mutate_original(self):
        line = _line()
        _ = line.with_y(Y + 1.0)
        np.testing.assert_array_equal(line.y, Y)
        np.testing.assert_allclose(line(np.array([1.5])), [2.5])

    def test_with_y_validates_values(self):
        line = _line()
        with self.assertRaises(ValueError):
            line.with_y(np.array([0.0, 1.0]))
        with self.assertRaises(ValueError):
            line.with_y(np.array([0.0, np.nan, 4.0, 9.0]))


class TestLine1dValidation(UnitTest):
    COVERAGE = ["common.math.line"]

    def test_unsorted_or_duplicate_x_rejected(self):
        with self.assertRaises(ValueError):
            Line1d(np.array([0.0, 2.0, 1.0]), np.zeros(3))
        with self.assertRaises(ValueError):
            Line1d(np.array([0.0, 1.0, 1.0]), np.zeros(3))

    def test_non_finite_rejected(self):
        with self.assertRaises(ValueError):
            Line1d(np.array([0.0, np.nan]), np.zeros(2))
        with self.assertRaises(ValueError):
            Line1d(np.array([0.0, 1.0]), np.array([0.0, np.inf]))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            Line1d(np.array([0.0, 1.0, 2.0]), np.zeros(2))
        with self.assertRaises(ValueError):
            Line1d(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_single_node_only_with_flat(self):
        line = Line1d(np.array([0.0]), np.array([1.0]), Flat(), left=Extrapolation.Flat, right=Extrapolation.Flat)
        np.testing.assert_array_equal(line(np.array([-1.0, 0.0, 1.0])), [1.0, 1.0, 1.0])
        for interpolator in (Linear(), Cubic(), Quadratic()):
            with self.subTest(interpolator=interpolator), self.assertRaises(ValueError):
                Line1d(np.array([0.0]), np.array([1.0]), interpolator)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            Line1d(np.array([]), np.array([]), Flat())

    def test_repr_names_configuration(self):
        text = repr(_line(Cubic(), right=Extrapolation.Linear))
        self.assertIn("Cubic", text)
        self.assertIn("right=Linear", text)
