"""Interpolation schemes: each checked against an independent reference."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PPoly

from common.math.interpolation import BoundInterpolator, Cubic, Flat, Interpolator, Linear, Mixed, Quadratic
from common.testing import UnitTest

X = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
Y = np.array([1.0, 0.5, 2.0, 1.5, 3.0])


def _quadratic_reference(x: np.ndarray, y: np.ndarray, alpha: float, beta: float) -> PPoly:
    """The original loop-based blended-sweep implementation, kept verbatim as the oracle."""
    n = len(x)
    h = np.diff(x)
    d = np.diff(y) / h
    m_fwd = np.zeros(n)
    m_fwd[0] = alpha
    for i in range(n - 1):
        m_fwd[i + 1] = 2.0 * d[i] - m_fwd[i]
    m_bwd = np.zeros(n)
    m_bwd[-1] = beta
    for i in range(n - 2, -1, -1):
        m_bwd[i] = 2.0 * d[i] - m_bwd[i + 1]
    w = np.linspace(0.0, 1.0, n)
    m = (1.0 - w) * m_fwd + w * m_bwd
    m[0] = alpha
    m[-1] = beta
    c_arr = np.zeros((3, n - 1))
    for i in range(n - 1):
        c_arr[2, i] = y[i]
        c_arr[1, i] = m[i]
        c_arr[0, i] = (d[i] - m[i]) / h[i]
    return PPoly(c_arr, x)


class TestProtocols(UnitTest):
    COVERAGE = ["common.math.interpolation"]

    def test_schemes_satisfy_protocols(self):
        for scheme in (Flat(), Linear(), Cubic(), Quadratic(), Mixed(Linear(), Cubic(), 2)):
            with self.subTest(scheme=scheme):
                self.assertIsInstance(scheme, Interpolator)
                bound = scheme.fit(X, Y)
                self.assertIsInstance(bound, BoundInterpolator)
                np.testing.assert_array_equal(bound.x, X)

    def test_schemes_are_hashable_values(self):
        self.assertEqual(Cubic(), Cubic("natural"))
        self.assertEqual(hash(Quadratic(0.1, 0.2)), hash(Quadratic(0.1, 0.2)))
        self.assertNotEqual(Mixed(Linear(), Cubic(), 1), Mixed(Linear(), Cubic(), 2))


class TestFlat(UnitTest):
    COVERAGE = ["common.math.interpolation"]

    def test_holds_previous_value_between_nodes(self):
        # Fri | Sat | Sun | Mon : Friday's print covers the weekend, Monday's takes over on Monday
        fri, mon = 0.0, 3.0
        bound = Flat().fit(np.array([fri, mon]), np.array([5.30, 5.32]))
        np.testing.assert_allclose(bound(np.array([0.0, 1.0, 2.0, 3.0])), [5.30, 5.30, 5.30, 5.32])

    def test_last_node_returns_last_value(self):
        bound = Flat().fit(X, Y)
        np.testing.assert_allclose(bound(np.array([X[-1]])), [Y[-1]])

    def test_derivative_is_zero(self):
        np.testing.assert_array_equal(Flat().fit(X, Y).derivative(np.array([0.5, 3.0])), [0.0, 0.0])

    def test_coefficients_are_left_values(self):
        c = Flat().fit(X, Y).coefficients
        self.assertEqual(c.shape, (1, X.size - 1))
        np.testing.assert_array_equal(c[0], Y[:-1])


class TestLinear(UnitTest):
    COVERAGE = ["common.math.interpolation"]

    def test_matches_np_interp(self):
        xq = np.linspace(X[0], X[-1], 41)
        np.testing.assert_allclose(Linear().fit(X, Y)(xq), np.interp(xq, X, Y))

    def test_coefficients_and_derivative_are_interval_slopes(self):
        bound = Linear().fit(X, Y)
        slopes = np.diff(Y) / np.diff(X)
        self.assertEqual(bound.coefficients.shape, (2, X.size - 1))
        np.testing.assert_allclose(bound.coefficients[0], slopes)
        np.testing.assert_allclose(bound.coefficients[1], Y[:-1])
        mids = 0.5 * (X[:-1] + X[1:])
        np.testing.assert_allclose(bound.derivative(mids), slopes)


class TestCubic(UnitTest):
    COVERAGE = ["common.math.interpolation"]

    def test_natural_matches_scipy(self):
        xq = np.linspace(X[0], X[-1], 41)
        np.testing.assert_allclose(Cubic().fit(X, Y)(xq), CubicSpline(X, Y, bc_type="natural")(xq))

    def test_prescribed_second_derivatives_are_honoured(self):
        a, b = 0.7, -0.3
        c = Cubic(bc_type=((2, a), (2, b))).fit(X, Y).coefficients
        pp = PPoly(c, X)
        self.assertAlmostEqual(pp(X[0], nu=2), a, places=10)
        self.assertAlmostEqual(pp(X[-1], nu=2), b, places=10)

    def test_clamped_has_zero_end_slopes(self):
        bound = Cubic(bc_type="clamped").fit(X, Y)
        np.testing.assert_allclose(bound.derivative(np.array([X[0], X[-1]])), [0.0, 0.0], atol=1e-12)

    def test_coefficient_layout(self):
        self.assertEqual(Cubic().fit(X, Y).coefficients.shape, (4, X.size - 1))

    def test_two_nodes_degenerates_to_linear(self):
        bound = Cubic().fit(X[:2], Y[:2])
        np.testing.assert_allclose(bound(np.array([0.5])), [0.75])


class TestQuadratic(UnitTest):
    COVERAGE = ["common.math.interpolation"]

    def test_passes_through_nodes(self):
        np.testing.assert_allclose(Quadratic(0.2, -0.1).fit(X, Y)(X), Y, atol=1e-12)

    def test_matches_loop_reference(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            n = int(rng.integers(2, 12))
            x = np.cumsum(rng.uniform(0.2, 3.0, n))
            y = rng.normal(size=n)
            alpha, beta = rng.normal(size=2)
            xq = np.linspace(x[0], x[-1], 50)
            expected = _quadratic_reference(x, y, alpha, beta)(xq)
            np.testing.assert_allclose(Quadratic(alpha, beta).fit(x, y)(xq), expected, rtol=1e-11, atol=1e-11)

    def test_left_slope_is_honoured(self):
        bound = Quadratic(left_slope=0.35).fit(X, Y)
        self.assertAlmostEqual(bound.derivative(np.array([X[0]]))[0], 0.35, places=12)

    def test_coefficient_layout(self):
        self.assertEqual(Quadratic().fit(X, Y).coefficients.shape, (3, X.size - 1))


class TestMixed(UnitTest):
    COVERAGE = ["common.math.interpolation"]

    def test_passes_through_nodes(self):
        np.testing.assert_allclose(Mixed(Linear(), Cubic(), 2).fit(X, Y)(X), Y, atol=1e-12)

    def test_each_side_matches_its_scheme(self):
        k = 2
        bound = Mixed(Linear(), Cubic(), k).fit(X, Y)
        left_q = np.linspace(X[0], X[k], 11)
        right_q = np.linspace(X[k], X[-1], 11)
        np.testing.assert_allclose(bound(left_q), np.interp(left_q, X[: k + 1], Y[: k + 1]))
        np.testing.assert_allclose(bound(right_q), CubicSpline(X[k:], Y[k:], bc_type="natural")(right_q))

    def test_value_continuous_at_switch(self):
        bound = Mixed(Linear(), Cubic(), 2).fit(X, Y)
        eps = 1e-9
        before, after = bound(np.array([X[2] - eps, X[2] + eps]))
        self.assertAlmostEqual(before, after, places=7)

    def test_coefficients_padded_to_higher_order(self):
        c = Mixed(Linear(), Cubic(), 2).fit(X, Y).coefficients
        self.assertEqual(c.shape, (4, X.size - 1))
        np.testing.assert_array_equal(c[:2, :2], 0.0)  # linear pieces have no cubic/quadratic terms

    def test_switch_must_leave_two_nodes_each_side(self):
        for k in (0, X.size - 1, -1, X.size):
            with self.subTest(k=k), self.assertRaises(ValueError):
                Mixed(Linear(), Cubic(), k).fit(X, Y)
        Mixed(Linear(), Cubic(), 1).fit(X, Y)
        Mixed(Linear(), Cubic(), X.size - 2).fit(X, Y)

    def test_nested_mixed_is_allowed(self):
        inner = Mixed(Linear(), Quadratic(), 1)
        bound = Mixed(inner, Cubic(), 3).fit(X, Y)
        np.testing.assert_allclose(bound(X), Y, atol=1e-12)
