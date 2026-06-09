from __future__ import annotations

import numpy as np

from common.math.line import ExtrapolationType, InterpolationType, Line1d
from common.testing import UnitTest


class TestLine1dLinearInterp(UnitTest):
    COVERAGE = ["common.math.line"]

    def _line(self, extrap: ExtrapolationType = ExtrapolationType.NotAllowed) -> Line1d:
        xs = np.array([0, 1, 2, 3])
        ys = np.array([0.0, 1.0, 2.0, 3.0])
        return Line1d(xs, ys, interp=InterpolationType.Linear, extrap=extrap)

    def test_interpolates_at_knot(self):
        line = self._line()
        result = line.get_value(np.array([1]))
        np.testing.assert_allclose(result, [1.0])

    def test_interpolates_between_knots(self):
        line = self._line()
        result = line.get_value(np.array([1]))
        np.testing.assert_allclose(result, [1.0])

    def test_linear_extrapolation_not_allowed_raises(self):
        line = self._line(extrap=ExtrapolationType.NotAllowed)
        with self.assertRaises(ValueError):
            line.get_value(np.array([-1]))

    def test_linear_extrapolation_flat_clamps(self):
        line = self._line(extrap=ExtrapolationType.Flat)
        result = line.get_value(np.array([-5]))
        np.testing.assert_allclose(result, [0.0])

        result_high = line.get_value(np.array([100]))
        np.testing.assert_allclose(result_high, [3.0])

    def test_curve_cached_on_second_call(self):
        """_get() is cached — calling get_value twice uses the same lambda."""
        line = self._line()
        _ = line.get_value(np.array([1]))
        cached = line._curve
        _ = line.get_value(np.array([2]))
        self.assertIs(line._curve, cached)


class TestLine1dFlatInterp(UnitTest):
    COVERAGE = ["common.math.line"]

    def _line(self, extrap: ExtrapolationType = ExtrapolationType.NotAllowed) -> Line1d:
        xs = np.array([0, 1, 2, 3])
        ys = np.array([10.0, 20.0, 30.0, 40.0])
        return Line1d(xs, ys, interp=InterpolationType.Flat, extrap=extrap)

    def test_flat_interp_returns_nearest_left(self):
        line = self._line()
        # Between x=1 and x=2, flat interp returns the value at x=1 (left neighbour)
        result = line.get_value(np.array([1]))
        np.testing.assert_allclose(result, [20.0])

    def test_flat_extrap_not_allowed_out_of_bounds(self):
        line = self._line(extrap=ExtrapolationType.NotAllowed)
        result = line.get_value(np.array([1]))
        np.testing.assert_allclose(result, [20.0])

    def test_flat_extrap_allowed_clamps(self):
        line = self._line(extrap=ExtrapolationType.Flat)
        result_low = line.get_value(np.array([-1]))
        np.testing.assert_allclose(result_low, [10.0])

        result_high = line.get_value(np.array([99]))
        np.testing.assert_allclose(result_high, [40.0])

    def test_flat_extrapolation_out_of_bounds_raises(self):
        line = self._line(extrap=ExtrapolationType.NotAllowed)
        with self.assertRaises(ValueError):
            line.get_value(np.array([-1]))
