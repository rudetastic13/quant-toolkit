from __future__ import annotations

import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.markets.paths import FlatPath, InvertedPath, SinglePath


class TestFlatPath(UnitTest):
    COVERAGE = ["finance.markets.paths.single_path"]

    def test_get_value_returns_constant_array(self):
        path = FlatPath(name="SOFR", as_of_date=Date(2025, 1, 1), value=0.05)
        dates = np.array(["2025-03-01", "2025-06-01", "2025-09-01"], dtype="datetime64[D]")
        result = path.get_value(dates)
        np.testing.assert_array_equal(result, [0.05, 0.05, 0.05])

    def test_get_value_dtype_is_float64(self):
        path = FlatPath(name="FED", as_of_date=Date(2025, 1, 1), value=0.03)
        dates = np.array(["2025-06-01"], dtype="datetime64[D]")
        result = path.get_value(dates)
        self.assertEqual(result.dtype, np.float64)


class TestSinglePath(UnitTest):
    COVERAGE = ["finance.markets.paths.single_path"]

    def _make_path(self) -> SinglePath:
        dates = np.array(["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"], dtype="datetime64[D]")
        values = np.array([0.04, 0.045, 0.05, 0.055])
        return SinglePath(name="SOFR", as_of_date=Date(2025, 1, 1), dates=dates, values=values)

    def test_get_value_at_knot(self):
        path = self._make_path()
        result = path.get_value(np.array(["2025-01-01"], dtype="datetime64[D]"))
        self.assertAlmostEqual(float(result[0]), 0.04, places=10)

    def test_curve_is_initialized(self):
        path = self._make_path()
        self.assertIsNotNone(path.curve)


class TestInvertedPath(UnitTest):
    COVERAGE = ["finance.markets.paths.single_path"]

    def test_zero_values_become_one(self):
        dates = np.array(["2025-01-01", "2025-04-01"], dtype="datetime64[D]")
        values = np.array([0.0, 0.05])
        path = InvertedPath(name="test", as_of_date=Date(2025, 1, 1), dates=dates, values=values)
        # The 0.0 value should have been replaced with 1.0 to avoid division by zero
        self.assertAlmostEqual(float(path.values[0]), 1.0)

    def test_non_zero_values_unchanged(self):
        dates = np.array(["2025-01-01", "2025-04-01"], dtype="datetime64[D]")
        values = np.array([0.04, 0.05])
        path = InvertedPath(name="test", as_of_date=Date(2025, 1, 1), dates=dates, values=values)
        self.assertAlmostEqual(float(path.values[0]), 0.04)
        self.assertAlmostEqual(float(path.values[1]), 0.05)
