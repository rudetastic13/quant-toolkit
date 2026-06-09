from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from common.testing import UnitTest
from finance.coupons.calculators import (
    calculate_arithmetic_average,
    calculate_custom,
    calculate_fixed,
    calculate_floating,
    calculate_geometric_average,
)
from finance.dates import Date
from finance.instruments.enums import MarginTreatment
from finance.instruments.schedules.coupon_schedule import CouponSchedule, FixedCouponEvent, FloatingCouponEvent


class TestCalculateFixed(UnitTest):
    COVERAGE = ["finance.coupons.calculators"]

    def test_fills_array_with_rate(self):
        out = np.zeros(5, dtype=np.float64)
        result = calculate_fixed(0.05, out)
        np.testing.assert_array_equal(result, [0.05] * 5)

    def test_returns_same_array(self):
        out = np.zeros(3, dtype=np.float64)
        result = calculate_fixed(0.03, out)
        self.assertIs(result, out)

    def test_overwrites_existing_values(self):
        out = np.ones(4, dtype=np.float64)
        calculate_fixed(0.07, out)
        np.testing.assert_array_equal(out, [0.07] * 4)


class TestCalculateFloating(UnitTest):
    COVERAGE = ["finance.coupons.calculators"]

    def _mock_market(self, rates: np.ndarray) -> MagicMock:
        mkt = MagicMock()
        mkt.get_rates.return_value = rates.copy()
        return mkt

    def test_plain_float_no_bounds(self):
        """No index_floor/cap/floor — result equals market rates plus spread."""
        rates = np.array([0.04, 0.05, 0.06])
        mkt = self._mock_market(rates)
        out = np.zeros_like(rates)
        result = calculate_floating("USD SOFR 1M", spread=0.001, index_floor=None, cap=None, floor=None,
                                    market=mkt, reset_dates=rates, out=out)
        np.testing.assert_allclose(result, rates + 0.001)

    def test_index_floor_applied_before_spread(self):
        """index_floor clips the index rate before spread is added."""
        rates = np.array([-0.01, 0.02, 0.05])
        mkt = self._mock_market(rates)
        out = np.zeros_like(rates)
        result = calculate_floating("USD SOFR 1M", spread=0.005, index_floor=0.0, cap=None, floor=None,
                                    market=mkt, reset_dates=rates, out=out)
        np.testing.assert_allclose(result, [0.005, 0.025, 0.055])

    def test_floor_on_final_rate(self):
        rates = np.array([0.001, 0.01, 0.05])
        mkt = self._mock_market(rates)
        out = np.zeros_like(rates)
        result = calculate_floating("USD SOFR 1M", spread=0.0, index_floor=None, cap=None, floor=0.02,
                                    market=mkt, reset_dates=rates, out=out)
        np.testing.assert_allclose(result, [0.02, 0.02, 0.05])

    def test_cap_on_final_rate(self):
        rates = np.array([0.02, 0.05, 0.08])
        mkt = self._mock_market(rates)
        out = np.zeros_like(rates)
        result = calculate_floating("USD SOFR 1M", spread=0.0, index_floor=None, cap=0.06, floor=None,
                                    market=mkt, reset_dates=rates, out=out)
        np.testing.assert_allclose(result, [0.02, 0.05, 0.06])


class TestCalculateCustom(UnitTest):
    COVERAGE = ["finance.coupons.calculators"]

    def test_fixed_event_fills_out_with_coupon_rate(self):
        """calculate_custom with a single FixedCouponEvent fills the output with the rate."""
        event = FixedCouponEvent(start_date=Date(2025, 1, 1), coupon_rate=0.05)
        schedule = CouponSchedule(events=[event])
        accrual_grid = np.array(["2025-01-01", "2025-04-01", "2025-07-01"], dtype="datetime64[D]")
        out = np.zeros(3, dtype=np.float64)
        result = calculate_custom(schedule, accrual_grid, out)
        np.testing.assert_allclose(result, [0.05, 0.05, 0.05])

    def test_floating_event_dispatches_to_market(self):
        """calculate_custom with a FloatingCouponEvent calls the market for rates."""
        mock_mkt = MagicMock()
        mock_mkt.get_rates.return_value = np.array([0.04, 0.045, 0.05])
        event = FloatingCouponEvent(start_date=Date(2025, 1, 1), rate_index="USD SOFR 1M", spread=0.0)
        schedule = CouponSchedule(events=[event])
        accrual_grid = np.array(["2025-01-01", "2025-04-01", "2025-07-01"], dtype="datetime64[D]")
        reset_dates = np.array(["2025-01-01", "2025-04-01", "2025-07-01"], dtype="datetime64[D]")
        out = np.zeros(3, dtype=np.float64)
        result = calculate_custom(schedule, accrual_grid, out, market=mock_mkt, reset_dates=reset_dates)
        self.assertEqual(result.shape, (3,))


class TestCalculateGeometricAverage(UnitTest):
    COVERAGE = ["finance.coupons.calculators"]

    def _mock_market(self, daily_rates: np.ndarray) -> MagicMock:
        mkt = MagicMock()
        mkt.get_rates.return_value = daily_rates.copy()
        return mkt

    def test_basic_geo_avg_exclusive_margin(self):
        """Geometric average with no floor/cap, exclusive spread."""
        n = 3
        daily_rates = np.full(n, 0.04)
        weights = np.full(n, 1.0 / 360.0)
        mkt = self._mock_market(daily_rates)
        out = np.zeros(1, dtype=np.float64)
        result = calculate_geometric_average(
            rate_index="USD SOFR",
            spread=0.001,
            index_floor=None,
            cap=None,
            floor=None,
            margin_treatment=MarginTreatment.Exclusive,
            market=mkt,
            fixing_dates=np.arange(n, dtype=np.float64),
            rate_weights=weights,
            out=out,
        )
        self.assertEqual(result.shape, (1,))

    def test_geo_avg_inclusive_margin(self):
        n = 3
        daily_rates = np.full(n, 0.04)
        weights = np.full(n, 1.0 / 360.0)
        mkt = self._mock_market(daily_rates)
        out = np.zeros(1, dtype=np.float64)
        result = calculate_geometric_average(
            rate_index="USD SOFR",
            spread=0.001,
            index_floor=0.0,
            cap=0.10,
            floor=0.0,
            margin_treatment=MarginTreatment.Inclusive,
            market=mkt,
            fixing_dates=np.arange(n, dtype=np.float64),
            rate_weights=weights,
            out=out,
        )
        self.assertEqual(result.shape, (1,))

    def test_geo_avg_index_floor_applied(self):
        n = 2
        daily_rates = np.array([-0.01, 0.05])
        weights = np.full(n, 1.0 / 360.0)
        mkt = self._mock_market(daily_rates)
        out = np.zeros(1, dtype=np.float64)
        calculate_geometric_average(
            rate_index="USD SOFR",
            spread=0.0,
            index_floor=0.001,  # truthy non-zero floor
            cap=None,
            floor=None,
            margin_treatment=MarginTreatment.Exclusive,
            market=mkt,
            fixing_dates=np.arange(n, dtype=np.float64),
            rate_weights=weights,
            out=out,
        )
        # Should not raise; index_floor prevented negative rates entering the product

    def test_geo_avg_with_floor_and_cap(self):
        n = 2
        daily_rates = np.full(n, 0.04)
        weights = np.full(n, 1.0 / 360.0)
        mkt = self._mock_market(daily_rates)
        out = np.zeros(1, dtype=np.float64)
        calculate_geometric_average(
            rate_index="USD SOFR",
            spread=0.0,
            index_floor=None,
            cap=0.10,
            floor=0.01,   # truthy non-zero floor
            margin_treatment=MarginTreatment.Exclusive,
            market=mkt,
            fixing_dates=np.arange(n, dtype=np.float64),
            rate_weights=weights,
            out=out,
        )
        # Should produce a value in [floor, cap]
        self.assertGreaterEqual(float(out[0]), 0.01)


class TestCalculateArithmeticAverage(UnitTest):
    COVERAGE = ["finance.coupons.calculators"]

    def _mock_market(self, daily_rates: np.ndarray) -> MagicMock:
        mkt = MagicMock()
        mkt.get_rates.return_value = daily_rates.copy()
        return mkt

    def test_basic_arith_avg_exclusive_margin(self):
        n = 3
        daily_rates = np.full(n, 0.04)
        weights = np.full(n, 1.0 / 360.0)
        mkt = self._mock_market(daily_rates)
        out = np.zeros(1, dtype=np.float64)
        result = calculate_arithmetic_average(
            rate_index="USD SOFR",
            spread=0.001,
            index_floor=None,
            cap=None,
            floor=None,
            margin_treatment=MarginTreatment.Exclusive,
            market=mkt,
            fixing_dates=np.arange(n, dtype=np.float64),
            rate_weights=weights,
            out=out,
        )
        self.assertEqual(result.shape, (1,))

    def test_arith_avg_inclusive_margin_with_bounds(self):
        n = 3
        daily_rates = np.full(n, 0.04)
        weights = np.full(n, 1.0 / 360.0)
        mkt = self._mock_market(daily_rates)
        out = np.zeros(1, dtype=np.float64)
        result = calculate_arithmetic_average(
            rate_index="USD SOFR",
            spread=0.001,
            index_floor=0.001,   # non-zero so the index_floor branch fires
            cap=0.10,
            floor=0.005,          # non-zero so the floor branch fires
            margin_treatment=MarginTreatment.Inclusive,
            market=mkt,
            fixing_dates=np.arange(n, dtype=np.float64),
            rate_weights=weights,
            out=out,
        )
        self.assertEqual(result.shape, (1,))
