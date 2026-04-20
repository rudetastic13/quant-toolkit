"""Tests for PaymentSchedule and build_payment_schedule."""
import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date, Frequency, DayCountMethod, BDC, Roll, Direction, Term, TermType
from finance.instruments.schedules import PaymentSchedule, build_payment_schedule


class TestBuildPaymentSchedule(UnitTest):
    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

    def _build_5y_semi(self, **overrides) -> PaymentSchedule:
        defaults = dict(
            effective=Date(2025, 1, 15),
            maturity=Date(2030, 1, 15),
            frequency=Frequency.SemiAnnually,
            day_count_method=DayCountMethod.Thirty360,
            bdc=BDC.NoAdjustment,
            calendar="no_holidays",
        )
        defaults.update(overrides)
        return build_payment_schedule(**defaults)

    def test_period_count(self):
        ps = self._build_5y_semi()
        self.assertEqual(ps.n_periods, 10)

    def test_accrual_starts_first_equals_effective(self):
        ps = self._build_5y_semi()
        self.assertEqual(
            ps.accrual_starts[0],
            np.datetime64("2025-01-15"),
        )

    def test_accrual_ends_last_equals_maturity(self):
        ps = self._build_5y_semi()
        self.assertEqual(
            ps.accrual_ends[-1],
            np.datetime64("2030-01-15"),
        )

    def test_accrual_starts_ends_contiguous(self):
        ps = self._build_5y_semi()
        np.testing.assert_array_equal(ps.accrual_starts[1:], ps.accrual_ends[:-1])

    def test_period_fracs_thirty_360(self):
        ps = self._build_5y_semi()
        np.testing.assert_array_almost_equal(ps.period_fracs, np.full(10, 0.5))

    def test_payment_dates_equal_accrual_ends_no_adjustment(self):
        ps = self._build_5y_semi()
        np.testing.assert_array_equal(ps.payment_dates, ps.accrual_ends)

    def test_reset_dates_none_when_no_reset_frequency(self):
        ps = self._build_5y_semi()
        self.assertIsNone(ps.reset_dates)

    def test_notional_schedule_initially_none(self):
        ps = self._build_5y_semi()
        self.assertIsNone(ps.notional_schedule)

    def test_fixing_dates_initially_none(self):
        ps = self._build_5y_semi()
        self.assertIsNone(ps.fixing_dates)


class TestBuildPaymentScheduleQuarterly(UnitTest):
    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

    def test_quarterly_1y(self):
        ps = build_payment_schedule(
            effective=Date(2025, 3, 15),
            maturity=Date(2026, 3, 15),
            frequency=Frequency.Quarterly,
            day_count_method=DayCountMethod.Actual360,
            bdc=BDC.NoAdjustment,
            calendar="no_holidays",
        )
        self.assertEqual(ps.n_periods, 4)

    def test_monthly_1y(self):
        ps = build_payment_schedule(
            effective=Date(2025, 1, 1),
            maturity=Date(2026, 1, 1),
            frequency=Frequency.Monthly,
            day_count_method=DayCountMethod.Actual365,
            bdc=BDC.NoAdjustment,
            calendar="no_holidays",
        )
        self.assertEqual(ps.n_periods, 12)


class TestBuildPaymentScheduleResetFrequency(UnitTest):
    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

    def test_different_reset_frequency_produces_reset_dates(self):
        ps = build_payment_schedule(
            effective=Date(2025, 1, 15),
            maturity=Date(2026, 1, 15),
            frequency=Frequency.Quarterly,
            day_count_method=DayCountMethod.Actual360,
            bdc=BDC.NoAdjustment,
            calendar="no_holidays",
            reset_frequency=Frequency.Monthly,
        )
        self.assertIsNotNone(ps.reset_dates)
        # Monthly resets over 1Y = 12 reset dates
        self.assertEqual(len(ps.reset_dates), 12)

    def test_same_reset_frequency_produces_no_reset_dates(self):
        ps = build_payment_schedule(
            effective=Date(2025, 1, 15),
            maturity=Date(2026, 1, 15),
            frequency=Frequency.Quarterly,
            day_count_method=DayCountMethod.Actual360,
            bdc=BDC.NoAdjustment,
            calendar="no_holidays",
            reset_frequency=Frequency.Quarterly,
        )
        self.assertIsNone(ps.reset_dates)


class TestBuildPaymentScheduleStubDates(UnitTest):
    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

    def test_front_stub(self):
        ps = build_payment_schedule(
            effective=Date(2025, 2, 1),
            maturity=Date(2026, 1, 15),
            frequency=Frequency.Quarterly,
            day_count_method=DayCountMethod.Actual360,
            bdc=BDC.NoAdjustment,
            calendar="no_holidays",
            first_regular=Date(2025, 4, 15),
        )
        # First period is a short stub: Feb 1 → Apr 15
        self.assertEqual(ps.accrual_starts[0], np.datetime64("2025-02-01"))
        self.assertEqual(ps.accrual_ends[0], np.datetime64("2025-04-15"))
        # Stub fraction should be different from regular periods
        self.assertNotAlmostEqual(ps.period_fracs[0], ps.period_fracs[1], places=4)
