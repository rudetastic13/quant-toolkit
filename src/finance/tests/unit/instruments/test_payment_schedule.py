"""Tests for PaymentSchedule and build_payment_schedule."""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date, Frequency, DayCountMethod, BDC
from finance.instruments.schedules import PaymentSchedule, build_payment_schedule
from finance.instruments.schedules.payment_schedule import adjust_schedule_boundaries


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

    def test_reset_windows_default_to_accrual_windows(self):
        # no reset_frequency (aligned) + Arrears -> governing window IS the accrual window
        ps = self._build_5y_semi()
        np.testing.assert_array_equal(ps.reset_starts, ps.accrual_starts)
        np.testing.assert_array_equal(ps.reset_ends, ps.accrual_ends)

    def test_notional_schedule_initially_none(self):
        ps = self._build_5y_semi()
        self.assertIsNone(ps.notional_schedule)


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

    def test_faster_reset_without_observations_raises(self):
        # multiple fixings per period is observation-grid territory, not single-fixing float
        with self.assertRaises(ValueError):
            build_payment_schedule(
                effective=Date(2025, 1, 15),
                maturity=Date(2026, 1, 15),
                frequency=Frequency.Quarterly,
                day_count_method=DayCountMethod.Actual360,
                bdc=BDC.NoAdjustment,
                calendar="no_holidays",
                reset_frequency=Frequency.Monthly,
            )

    def test_same_reset_frequency_gives_accrual_windows(self):
        ps = build_payment_schedule(
            effective=Date(2025, 1, 15),
            maturity=Date(2026, 1, 15),
            frequency=Frequency.Quarterly,
            day_count_method=DayCountMethod.Actual360,
            bdc=BDC.NoAdjustment,
            calendar="no_holidays",
            reset_frequency=Frequency.Quarterly,
        )
        np.testing.assert_array_equal(ps.reset_starts, ps.accrual_starts)
        np.testing.assert_array_equal(ps.reset_ends, ps.accrual_ends)


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


class TestAccrualAdjustment(UnitTest):
    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

    # effective is a Saturday, maturity a Sunday, with an interior Sunday roll date.
    EFFECTIVE = Date(2025, 3, 15)   # Sat
    MATURITY = Date(2026, 3, 15)    # Sun

    def _build(self, **overrides):
        defaults = dict(
            effective=self.EFFECTIVE,
            maturity=self.MATURITY,
            frequency=Frequency.Quarterly,
            day_count_method=DayCountMethod.Actual360,
            bdc=BDC.ModifiedFollowing,
            calendar="no_holidays",
        )
        defaults.update(overrides)
        return build_payment_schedule(**defaults)

    def test_endpoints_adjusted_by_default(self):
        ps = self._build()  # adjust_endpoints=True default
        # all accrual boundaries land on good business days
        self.assertTrue(np.all(np.is_busday(ps.accrual_starts)))
        self.assertTrue(np.all(np.is_busday(ps.accrual_ends)))
        # Sat 2025-03-15 -> Mon 2025-03-17
        self.assertEqual(ps.accrual_starts[0], np.datetime64("2025-03-17"))

    def test_endpoints_flow_when_disabled(self):
        ps = self._build(adjust_endpoints=False)
        # explicit endpoints kept raw even though they are weekends
        self.assertEqual(ps.accrual_starts[0], np.datetime64("2025-03-15"))
        self.assertEqual(ps.accrual_ends[-1], np.datetime64("2026-03-15"))
        self.assertFalse(np.is_busday(ps.accrual_starts[0]))
        # ...but interior generated roll dates are still adjusted (Sun 06-15 -> Mon 06-16)
        self.assertEqual(ps.accrual_ends[0], np.datetime64("2025-06-16"))

    def test_no_adjustment_bdc_is_noop(self):
        ps = self._build(bdc=BDC.NoAdjustment)
        self.assertEqual(ps.accrual_starts[0], np.datetime64("2025-03-15"))
        self.assertEqual(ps.accrual_ends[0], np.datetime64("2025-06-15"))

    def test_adjust_schedule_boundaries_protects_only_listed(self):
        raw = np.array(["2025-03-15", "2025-06-15", "2026-03-15"], dtype="datetime64[D]")  # Sat, Sun, Sun
        protected = {np.datetime64("2025-03-15"), np.datetime64("2026-03-15")}
        out = adjust_schedule_boundaries(
            raw, BDC.ModifiedFollowing, "no_holidays", protected=protected, adjust_endpoints=False
        )
        # protected endpoints stay raw; the unprotected interior Sunday rolls to Monday
        self.assertEqual(out[0], np.datetime64("2025-03-15"))
        self.assertEqual(out[1], np.datetime64("2025-06-16"))
        self.assertEqual(out[2], np.datetime64("2026-03-15"))
