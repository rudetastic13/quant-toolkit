"""Reset/fixing grid generalization: aligned / intra-period / super-period × Advance/Arrears.

Cases are keyed off ``reset_frequency`` vs ``payment_frequency``:

- aligned (equal or unset): one fixing per period — accrual window (Arrears) or the
  previous window (Advance, with a synthetic pre-effective first window);
- intra-period (reset faster, e.g. pay Q / reset M): observation grid with sub-periods at
  the reset frequency (daily is the classic special case);
- super-period (reset slower, e.g. pay M / reset A): each period maps to its governing
  reset window via searchsorted — repeated windows across the periods it governs.
"""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date, Frequency, DayCountMethod, BDC
from finance.dates.enums import FixingType
from finance.instruments.schedules.payment_schedule import build_payment_schedule


def _build(**overrides):
    defaults = dict(
        effective=Date(2026, 1, 15),
        maturity=Date(2028, 1, 15),
        frequency=Frequency.Quarterly,
        day_count_method=DayCountMethod.Actual360,
        bdc=BDC.ModifiedFollowing,
        calendar="no_holidays",
    )
    defaults.update(overrides)
    return build_payment_schedule(**defaults)


class TestAlignedResetWindows(UnitTest):
    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

    def test_arrears_windows_equal_accrual_windows(self):
        ps = _build(fixing_type=FixingType.Arrears)
        np.testing.assert_array_equal(ps.reset_starts, ps.accrual_starts)
        np.testing.assert_array_equal(ps.reset_ends, ps.accrual_ends)

    def test_advance_windows_are_previous_accrual_windows(self):
        ps = _build(fixing_type=FixingType.Advance)
        # periods 1..N-1 read the previous accrual window
        np.testing.assert_array_equal(ps.reset_starts[1:], ps.accrual_starts[:-1])
        np.testing.assert_array_equal(ps.reset_ends[1:], ps.accrual_ends[:-1])
        # first period reads the synthetic pre-effective window ending at effective
        self.assertEqual(ps.reset_ends[0], ps.accrual_starts[0])
        self.assertLess(ps.reset_starts[0], ps.accrual_starts[0])

    def test_advance_first_window_is_one_payment_term_back(self):
        ps = _build(fixing_type=FixingType.Advance)
        # quarterly pay: pre-effective window starts ~3M before effective (2025-10-15)
        self.assertEqual(ps.reset_starts[0], np.datetime64("2025-10-15"))


class TestSuperPeriodResetWindows(UnitTest):
    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

    def _monthly_pay_annual_reset(self, fixing_type):
        return _build(
            frequency=Frequency.Monthly, reset_frequency=Frequency.Annually,
            fixing_type=fixing_type,
        )

    def test_arrears_periods_share_containing_annual_window(self):
        ps = self._monthly_pay_annual_reset(FixingType.Arrears)
        self.assertEqual(ps.n_periods, 24)
        # first 12 monthly periods all read year-1 [2026-01-15, 2027-01-15]
        np.testing.assert_array_equal(ps.reset_starts[:12], np.full(12, np.datetime64("2026-01-15")))
        np.testing.assert_array_equal(ps.reset_ends[:12], np.full(12, np.datetime64("2027-01-15")))
        # next 12 read year-2 (2028-01-15 is a Saturday -> MF rolls the boundary to the 17th)
        np.testing.assert_array_equal(ps.reset_starts[12:], np.full(12, np.datetime64("2027-01-15")))
        np.testing.assert_array_equal(ps.reset_ends[12:], np.full(12, np.datetime64("2028-01-17")))
        # every accrual start falls inside its governing window
        self.assertTrue((ps.reset_starts <= ps.accrual_starts).all())
        self.assertTrue((ps.accrual_starts < ps.reset_ends).all())

    def test_advance_periods_read_previous_annual_window(self):
        ps = self._monthly_pay_annual_reset(FixingType.Advance)
        # first year reads the synthetic pre-effective window [2025-01-15, 2026-01-15]
        np.testing.assert_array_equal(ps.reset_starts[:12], np.full(12, np.datetime64("2025-01-15")))
        np.testing.assert_array_equal(ps.reset_ends[:12], np.full(12, np.datetime64("2026-01-15")))
        # second year reads year-1
        np.testing.assert_array_equal(ps.reset_starts[12:], np.full(12, np.datetime64("2026-01-15")))
        np.testing.assert_array_equal(ps.reset_ends[12:], np.full(12, np.datetime64("2027-01-15")))
        # windows are always fully known at (or before) each period start
        self.assertTrue((ps.reset_ends <= ps.accrual_starts).all())


class TestIntraPeriodObservationGrid(UnitTest):
    COVERAGE = [
        "finance.instruments.schedules.payment_schedule",
        "finance.instruments.schedules.observation",
    ]

    def test_monthly_resets_in_quarterly_periods(self):
        ps = _build(reset_frequency=Frequency.Monthly, build_observations=True)
        self.assertEqual(ps.n_periods, 8)
        counts = np.diff(np.append(ps.obs_offsets, ps.obs_weights.shape[0]))
        # ~3 monthly fixings per quarter, NOT ~63 daily ones
        self.assertTrue((counts == 3).all(), counts)
        # each period's sub-period weights tile the accrual period
        for p in range(ps.n_periods):
            s, e = ps.obs_offsets[p], (ps.obs_offsets[p + 1] if p + 1 < ps.n_periods else None)
            w = ps.obs_weights[s:e]
            self.assertAlmostEqual(w.sum(), ps.period_fracs[p], places=12)

    def test_daily_grid_unchanged_for_daily_reset(self):
        daily_none = _build(build_observations=True)  # no reset_frequency -> aligned windows + obs
        daily_explicit = _build(reset_frequency=Frequency.Daily, build_observations=True)
        np.testing.assert_array_equal(daily_explicit.obs_read_starts, np.asarray(daily_explicit.obs_read_starts))
        counts = np.diff(np.append(daily_explicit.obs_offsets, daily_explicit.obs_weights.shape[0]))
        self.assertGreater(counts.min(), 50)  # business-day granularity inside a quarter
        # aligned (no reset_frequency) with observations collapses to one fixing per period
        counts_aligned = np.diff(np.append(daily_none.obs_offsets, daily_none.obs_weights.shape[0]))
        self.assertTrue((counts_aligned == 1).all())

    def test_lookback_applies_to_intra_period_fixings(self):
        plain = _build(reset_frequency=Frequency.Monthly, build_observations=True)
        shifted = _build(reset_frequency=Frequency.Monthly, build_observations=True, rate_lookback=5)
        self.assertTrue((shifted.obs_read_starts < plain.obs_read_starts).all())
        np.testing.assert_array_equal(shifted.obs_weights, plain.obs_weights)  # ISDA: weights stay real

    def test_advance_intra_period_is_designed_seam(self):
        with self.assertRaises(NotImplementedError):
            _build(reset_frequency=Frequency.Monthly, build_observations=True,
                   fixing_type=FixingType.Advance)


class TestSuperPeriodObservationGrid(UnitTest):
    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

    def test_slower_reset_collapses_to_one_governing_fixing_per_period(self):
        ps = _build(frequency=Frequency.Monthly, reset_frequency=Frequency.Annually,
                    build_observations=True)
        counts = np.diff(np.append(ps.obs_offsets, ps.obs_weights.shape[0]))
        self.assertTrue((counts == 1).all())
        np.testing.assert_array_equal(ps.obs_read_starts, ps.reset_starts)
        np.testing.assert_array_equal(ps.obs_read_ends, ps.reset_ends)
