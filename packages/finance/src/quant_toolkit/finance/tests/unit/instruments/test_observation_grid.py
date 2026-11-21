"""Observation grid: continuous-boundary + searchsorted construction of the Tier-2 grid."""
import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.dates import Calendars, Date, DayCountMethod, Frequency, BDC, period_fractions
from quant_toolkit.finance.dates.calendars.calendar import StaticCalendar
from quant_toolkit.finance.instruments.schedules.observation import build_observation_grid, LookbackStyle
from quant_toolkit.finance.instruments.schedules.payment_schedule import build_payment_schedule

ACT360 = DayCountMethod.Actual360
CAL = "no_holidays"

MEMO_CAL = "test_isda_memo"


def _register_memo_calendar(test: UnitTest) -> None:
    """The ISDA memo's calendar (weekends plus an assumed holiday on 2 Dec 2020), for one test."""
    Calendars().register(StaticCalendar(name=MEMO_CAL, week_mask=[1, 1, 1, 1, 1, 0, 0], holidays={Date(2020, 12, 2)}))
    test.addCleanup(Calendars._calendars.pop, MEMO_CAL, None)


def _d(*dates: str) -> np.ndarray:
    return np.array(dates, dtype="datetime64[D]")


def _semi_annual():
    # 1y, two semiannual accrual periods
    accrual_starts = np.array(["2026-06-01", "2026-12-01"], dtype="datetime64[D]")
    accrual_ends = np.array(["2026-12-01", "2027-06-01"], dtype="datetime64[D]")
    return accrual_starts, accrual_ends


class TestObservationGridStructure(UnitTest):

    def setUp(self):
        self.starts, self.ends = _semi_annual()
        self.grid = build_observation_grid(self.starts, self.ends, CAL, ACT360)

    def test_offsets_well_formed(self):
        off = self.grid.offsets
        self.assertEqual(off[0], 0)
        self.assertTrue(np.all(np.diff(off) > 0))      # strictly increasing
        self.assertEqual(len(off), len(self.ends))     # one per accrual period

    def test_subperiods_tile_exactly(self):
        # no gaps / overlaps: each interest_end equals the next interest_start; spans [effective, maturity]
        np.testing.assert_array_equal(self.grid.interest_ends[:-1], self.grid.interest_starts[1:])
        self.assertEqual(self.grid.interest_starts[0], self.starts[0])
        self.assertEqual(self.grid.interest_ends[-1], self.ends[-1])

    def test_each_subperiod_in_one_accrual_period(self):
        # period membership via searchsorted; interest_start in [accrual_start, accrual_end)
        period_idx = np.searchsorted(self.ends, self.grid.interest_starts, side="right")
        for k, p in enumerate(period_idx):
            self.assertGreaterEqual(self.grid.interest_starts[k], self.starts[p])
            self.assertLess(self.grid.interest_starts[k], self.ends[p])

    def test_weights_sum_to_period_fraction(self):
        # Act/360 is additive over contiguous sub-periods -> per-period weights sum exactly
        off = np.append(self.grid.offsets, self.grid.n_obs)
        period_fracs = period_fractions(ACT360, self.starts, self.ends)
        for i in range(len(self.ends)):
            seg = self.grid.weights[off[i] : off[i + 1]]
            self.assertAlmostEqual(seg.sum(), period_fracs[i], places=12)


class TestLookbackLockout(UnitTest):

    def setUp(self):
        self.starts, self.ends = _semi_annual()

    def test_isda_lookback_shifts_rate_keeps_real_weight(self):
        # ISDA Lookback (default): weight from the real sub-period (unchanged); read start
        # shifted back; read end is the overnight next-business-day of the shifted start.
        base = build_observation_grid(self.starts, self.ends, CAL, ACT360, lookback=0)
        lb = build_observation_grid(self.starts, self.ends, CAL, ACT360, lookback=2)
        np.testing.assert_array_equal(base.weights, lb.weights)  # weight basis = real period
        np.testing.assert_array_equal(lb.fixing_starts, np.busday_offset(base.interest_starts, -2, roll="preceding"))
        np.testing.assert_array_equal(lb.fixing_ends, np.busday_offset(lb.fixing_starts, 1, roll="following"))

    def test_observation_shift_shifts_both_window_and_weight(self):
        base = build_observation_grid(self.starts, self.ends, CAL, ACT360, lookback=0)
        os = build_observation_grid(
            self.starts, self.ends, CAL, ACT360, lookback=2, lookback_style=LookbackStyle.ObservationShift
        )
        # both ends shifted; weight comes from the shifted window (so it differs from real)
        np.testing.assert_array_equal(os.fixing_starts, np.busday_offset(base.interest_starts, -2, roll="preceding"))
        np.testing.assert_array_equal(os.fixing_ends, np.busday_offset(base.interest_ends, -2, roll="preceding"))
        self.assertFalse(np.array_equal(os.weights, base.weights))

    def test_lockout_freezes_tail(self):
        lk = build_observation_grid(self.starts, self.ends, CAL, ACT360, lockout=3)
        # the cutoff for period 0 is accrual_end[0] - 3 BD; tail obs are clamped there
        cutoff0 = np.busday_offset(self.ends[0], -3, roll="preceding")
        period_idx = np.searchsorted(self.ends, lk.interest_starts, side="right")
        tail0 = (period_idx == 0) & (lk.interest_starts > cutoff0)
        self.assertTrue(tail0.any())
        np.testing.assert_array_equal(lk.fixing_starts[tail0], np.full(tail0.sum(), cutoff0))


class TestIsdaMemoVectors(UnitTest):
    """Worked examples from the ISDA memo on compounding RFRs under the 2006 Definitions.

    Calculation Period 22 Sep 2020 to 21 Dec 2020 inclusive (Period End Date 22 Dec), five
    business days for each method, 2 Dec 2020 a holiday.
    """


    def setUp(self):
        _register_memo_calendar(self)
        self.starts, self.ends = _d("2020-09-22"), _d("2020-12-22")

    def _row(self, grid, day: str) -> int:
        return int(np.flatnonzero(grid.interest_starts == np.datetime64(day))[0])

    def test_ois_base_case(self):
        grid = build_observation_grid(self.starts, self.ends, MEMO_CAL, ACT360)
        self.assertEqual(grid.n_obs, 64)
        self.assertAlmostEqual(grid.weights.sum() * 360, 91.0)
        np.testing.assert_array_equal(grid.fixing_starts, grid.interest_starts)

    def test_lookback(self):
        grid = build_observation_grid(self.starts, self.ends, MEMO_CAL, ACT360, lookback=5)
        i = self._row(grid, "2020-12-01")
        self.assertAlmostEqual(grid.weights[i] * 360, 2.0)  # 2 Dec is a holiday
        self.assertEqual(grid.fixing_starts[i], np.datetime64("2020-11-24"))
        self.assertEqual((grid.fixing_starts[0], grid.fixing_starts[-1]), (np.datetime64("2020-09-15"), np.datetime64("2020-12-14")))

    def test_observation_period_shift(self):
        grid = build_observation_grid(
            self.starts, self.ends, MEMO_CAL, ACT360, lookback=5, lookback_style=LookbackStyle.ObservationShift
        )
        self.assertEqual((grid.fixing_starts[0], grid.fixing_ends[-1]), (np.datetime64("2020-09-15"), np.datetime64("2020-12-15")))
        i = int(np.flatnonzero(grid.fixing_starts == np.datetime64("2020-12-01"))[0])
        self.assertAlmostEqual(grid.weights[i] * 360, 2.0)  # the fixing FOR 1 Dec carries the holiday
        self.assertAlmostEqual(grid.weights.sum() * 360, 91.0)  # d = Observation Period days

    def test_lockout(self):
        base = build_observation_grid(self.starts, self.ends, MEMO_CAL, ACT360)
        grid = build_observation_grid(self.starts, self.ends, MEMO_CAL, ACT360, lockout=5)
        used = grid.fixing_starts == np.datetime64("2020-12-15")  # the Lockout Date
        np.testing.assert_array_equal(grid.interest_starts[used], _d("2020-12-15", "2020-12-16", "2020-12-17", "2020-12-18", "2020-12-21"))
        np.testing.assert_array_equal(grid.weights, base.weights)  # lockout never changes the weights
        np.testing.assert_array_equal(grid.fixing_starts[~used], base.fixing_starts[~used])

    def test_a_lockout_of_one_freezes_nothing(self):
        base = build_observation_grid(self.starts, self.ends, MEMO_CAL, ACT360)
        grid = build_observation_grid(self.starts, self.ends, MEMO_CAL, ACT360, lockout=1)
        np.testing.assert_array_equal(grid.fixing_starts, base.fixing_starts)


class TestNonBusinessDayBoundaries(UnitTest):
    """A period running Saturday 1 Feb 2025 to Saturday 1 Mar 2025 (unadjusted boundaries)."""


    def setUp(self):
        _register_memo_calendar(self)
        self.starts, self.ends = _d("2025-02-01"), _d("2025-03-01")

    def test_the_first_day_reads_the_preceding_business_day(self):
        grid = build_observation_grid(self.starts, self.ends, MEMO_CAL, ACT360)
        self.assertEqual((grid.interest_starts[0], grid.interest_ends[0]), (np.datetime64("2025-02-01"), np.datetime64("2025-02-03")))
        self.assertAlmostEqual(grid.weights[0] * 360, 2.0)
        self.assertEqual((grid.fixing_starts[0], grid.fixing_ends[0]), (np.datetime64("2025-01-31"), np.datetime64("2025-02-03")))
        self.assertAlmostEqual(grid.weights.sum() * 360, 28.0)

    def test_the_first_day_with_lookback_is_one_further_back(self):
        grid = build_observation_grid(self.starts, self.ends, MEMO_CAL, ACT360, lookback=2)
        np.testing.assert_array_equal(grid.fixing_starts[:2], _d("2025-01-29", "2025-01-30"))

    def test_the_lockout_fixing_is_used_lockout_times(self):
        grid = build_observation_grid(self.starts, self.ends, MEMO_CAL, ACT360, lockout=5)
        used = grid.fixing_starts == np.datetime64("2025-02-24")
        np.testing.assert_array_equal(grid.interest_starts[used], _d("2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"))


class TestPaymentScheduleWiring(UnitTest):

    def test_off_by_default(self):
        ps = build_payment_schedule(
            Date(2026, 6, 1), Date(2028, 6, 1), Frequency.SemiAnnually,
            ACT360, BDC.ModifiedFollowing, CAL,
        )
        self.assertFalse(ps.has_observation_grid)
        self.assertIsNone(ps.obs_offsets)

    def test_built_when_requested(self):
        ps = build_payment_schedule(
            Date(2026, 6, 1), Date(2028, 6, 1), Frequency.SemiAnnually,
            ACT360, BDC.ModifiedFollowing, CAL,
            build_observations=True,
        )
        self.assertTrue(ps.has_observation_grid)
        self.assertEqual(len(ps.obs_offsets), ps.n_periods)
        # per-period weights sum to the period fraction
        off = np.append(ps.obs_offsets, len(ps.obs_weights))
        for i in range(ps.n_periods):
            self.assertAlmostEqual(ps.obs_weights[off[i] : off[i + 1]].sum(), ps.period_fracs[i], places=12)
