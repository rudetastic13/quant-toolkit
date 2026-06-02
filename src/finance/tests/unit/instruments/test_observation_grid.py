"""Observation grid: continuous-boundary + searchsorted construction of the Tier-2 grid."""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date, DayCountMethod, Frequency, BDC, period_fractions
from finance.instruments.schedules.observation import build_observation_grid, LookbackStyle
from finance.instruments.schedules.payment_schedule import build_payment_schedule

ACT360 = DayCountMethod.Actual360
CAL = "no_holidays"


def _semi_annual():
    # 1y, two semiannual accrual periods
    accrual_starts = np.array(["2026-06-01", "2026-12-01"], dtype="datetime64[D]")
    accrual_ends = np.array(["2026-12-01", "2027-06-01"], dtype="datetime64[D]")
    return accrual_starts, accrual_ends


class TestObservationGridStructure(UnitTest):
    COVERAGE = ["finance.instruments.schedules.observation"]

    def setUp(self):
        self.starts, self.ends = _semi_annual()
        self.grid = build_observation_grid(self.starts, self.ends, CAL, ACT360)

    def test_offsets_well_formed(self):
        off = self.grid.offsets
        self.assertEqual(off[0], 0)
        self.assertTrue(np.all(np.diff(off) > 0))      # strictly increasing
        self.assertEqual(len(off), len(self.ends))     # one per accrual period

    def test_subperiods_tile_exactly(self):
        # no gaps / overlaps: each sub_end equals the next sub_start; spans [effective, maturity]
        np.testing.assert_array_equal(self.grid.sub_ends[:-1], self.grid.sub_starts[1:])
        self.assertEqual(self.grid.sub_starts[0], self.starts[0])
        self.assertEqual(self.grid.sub_ends[-1], self.ends[-1])

    def test_each_subperiod_in_one_accrual_period(self):
        # period membership via searchsorted; sub_start in [accrual_start, accrual_end)
        period_idx = np.searchsorted(self.ends, self.grid.sub_starts, side="right")
        for k, p in enumerate(period_idx):
            self.assertGreaterEqual(self.grid.sub_starts[k], self.starts[p])
            self.assertLess(self.grid.sub_starts[k], self.ends[p])

    def test_weights_sum_to_period_fraction(self):
        # Act/360 is additive over contiguous sub-periods -> per-period weights sum exactly
        off = np.append(self.grid.offsets, self.grid.n_obs)
        period_fracs = period_fractions(ACT360, self.starts, self.ends)
        for i in range(len(self.ends)):
            seg = self.grid.weights[off[i] : off[i + 1]]
            self.assertAlmostEqual(seg.sum(), period_fracs[i], places=12)


class TestLookbackLockout(UnitTest):
    COVERAGE = ["finance.instruments.schedules.observation"]

    def setUp(self):
        self.starts, self.ends = _semi_annual()

    def test_isda_lookback_shifts_rate_keeps_real_weight(self):
        # ISDA Lookback (default): weight from the real sub-period (unchanged); read start
        # shifted back; read end is the overnight next-business-day of the shifted start.
        base = build_observation_grid(self.starts, self.ends, CAL, ACT360, lookback=0)
        lb = build_observation_grid(self.starts, self.ends, CAL, ACT360, lookback=2)
        np.testing.assert_array_equal(base.weights, lb.weights)  # weight basis = real period
        np.testing.assert_array_equal(lb.read_starts, np.busday_offset(base.sub_starts, -2, roll="preceding"))
        np.testing.assert_array_equal(lb.read_ends, np.busday_offset(lb.read_starts, 1, roll="following"))

    def test_observation_shift_shifts_both_window_and_weight(self):
        base = build_observation_grid(self.starts, self.ends, CAL, ACT360, lookback=0)
        os = build_observation_grid(
            self.starts, self.ends, CAL, ACT360, lookback=2, lookback_style=LookbackStyle.ObservationShift
        )
        # both ends shifted; weight comes from the shifted window (so it differs from real)
        np.testing.assert_array_equal(os.read_starts, np.busday_offset(base.sub_starts, -2, roll="preceding"))
        np.testing.assert_array_equal(os.read_ends, np.busday_offset(base.sub_ends, -2, roll="preceding"))
        self.assertFalse(np.array_equal(os.weights, base.weights))

    def test_lockout_freezes_tail(self):
        lk = build_observation_grid(self.starts, self.ends, CAL, ACT360, lockout=3)
        # the cutoff for period 0 is accrual_end[0] - 3 BD; tail obs are clamped there
        cutoff0 = np.busday_offset(self.ends[0], -3, roll="preceding")
        period_idx = np.searchsorted(self.ends, lk.sub_starts, side="right")
        tail0 = (period_idx == 0) & (lk.sub_starts > cutoff0)
        self.assertTrue(tail0.any())
        np.testing.assert_array_equal(lk.read_starts[tail0], np.full(tail0.sum(), cutoff0))


class TestPaymentScheduleWiring(UnitTest):
    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

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
