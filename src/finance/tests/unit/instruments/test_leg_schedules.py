"""The leg's two faces: flat coupon fields for the simple case, a schedule as the truth the pricer reads."""
import numpy as np

from common.object import ValidationException
from common.testing import UnitTest
from finance.dates import BDC, Date, DayCountMethod, Frequency
from finance.instruments import CommonInstrument
from finance.instruments.enums import CouponType, PrincipalExchange
from finance.instruments.resolution import Swap
from finance.instruments.schedules import (
    CompoundedCouponEvent,
    CouponSchedule,
    FixedCouponEvent,
    build_payment_schedule,
)
from finance.pricing.pricers import SwapPricer
from finance.pricing.types import RateKind

EFFECTIVE = Date(2026, 1, 15)
MATURITY = Date(2031, 1, 15)


def _leg(**overrides) -> CommonInstrument:
    fields = dict(
        effective=EFFECTIVE, maturity=MATURITY, currency="USD", notional=1e6,
        payment_frequency=Frequency.Annually, coupon_type=CouponType.Fixed, coupon_rate=0.04,
    )
    fields.update(overrides)
    return CommonInstrument(**fields)


def _piecewise() -> CouponSchedule:
    return CouponSchedule(events=[
        FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.03),
        CompoundedCouponEvent(start_date=Date(2028, 1, 15), rate_index="USD.SOFR.1D", spread=0.001),
    ])


class TestFlatFieldsDeriveTheSchedule(UnitTest):
    COVERAGE = ["finance.instruments.common_instrument", "finance.instruments.schedules.coupon_schedule"]

    def test_fixed_leg_lowers_to_one_fixed_event(self):
        leg = _leg()
        self.assertEqual(leg.coupon_schedule.events, [FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.04)])
        self.assertFalse(leg.is_floating)
        self.assertFalse(leg.needs_observation_grid)

    def test_compounded_leg_lowers_to_one_compounded_event(self):
        leg = _leg(coupon_type=CouponType.GeometricAveraged, rate_index="USD.SOFR.1D", spread=0.0005, cap=0.07)
        [event] = leg.coupon_schedule.events
        self.assertIsInstance(event, CompoundedCouponEvent)
        self.assertEqual((event.rate_index, event.spread, event.cap, event.floor), ("USD.SOFR.1D", 0.0005, 0.07, None))
        self.assertTrue(leg.is_floating)
        self.assertTrue(leg.needs_observation_grid)

    def test_default_coupon_is_a_zero_rate_fixed_event(self):
        leg = CommonInstrument(
            effective=EFFECTIVE, maturity=MATURITY, currency="USD", notional=1e6,
            payment_frequency=Frequency.Annually, principal_exchange=PrincipalExchange.Final,
        )
        self.assertEqual(leg.coupon_schedule.events, [FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.0)])

    def test_custom_without_a_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Custom"):
            _leg(coupon_type=CouponType.Custom)

    def test_unlowerable_coupon_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unused"):
            _leg(coupon_type=CouponType.Unused)


class TestSuppliedScheduleIsTheTruth(UnitTest):
    COVERAGE = ["finance.instruments.common_instrument", "finance.instruments.schedules.coupon_schedule"]

    def test_supplied_schedule_tags_the_leg_custom(self):
        sched = _piecewise()
        leg = _leg(coupon_schedule=sched, rate_index="USD.SOFR.1D")  # flat fields said Fixed 4%
        self.assertIs(leg.coupon_schedule, sched)
        self.assertIs(leg.coupon_type, CouponType.Custom)
        self.assertTrue(leg.is_floating)
        self.assertTrue(leg.needs_observation_grid)
        self.assertEqual(leg.coupon_schedule.coupon_types, (CouponType.Fixed, CouponType.GeometricAveraged))

    def test_single_event_schedule_is_still_custom(self):
        sched = CouponSchedule(events=[FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.05)])
        self.assertIs(_leg(coupon_schedule=sched).coupon_type, CouponType.Custom)

    def test_floating_segment_needs_a_leg_rate_index(self):
        with self.assertRaises(ValidationException):
            _leg(coupon_schedule=_piecewise())

    def test_empty_schedule_is_rejected(self):
        with self.assertRaises(ValidationException):
            _leg(coupon_schedule=CouponSchedule(events=[]))


class TestReplaceKeepsTheScheduleHonest(UnitTest):
    COVERAGE = ["finance.instruments.common_instrument"]

    def test_derived_schedule_is_rebuilt_when_a_coupon_field_changes(self):
        unit = _leg().replace(coupon_rate=1.0)
        self.assertEqual(unit.coupon_schedule.events, [FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=1.0)])

    def test_derived_schedule_is_kept_when_only_dating_changes(self):
        leg = _leg()
        moved = leg.replace(maturity=Date(2036, 1, 15))
        self.assertEqual(moved.coupon_schedule.events, leg.coupon_schedule.events)

    def test_custom_leg_keeps_its_schedule_through_a_flat_field_change(self):
        sched = _piecewise()
        leg = _leg(coupon_schedule=sched, rate_index="USD.SOFR.1D")
        changed = leg.replace(coupon_rate=0.09)
        self.assertIs(changed.coupon_schedule, sched)
        self.assertIs(changed.coupon_type, CouponType.Custom)
        self.assertEqual(changed.coupon_rate, 0.09)

    def test_custom_leg_returns_to_flat_when_told_to(self):
        leg = _leg(coupon_schedule=_piecewise(), rate_index="USD.SOFR.1D")
        flat = leg.replace(coupon_schedule=None, coupon_type=CouponType.Fixed, coupon_rate=0.02)
        self.assertIs(flat.coupon_type, CouponType.Fixed)
        self.assertEqual(flat.coupon_schedule.events, [FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.02)])


class TestSerialization(UnitTest):
    COVERAGE = ["finance.instruments.common_instrument", "finance.instruments.schedules.base_schedule"]

    def test_leg_with_piecewise_schedule_round_trips(self):
        leg = _leg(coupon_schedule=_piecewise(), rate_index="USD.SOFR.1D", business_day_convention=BDC.ModifiedFollowing)
        back = CommonInstrument.from_json(leg.to_json())
        self.assertEqual(back, leg)
        self.assertIsInstance(back.coupon_schedule.events[1], CompoundedCouponEvent)

    def test_flat_leg_round_trips(self):
        leg = _leg(day_count_method=DayCountMethod.Thirty360)
        self.assertEqual(CommonInstrument.from_json(leg.to_json()), leg)


class TestPiecewiseCouponsReachThePricer(UnitTest):
    """A schedule on a Swap leg is what gets compiled — the compiler never reads the flat fields."""

    COVERAGE = ["finance.pricing.pricers.legs"]

    def test_fixed_then_compounded_leg_switches_rate_kind(self):
        swap = Swap.fixed_float_swap(notional=1e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=Date(2026, 1, 13))
        fixed = swap.receive_leg
        sched = CouponSchedule(events=[
            FixedCouponEvent(start_date=fixed.effective, coupon_rate=0.03),
            CompoundedCouponEvent(start_date=Date(2028, 1, 15), rate_index="USD.SOFR.1D"),
        ])
        piecewise = fixed.replace(coupon_schedule=sched, rate_index="USD.SOFR.1D", reset_frequency=Frequency.Daily,
                                  rate_calendar=swap.pay_leg.rate_calendar)
        program = SwapPricer().compile([Swap(receive_leg=piecewise, pay_leg=swap.pay_leg, currency="USD", index_name="USD.SOFR.1D")])
        ki = program.inputs
        first_leg = slice(ki.leg_offsets[0], ki.leg_offsets[1])
        kinds = ki.rate_kind[first_leg]
        self.assertEqual(kinds[0], RateKind.Fixed)
        self.assertEqual(kinds[-1], RateKind.Compounded)
        self.assertTrue(np.any(kinds == RateKind.Fixed) and np.any(kinds == RateKind.Compounded))
        self.assertGreater(ki.n_obs_periods, 0)


class TestLookbackStyleReachesTheGrid(UnitTest):
    """A term-rate compounder (monthly fixings in a quarterly coupon) must read each month, not an overnight."""

    COVERAGE = ["finance.pricing.pricers.legs", "finance.instruments.common_instrument"]

    def _inputs(self, style):
        from finance.instruments.schedules.observation import LookbackStyle
        from finance.pricing.pricers.legs import leg_spec

        leg = _leg(
            payment_frequency=Frequency.Quarterly, reset_frequency=Frequency.Monthly,
            coupon_type=CouponType.GeometricAveraged, rate_index="USD.SOFR.1D",
            lookback_style=LookbackStyle[style],
        )
        return leg_spec(leg, 1.0, "USD.SOFR.1D", "USD.SOFR.1D", 0).schedule

    def test_observation_shift_reads_the_whole_sub_period(self):
        ps = self._inputs("ObservationShift")
        self.assertEqual(ps.obs_weights.size, 60)  # 20 quarters x 3 months
        self.assertEqual(ps.obs_read_starts[0], EFFECTIVE.to_numpy())
        self.assertEqual(ps.obs_read_ends[0], Date(2026, 2, 15).to_numpy())

    def test_isda_lookback_reads_an_overnight_at_each_sub_period_start(self):
        ps = self._inputs("Lookback")
        self.assertEqual(ps.obs_read_ends[0], Date(2026, 1, 16).to_numpy())


class TestOnePeriodLegWithDailyFixings(UnitTest):
    """``Frequency.Once`` is the slowest frequency, so a daily reset under it is intra-period."""

    COVERAGE = ["finance.instruments.schedules.payment_schedule"]

    def test_once_plus_daily_reset_builds_the_full_fixing_grid(self):
        start, end = Date(2026, 9, 16), Date(2026, 12, 16)
        ps = build_payment_schedule(
            start, end, Frequency.Once, DayCountMethod.Actual360, BDC.NoAdjustment, "no_holidays",
            reset_frequency=Frequency.Daily, build_observations=True,
        )
        self.assertEqual(ps.n_periods, 1)
        self.assertEqual(ps.obs_weights.size, int(np.busday_count(start.to_numpy(), end.to_numpy())))
