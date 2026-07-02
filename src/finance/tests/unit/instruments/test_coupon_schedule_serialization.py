"""Round-trip tests for CouponEvent/CouponSchedule serialization."""
from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.common_instrument import CommonInstrument
from finance.instruments.enums import CouponType, MarginTreatment
from finance.dates.enums import Frequency
from finance.instruments.schedules.coupon_schedule import (
    AveragedCouponEvent,
    CompoundedCouponEvent,
    CouponEvent,
    CouponSchedule,
    FixedCouponEvent,
    FloatingCouponEvent,
)


class TestCouponEventSerialization(UnitTest):
    COVERAGE = ["finance.instruments.schedules.coupon_schedule"]

    def test_fixed_round_trip(self):
        e = FixedCouponEvent(start_date=Date(2026, 1, 15), coupon_rate=0.045)
        r = CouponEvent.from_dict(e.to_dict())
        self.assertEqual(r, e)
        self.assertIsInstance(r, FixedCouponEvent)

    def test_floating_round_trip_preserves_none_bounds(self):
        e = FloatingCouponEvent(
            start_date=Date(2026, 1, 15), rate_index="USD SOFR", spread=0.001,
            index_floor=0.0, cap=None, floor=None,
        )
        r = CouponEvent.from_dict(e.to_dict())
        self.assertEqual(r, e)
        self.assertEqual(r.index_floor, 0.0)  # genuine 0% floor, not "unfloored"
        self.assertIsNone(r.cap)

    def test_averaged_and_compounded_round_trip(self):
        for cls in (AveragedCouponEvent, CompoundedCouponEvent):
            e = cls(
                start_date=Date(2026, 3, 1), rate_index="USD SOFR",
                margin_treatment=MarginTreatment.Exclusive,
            )
            r = CouponEvent.from_dict(e.to_dict())
            self.assertEqual(r, e)
            self.assertIsInstance(r, cls)

    def test_unknown_coupon_type_raises(self):
        with self.assertRaises(ValueError):
            CouponEvent.from_dict({"coupon_type": "NotACouponType", "start_date": Date(2026, 1, 1)})

    def test_missing_coupon_type_raises(self):
        with self.assertRaises(ValueError):
            CouponEvent.from_dict({"start_date": Date(2026, 1, 1)})

    def test_extra_keys_ignored(self):
        e = CouponEvent.from_dict({
            "coupon_type": CouponType.Fixed, "start_date": Date(2026, 1, 1),
            "coupon_rate": 0.03, "not_an_event_field": object(),
        })
        self.assertEqual(e.coupon_rate, 0.03)


class TestCouponScheduleSerialization(UnitTest):
    COVERAGE = ["finance.instruments.schedules.coupon_schedule"]

    def test_schedule_round_trip(self):
        sched = CouponSchedule(events=[
            FixedCouponEvent(start_date=Date(2026, 1, 15), coupon_rate=0.04),
            CompoundedCouponEvent(start_date=Date(2028, 1, 15), rate_index="USD SOFR", spread=0.0005),
        ])
        r = CouponSchedule.from_list(sched.to_list())
        self.assertEqual(r.events, sched.events)

    def test_schedule_from_instrument_fallback(self):
        leg = CommonInstrument(
            effective=Date(2026, 1, 15), maturity=Date(2031, 1, 15), currency="USD",
            notional=1e6, payment_frequency=Frequency.Annually,
            coupon_type=CouponType.GeometricAveraged, rate_index="USD SOFR",
        )
        sched = CouponSchedule.schedule_from_instrument(leg)
        self.assertEqual(len(sched.events), 1)
        self.assertIsInstance(sched.events[0], CompoundedCouponEvent)
        self.assertEqual(sched.events[0].start_date, leg.effective)
        self.assertEqual(sched.events[0].rate_index, "USD SOFR")
        # leg carries spread=None -> event default 0.0 applies
        self.assertEqual(sched.events[0].spread, 0.0)

    def test_schedule_from_instrument_prefers_precomputed(self):
        pre = CouponSchedule(events=[FixedCouponEvent(start_date=Date(2026, 1, 15))])
        leg = CommonInstrument(
            effective=Date(2026, 1, 15), maturity=Date(2031, 1, 15), currency="USD",
            notional=1e6, payment_frequency=Frequency.Annually,
            coupon_type=CouponType.Fixed, schedules={"coupon": pre},
        )
        self.assertIs(CouponSchedule.schedule_from_instrument(leg), pre)
