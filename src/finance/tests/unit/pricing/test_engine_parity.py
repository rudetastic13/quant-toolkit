"""Every registered whole-portfolio engine must reproduce the numpy reference on a grid of coupon shapes.

The grid covers the shaping algebra (index floor, spread, floor, cap) crossed with compounded vs
averaged coupons and inclusive vs exclusive margin.  The test is driven by the program registry, so a
new backend (e.g. a primal-only Rust engine) is held to it by registering itself.
"""
import dataclasses
import itertools

import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.enums import CouponType, MarginTreatment
from finance.instruments.schedules import AveragedCouponEvent, CompoundedCouponEvent, CouponSchedule
from finance.instruments.resolution import Swap
from finance.markets.context import MarketContext
from finance.markets.curves import YieldCurve
from finance.pricing.engines.registry import program_registry
from finance.pricing.pricers import LegPricer
from finance.pricing.types import Backend

AS_OF = Date(2026, 6, 1)
SHAPES = {
    "plain": {},
    "spread": {"spread": 0.002},
    "index_floor": {"index_floor": 0.045},
    "cap_binding": {"cap": 0.04},
    "floor_binding": {"floor": 0.06},
    "all": {"spread": 0.001, "index_floor": 0.0, "floor": 0.01, "cap": 0.08},
}


def _market():
    origin = AS_OF.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 182, 365, 730, 1825, 3650)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.032, 0.035, 0.039, 0.043, 0.046])
    sofr = YieldCurve.build(dates, np.exp(-z * t), key="USD.SOFR.1D")
    return MarketContext(AS_OF, {sofr.name: sofr, "USD.STDCSA": sofr})


def _swap(shape, coupon_type, margin):
    swap = Swap.fixed_float_swap(
        notional=10e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="3Y", as_of=AS_OF, **shape
    )
    event_type = CompoundedCouponEvent if coupon_type == CouponType.GeometricAveraged else AveragedCouponEvent

    def with_margin(leg):
        old = leg.coupon_schedule.events[0]
        event = event_type(
            start_date=old.start_date, rate_index=old.rate_index, spread=old.spread,
            index_floor=old.index_floor, cap=old.cap, floor=old.floor, margin_treatment=margin,
        )
        return leg.replace(coupon_schedule=CouponSchedule(events=[event]))

    legs = [with_margin(leg) if leg.is_floating else leg for leg in swap]
    return dataclasses.replace(swap, receive_leg=legs[0], pay_leg=legs[1])


class TestEngineParity(UnitTest):
    COVERAGE = ["finance.pricing.engines.registry", "finance.pricing.pricers.base"]

    def test_every_registered_engine_matches_numpy(self):
        market = _market()
        book = [
            _swap(shape, coupon_type, margin)
            for shape, coupon_type, margin in itertools.product(
                SHAPES.values(),
                (CouponType.GeometricAveraged, CouponType.ArithmeticAveraged),
                (MarginTreatment.Inclusive, MarginTreatment.Exclusive),
            )
        ]
        program = LegPricer().compile(book)
        reference = program.reprice(market, backend=Backend.Numpy)
        backends = [b for b in Backend if b != Backend.Numpy and program_registry.has(b)]
        self.assertTrue(backends, "expected at least one non-numpy engine to compare")
        for backend in backends:
            with self.subTest(backend=backend.name):
                actual = program.reprice(market, backend=backend)
                np.testing.assert_allclose(actual.rate, reference.rate, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(actual.flow_pv, reference.flow_pv, rtol=1e-11, atol=1e-7)
                np.testing.assert_allclose(actual.instrument_pv, reference.instrument_pv, rtol=1e-11, atol=1e-6)
