"""Compiler + reprice: columnar lowering, kind-grouping, and piecewise coupons.

Flagship cases:
- fixed-rate STEP-UP bond (pure-fixed but exercises piecewise column lowering),
- par compounded-SOFR swap -> net PV ~ 0 (fixed leg + compounded leg in one reprice),
- a piecewise fixed -> compounded -> fixed leg (arbitrary switches).
"""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date, DayCountMethod, Frequency, BDC
from finance.markets.curves import YieldCurve, ZeroCurve
from finance.markets.context import MarketContext
from finance.markets.curves import CurveNamespace
from finance.instruments.schedules.payment_schedule import build_payment_schedule
from finance.instruments.schedules.coupon_schedule import (
    CouponSchedule,
    FixedCouponEvent,
    CompoundedCouponEvent,
)
from finance.pricing.kernels import LegSpec, compile_portfolio, reprice, StaticNotional
from finance.pricing.kernels.inputs import NO_CAP, NO_FLOOR
from finance.pricing.types import RateKind

ACT360 = DayCountMethod.Actual360
CAL = "no_holidays"
CURVE = "USD.SOFR"


def _market(origin="2026-06-01"):
    dates = np.array([origin, "2028-06-01", "2031-06-01"], dtype="datetime64[D]")
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    ns = CurveNamespace()
    yield_curve = YieldCurve.from_registry(
        ZeroCurve(dates, np.exp(-0.04 * t)),
        currency="USD",
        index_name="SOFR",
    )
    ns.bind(yield_curve)
    return MarketContext(as_of_date=Date(2026, 6, 1), curves=ns), yield_curve.zero_curve


def _annual_schedule(years=4, build_obs=False):
    return build_payment_schedule(
        Date(2026, 6, 1), Date(2026 + years, 6, 1), Frequency.Annually,
        ACT360, BDC.ModifiedFollowing, CAL, build_observations=build_obs,
    )


class TestFixedStepUpBond(UnitTest):
    COVERAGE = ["finance.pricing.kernels.compiler"]

    def test_step_up_rates_lowered_per_period(self):
        # 3% for y1-2, 4% y3, 5% y4 — a step-up encoded as 3 events on one leg
        ps = _annual_schedule(4)
        coupon = CouponSchedule(events=[
            FixedCouponEvent(start_date=Date(2026, 6, 1), coupon_rate=0.03),
            FixedCouponEvent(start_date=Date(2028, 6, 1), coupon_rate=0.04),
            FixedCouponEvent(start_date=Date(2029, 6, 1), coupon_rate=0.05),
        ])
        leg = LegSpec(ps, coupon, sign=1.0, discount_curve=CURVE, projection_curve=None,
                      notional=StaticNotional(100.0), instrument=0)
        ki = compile_portfolio([leg])
        # all fixed, but the fixed_rate column steps up
        self.assertTrue(np.all(ki.rate_kind == RateKind.Fixed))
        np.testing.assert_array_equal(ki.fixed_rate, [0.03, 0.03, 0.04, 0.05])

    def test_step_up_pv_equals_manual(self):
        mkt, curve = _market()
        ps = _annual_schedule(4)
        rates = [0.03, 0.03, 0.04, 0.05]
        coupon = CouponSchedule(events=[
            FixedCouponEvent(start_date=Date(2026, 6, 1), coupon_rate=0.03),
            FixedCouponEvent(start_date=Date(2028, 6, 1), coupon_rate=0.04),
            FixedCouponEvent(start_date=Date(2029, 6, 1), coupon_rate=0.05),
        ])
        leg = LegSpec(ps, coupon, 1.0, CURVE, None, StaticNotional(100.0), 0)
        res = reprice(compile_portfolio([leg]), mkt)

        df = curve.discount_factor(ps.payment_dates)
        manual = float(np.sum(100.0 * np.array(rates) * ps.period_fracs * df))
        self.assertAlmostEqual(res.instrument_pv[0], manual, places=10)


class TestParCompoundedSwap(UnitTest):
    COVERAGE = ["finance.pricing.kernels.compiler"]

    def test_par_swap_pv_zero(self):
        mkt, curve = _market()
        fixed_ps = _annual_schedule(4)
        float_ps = _annual_schedule(4, build_obs=True)

        # par rate from the curve
        df_end = curve.discount_factor(fixed_ps.accrual_ends)
        annuity = float(np.sum(fixed_ps.period_fracs * df_end))
        par = (1.0 - float(df_end[-1])) / annuity

        fixed_leg = LegSpec(
            fixed_ps,
            CouponSchedule(events=[FixedCouponEvent(start_date=Date(2026, 6, 1), coupon_rate=par)]),
            sign=1.0, discount_curve=CURVE, projection_curve=None,
            notional=StaticNotional(100.0), instrument=0,
        )
        float_leg = LegSpec(
            float_ps,
            CouponSchedule(events=[CompoundedCouponEvent(start_date=Date(2026, 6, 1))]),
            sign=-1.0, discount_curve=CURVE, projection_curve=CURVE,
            notional=StaticNotional(100.0), instrument=0,
        )
        res = reprice(compile_portfolio([fixed_leg, float_leg]), mkt)
        # two legs, one instrument -> net PV ~ 0
        self.assertEqual(res.leg_pv.shape[0], 2)
        self.assertEqual(res.instrument_pv.shape[0], 1)
        self.assertAlmostEqual(float(res.instrument_pv[0]), 0.0, places=8)

    def test_two_swaps_population(self):
        # two independent swaps in one compiled portfolio -> two instrument PVs
        mkt, curve = _market()
        fixed_ps = _annual_schedule(4)
        float_ps = _annual_schedule(4, build_obs=True)
        df_end = curve.discount_factor(fixed_ps.accrual_ends)
        par = (1.0 - float(df_end[-1])) / float(np.sum(fixed_ps.period_fracs * df_end))

        legs = []
        for inst in (0, 1):
            legs.append(LegSpec(fixed_ps, CouponSchedule(events=[FixedCouponEvent(start_date=Date(2026, 6, 1), coupon_rate=par)]),
                                1.0, CURVE, None, StaticNotional(100.0), inst))
            legs.append(LegSpec(float_ps, CouponSchedule(events=[CompoundedCouponEvent(start_date=Date(2026, 6, 1))]),
                                -1.0, CURVE, CURVE, StaticNotional(100.0), inst))
        res = reprice(compile_portfolio(legs), mkt)
        self.assertEqual(res.instrument_pv.shape[0], 2)
        np.testing.assert_allclose(res.instrument_pv, 0.0, atol=1e-8)


class TestPiecewiseSwitches(UnitTest):
    COVERAGE = ["finance.pricing.kernels.compiler"]

    def test_fixed_compounded_fixed(self):
        # one leg: y1 fixed, y2-3 compounded SOFR, y4 fixed — arbitrary switches
        mkt, _ = _market()
        ps = _annual_schedule(4, build_obs=True)
        coupon = CouponSchedule(events=[
            FixedCouponEvent(start_date=Date(2026, 6, 1), coupon_rate=0.035),
            CompoundedCouponEvent(start_date=Date(2027, 6, 1)),
            FixedCouponEvent(start_date=Date(2029, 6, 1), coupon_rate=0.05),
        ])
        leg = LegSpec(ps, coupon, 1.0, CURVE, CURVE, StaticNotional(100.0), 0)
        ki = compile_portfolio([leg])
        np.testing.assert_array_equal(
            ki.rate_kind,
            [RateKind.Fixed, RateKind.Compounded, RateKind.Compounded, RateKind.Fixed],
        )
        # only the two compounded periods contribute to the global obs grid
        self.assertEqual(ki.n_obs_periods, 2)

        res = reprice(ki, mkt)
        # fixed periods carry their stepped rates; compounded periods are positive & sensible
        self.assertAlmostEqual(res.rate[0], 0.035, places=12)
        self.assertAlmostEqual(res.rate[3], 0.05, places=12)
        self.assertTrue(np.all(res.rate[1:3] > 0))
        self.assertTrue(np.isfinite(res.instrument_pv[0]))


class TestOptionalBounds(UnitTest):
    """A genuine 0% bound must be distinguishable from 'no bound' in the lowered columns."""

    COVERAGE = ["finance.pricing.kernels.compiler"]

    def _compile_with(self, *, index_floor, cap, floor):
        ps = _annual_schedule(2, build_obs=True)
        ev = CompoundedCouponEvent(
            start_date=Date(2026, 6, 1), index_floor=index_floor, cap=cap, floor=floor,
        )
        leg = LegSpec(ps, CouponSchedule(events=[ev]), 1.0, CURVE, CURVE, StaticNotional(100.0), 0)
        return compile_portfolio([leg])

    def test_none_lowers_to_sentinels(self):
        ki = self._compile_with(index_floor=None, cap=None, floor=None)
        self.assertTrue(np.all(ki.index_floor == NO_FLOOR))
        self.assertTrue(np.all(ki.cap == NO_CAP))
        self.assertTrue(np.all(ki.floor == NO_FLOOR))

    def test_explicit_zero_is_preserved(self):
        # 0% floor (SOFR floored at 0) must survive as 0.0, NOT collapse to "no bound"
        ki = self._compile_with(index_floor=0.0, cap=0.0, floor=0.0)
        self.assertTrue(np.all(ki.index_floor == 0.0))
        self.assertTrue(np.all(ki.cap == 0.0))
        self.assertTrue(np.all(ki.floor == 0.0))
