"""Compiler + reprice: columnar lowering, kind-grouping, and piecewise coupons.

Flagship cases:
- fixed-rate STEP-UP bond (pure-fixed but exercises piecewise column lowering),
- par compounded-SOFR swap -> net PV ~ 0 (fixed leg + compounded leg in one reprice),
- a piecewise fixed -> compounded -> fixed leg (arbitrary switches).
"""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date, DayCountMethod, Frequency, BDC
from finance.markets.curves import YieldCurve
from finance.markets.context import MarketContext
from finance.instruments.schedules.payment_schedule import build_payment_schedule
from finance.instruments.schedules.coupon_schedule import (
    CouponSchedule,
    FixedCouponEvent,
    CompoundedCouponEvent,
)
from finance.instruments.enums import PrincipalExchange
from finance.pricing.kernels import LegSpec, compile_portfolio, reprice
from finance.pricing.kernels.inputs import NO_CAP, NO_FLOOR
from finance.pricing.types import RateKind

ACT360 = DayCountMethod.Actual360
CAL = "no_holidays"
CURVE = "USD.SOFR.1D"


def _notional(ps, amount: float = 100.0) -> np.ndarray:
    return np.full(ps.n_periods, amount)


def _market(origin="2026-06-01"):
    dates = np.array([origin, "2028-06-01", "2031-06-01"], dtype="datetime64[D]")
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    yield_curve = YieldCurve.build(dates, np.exp(-0.04 * t), key="USD.SOFR.1D")
    return MarketContext(as_of=Date(2026, 6, 1), curves={yield_curve.name: yield_curve, "USD.STDCSA": yield_curve}), yield_curve


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
                      notional=_notional(ps), instrument=0)
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
        leg = LegSpec(ps, coupon, 1.0, CURVE, None, _notional(ps), 0)
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
            notional=_notional(fixed_ps), instrument=0,
        )
        float_leg = LegSpec(
            float_ps,
            CouponSchedule(events=[CompoundedCouponEvent(start_date=Date(2026, 6, 1))]),
            sign=-1.0, discount_curve=CURVE, projection_curve=CURVE,
            notional=_notional(float_ps), instrument=0,
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
                                1.0, CURVE, None, _notional(fixed_ps), inst))
            legs.append(LegSpec(float_ps, CouponSchedule(events=[CompoundedCouponEvent(start_date=Date(2026, 6, 1))]),
                                -1.0, CURVE, CURVE, _notional(float_ps), inst))
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
        leg = LegSpec(ps, coupon, 1.0, CURVE, CURVE, _notional(ps), 0)
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
        leg = LegSpec(ps, CouponSchedule(events=[ev]), 1.0, CURVE, CURVE, _notional(ps), 0)
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


class TestPrincipalLowering(UnitTest):
    """Principal flows come from the per-period notional: a bullet repays once, an amortiser as it goes."""

    COVERAGE = ["finance.pricing.kernels.compiler"]

    def _leg(self, notional, exchange):
        ps = _annual_schedule(4)
        coupon = CouponSchedule(events=[FixedCouponEvent(start_date=Date(2026, 6, 1), coupon_rate=0.03)])
        return ps, LegSpec(ps, coupon, 1.0, CURVE, None, notional, 0, principal_exchange=exchange)

    def test_bullet_repays_at_maturity_only(self):
        ps, leg = self._leg(_notional(ps=_annual_schedule(4)), PrincipalExchange.Both)
        ki = compile_portfolio([leg])
        self.assertEqual(ki.n_flows, 4 + 2)
        np.testing.assert_array_equal(ki.notional[4:], [-100.0, 100.0])
        self.assertEqual(ki.pay_dates[4], ps.accrual_starts[0])
        self.assertEqual(ki.pay_dates[5], ps.payment_dates[-1])
        np.testing.assert_array_equal(ki.period_frac[4:], [1.0, 1.0])
        np.testing.assert_array_equal(ki.fixed_rate[4:], [1.0, 1.0])

    def test_amortiser_pays_each_step_down_on_its_payment_date(self):
        ps, leg = self._leg(np.array([100.0, 75.0, 50.0, 25.0]), PrincipalExchange.Both)
        ki = compile_portfolio([leg])
        np.testing.assert_array_equal(ki.notional[:4], [100.0, 75.0, 50.0, 25.0])  # coupons accrue on the outstanding
        self.assertEqual(ki.n_flows, 4 + 1 + 4)
        np.testing.assert_array_equal(ki.notional[4:], [-100.0, 25.0, 25.0, 25.0, 25.0])
        np.testing.assert_array_equal(ki.pay_dates[5:], ps.payment_dates)
        self.assertEqual(ki.notional[4:].sum(), 0.0)  # what went out comes back

    def test_principal_only_on_the_slower_grid_skips_flat_periods(self):
        # principal every second coupon period: the notional is flat in between, so no flow there
        ps, leg = self._leg(np.array([100.0, 100.0, 50.0, 50.0]), PrincipalExchange.Final)
        ki = compile_portfolio([leg])
        self.assertEqual(ki.n_flows, 4 + 2)
        np.testing.assert_array_equal(ki.notional[4:], [50.0, 50.0])
        np.testing.assert_array_equal(ki.pay_dates[4:], ps.payment_dates[[1, 3]])

    def test_amortising_pv_is_the_sum_of_its_flows(self):
        mkt, _ = _market()
        ps, leg = self._leg(np.array([100.0, 75.0, 50.0, 25.0]), PrincipalExchange.Both)
        res = reprice(compile_portfolio([leg]), mkt)
        self.assertAlmostEqual(res.instrument_pv[0], res.flow_pv.sum(), places=9)
        bullet = reprice(compile_portfolio([self._leg(_notional(ps), PrincipalExchange.Both)[1]]), mkt)
        self.assertLess(abs(res.instrument_pv[0]), abs(bullet.instrument_pv[0]))  # less duration, closer to par

    def test_notional_must_be_one_value_per_period(self):
        ps = _annual_schedule(4)
        coupon = CouponSchedule(events=[FixedCouponEvent(start_date=Date(2026, 6, 1), coupon_rate=0.03)])
        with self.assertRaises(ValueError):
            compile_portfolio([LegSpec(ps, coupon, 1.0, CURVE, None, np.array([100.0, 50.0]), 0)])
