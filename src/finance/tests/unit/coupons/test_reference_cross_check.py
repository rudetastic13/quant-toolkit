"""Reference calculators vs kernel path — the standing cross-check regression harness.

Prices the same legs two ways and asserts agreement:

- **kernel**: ``compile_portfolio`` lowers CouponSchedules to columnar KernelInputs,
  ``reprice`` evaluates them (the production hot path);
- **reference**: ``coupons.calculators`` computes each leg's rates directly against the
  MarketContext via ``CouponEvent.calculate`` dispatch (the slow, explicit twin).

Any divergence means the compiler lowering (sentinel encodings, margin masks, event
mapping) disagrees with the per-event semantics — exactly the regression this pins.
"""
import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date
from finance.dates.enums import BDC, DayCountMethod, Frequency, Roll
from finance.instruments.enums import MarginTreatment
from finance.instruments.schedules.coupon_schedule import (
    AveragedCouponEvent,
    CompoundedCouponEvent,
    CouponSchedule,
    FixedCouponEvent,
    FloatingCouponEvent,
)
from finance.instruments.schedules.payment_schedule import build_payment_schedule
from finance.coupons.calculators import calculate_custom
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, YieldCurve, ZeroCurve
from finance.pricing.kernels import LegSpec, StaticNotional, compile_portfolio
from finance.pricing.kernels.compiler import reprice

CURVE = "USD.SOFR"
INDEX = "USD SOFR"  # instrument-level label; resolves to CURVE


def _market(as_of: Date) -> MarketContext:
    origin = np.datetime64(as_of.to_str(), "D")
    nd = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 1095, 1825, 3650)], dtype="datetime64[D]")
    t = (nd.astype(np.int64) - origin.astype(np.int64)) / 365.0
    z = np.interp(t, [0.0, t[-1]], [0.035, 0.048])
    dfs = np.exp(-z * t)
    dfs[0] = 1.0
    ns = CurveNamespace()
    zero_curve = ZeroCurve(nd, dfs, CurveInterpolator.LogLinearDF)
    ns.bind(YieldCurve.from_registry(zero_curve, currency="USD", index_name="SOFR"))
    return MarketContext(as_of_date=as_of, curves=ns)


def _schedule(effective: Date, maturity: Date, *, observations: bool = False, lookback: int = 0, **overrides):
    return build_payment_schedule(
        effective=effective, maturity=maturity, frequency=Frequency.Quarterly,
        day_count_method=DayCountMethod.Actual360, bdc=BDC.ModifiedFollowing,
        calendar="no_holidays", roll=Roll.Empty,
        build_observations=observations, rate_lookback=lookback, **overrides,
    )


def _reference_leg_pv(ps, coupon: CouponSchedule, sign: float, notional: float, market) -> tuple:
    """Price one leg the slow way: per-event rates -> cash -> discount -> sum."""
    rate = np.zeros(ps.n_periods, dtype=np.float64)
    rs = ps.reset_starts if ps.reset_starts is not None else ps.accrual_starts
    re_ = ps.reset_ends if ps.reset_ends is not None else ps.accrual_ends
    calculate_custom(
        coupon, ps.accrual_starts, rate, market=market,
        reset_starts=rs, reset_ends=re_,
        obs_starts=ps.obs_read_starts, obs_ends=ps.obs_read_ends,
        obs_weights=ps.obs_weights, obs_offsets=ps.obs_offsets,
    )
    cash = notional * rate * ps.period_fracs
    df = market.zero_curve(CURVE).discount_factor(ps.payment_dates)
    return rate, sign * float(np.sum(cash * df))


@pytest.mark.calculators
class TestReferenceVsKernel(UnitTest):
    COVERAGE = ["finance.coupons.calculators", "finance.pricing.kernels.compiler"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        # far enough past as_of that a 5bd lookback never reads before the curve origin
        self.effective = Date(2026, 6, 15)
        self.maturity = Date(2028, 6, 15)

    def _cross_check(self, ps, coupon: CouponSchedule, sign: float = 1.0, notional: float = 10e6):
        leg = LegSpec(
            schedule=ps, coupon=coupon, sign=sign, discount_curve=CURVE,
            projection_curve=CURVE, notional=StaticNotional(notional), instrument=0,
        )
        kr = reprice(compile_portfolio([leg]), self.market)
        ref_rate, ref_pv = _reference_leg_pv(ps, coupon, sign, notional, self.market)
        np.testing.assert_allclose(kr.rate, ref_rate, atol=1e-13)
        self.assertAlmostEqual(kr.leg_pv[0], ref_pv, delta=abs(ref_pv) * 1e-12 + 1e-6)

    def test_fixed_leg(self):
        coupon = CouponSchedule(events=[FixedCouponEvent(start_date=self.effective, coupon_rate=0.042)])
        self._cross_check(_schedule(self.effective, self.maturity), coupon)

    def test_simple_float_leg_shaped(self):
        coupon = CouponSchedule(events=[FloatingCouponEvent(
            start_date=self.effective, rate_index=INDEX, spread=0.0025,
            index_floor=0.0, cap=0.06, floor=0.005,
        )])
        self._cross_check(_schedule(self.effective, self.maturity), coupon, sign=-1.0)

    def test_compounded_leg_with_lookback(self):
        coupon = CouponSchedule(events=[CompoundedCouponEvent(
            start_date=self.effective, rate_index=INDEX, spread=0.001,
        )])
        ps = _schedule(self.effective, self.maturity, observations=True, lookback=5)
        self._cross_check(ps, coupon)

    def test_averaged_leg_exclusive_margin(self):
        coupon = CouponSchedule(events=[AveragedCouponEvent(
            start_date=self.effective, rate_index=INDEX, spread=0.0015,
            margin_treatment=MarginTreatment.Exclusive,
        )])
        ps = _schedule(self.effective, self.maturity, observations=True)
        self._cross_check(ps, coupon)

    def test_piecewise_fixed_step_up(self):
        coupon = CouponSchedule(events=[
            FixedCouponEvent(start_date=self.effective, coupon_rate=0.03),
            FixedCouponEvent(start_date=Date(2027, 6, 15), coupon_rate=0.045),
        ])
        self._cross_check(_schedule(self.effective, self.maturity), coupon)

    def test_float_leg_advance_fixing(self):
        from finance.dates.enums import FixingType
        from finance.markets.paths import FlatPath

        coupon = CouponSchedule(events=[FloatingCouponEvent(
            start_date=self.effective, rate_index=INDEX, spread=0.001,
        )])
        ps = _schedule(self.effective, self.maturity, fixing_type=FixingType.Advance)
        # the first period's governing window was set before as_of — a realized fixing
        self.market = MarketContext(
            as_of_date=self.as_of, curves=self.market.curves,
            fixings={CURVE: FlatPath(name=CURVE, as_of_date=self.as_of, value=0.0345)},
        )
        self._cross_check(ps, coupon)

    def test_float_leg_super_period_reset(self):
        # pay quarterly, reset annually: repeated governing windows across periods
        coupon = CouponSchedule(events=[FloatingCouponEvent(
            start_date=self.effective, rate_index=INDEX, spread=0.0,
        )])
        ps = _schedule(self.effective, self.maturity, reset_frequency=Frequency.Annually)
        self._cross_check(ps, coupon)

    def test_compounded_leg_monthly_resets(self):
        # intra-period: monthly fixings compounded inside quarterly periods
        coupon = CouponSchedule(events=[CompoundedCouponEvent(
            start_date=self.effective, rate_index=INDEX, spread=0.0005,
        )])
        ps = _schedule(self.effective, self.maturity, observations=True,
                       reset_frequency=Frequency.Monthly)
        self._cross_check(ps, coupon)

    def test_piecewise_fixed_to_compounded_switch(self):
        coupon = CouponSchedule(events=[
            FixedCouponEvent(start_date=self.effective, coupon_rate=0.035),
            CompoundedCouponEvent(start_date=Date(2027, 6, 15), rate_index=INDEX, spread=0.0005),
        ])
        ps = _schedule(self.effective, self.maturity, observations=True)
        self._cross_check(ps, coupon)
