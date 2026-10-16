"""Every coupon shape a leg supports, pinned: structure the engine sees and the numbers it produces.

The shapes mirror ``research/one_leg_examples.py``.  Each is checked two ways:

- **structure** — period count, rate kinds, observation-grid size, reset windows, and the
  closed-form facts a shape implies (a step-up's rates, an in-advance fixing repeated across
  its year, a spread riding on top of the projected rate);
- **numbers** — per-period realized rate and flow PV against the stored table in
  ``expects/leg_shapes.py``, so a change in reset windows, observation grids, shaping or
  discounting shows up as a diff at the period it touches.

Refresh the stored table after a deliberate change with::

    REGRESSION_UPDATE_EXPECTS=1 pytest src/finance/tests/regression/test_leg_shapes.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from common.testing import UnitTest
from common.testing.expects_loader import ModuleLoader
from finance.dates import BDC, Date, DayCountMethod, Frequency, Roll
from finance.dates.enums import FixingType
from finance.instruments import CommonInstrument, join_legs
from finance.instruments.enums import CouponType
from finance.instruments.schedules import CompoundedCouponEvent, CouponSchedule, FixedCouponEvent
from finance.instruments.schedules.observation import LookbackStyle
from finance.markets import HistoricalFixings
from finance.markets.context import MarketContext
from finance.markets.curves import YieldCurve
from finance.markets.rate_generator import RateGenerator
from finance.pricing.pricers import LegPricer
from finance.pricing.types import RateKind

EXPECTS_MODULE = "finance.tests.regression.expects.leg_shapes"
UPDATE = os.environ.get("REGRESSION_UPDATE_EXPECTS") == "1"

AS_OF = Date(2026, 6, 1)
EFFECTIVE = Date(2026, 6, 3)
MATURITY = Date(2029, 6, 3)
JOIN = Date(2028, 6, 3)
INDEX = "USD.SOFR.ON"
CURVE = "USD.SOFR.ON"
NOTIONAL = 10_000_000.0
FIXING_1Y = 0.0435  # the 2025-06-03 print example 3 reads in advance


class Legs:
    def __init__(self, legs):
        self.legs = legs
        self.currency, self.index_name, self.funding_id = "USD", "SOFR", "STDCSA"


def market() -> MarketContext:
    origin = AS_OF.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 90, 365, 730, 1095, 1825)])
    t = (dates.astype(np.int64) - origin.astype(np.int64)) / 365.0
    z = np.array([0.0, 0.041, 0.040, 0.039, 0.040, 0.042])
    fixings = HistoricalFixings(dates=np.array(["2025-06-03"], dtype="datetime64[D]"), values=np.array([FIXING_1Y]))
    sofr = YieldCurve.build(dates, np.exp(-z * t), currency="USD", index_name="SOFR", historical_fixings=fixings)
    return MarketContext(as_of_date=AS_OF, curves={sofr.name: sofr, "USD.STDCSA": sofr})


def leg(**fields) -> CommonInstrument:
    base = dict(
        effective=EFFECTIVE, maturity=MATURITY, currency="USD", notional=NOTIONAL,
        day_count_method=DayCountMethod.Actual360, business_day_convention=BDC.ModifiedFollowing,
        roll_convention=Roll.Empty, pay_calendar="no_holidays",
    )
    base.update(fields)
    return CommonInstrument(**base)


def one(single: CommonInstrument):
    return ((single, 1.0),)


SHAPES = {
    "fixed_semi": one(leg(
        payment_frequency=Frequency.SemiAnnually, coupon_type=CouponType.Fixed, coupon_rate=0.0425,
    )),
    "float_quarterly": one(leg(
        payment_frequency=Frequency.Quarterly, coupon_type=CouponType.Floating, rate_index=INDEX, spread=0.0015,
    )),
    "float_annual_reset_advance_monthly_pay": one(leg(
        payment_frequency=Frequency.Monthly, reset_frequency=Frequency.Annually, fixing_type=FixingType.Advance,
        coupon_type=CouponType.Floating, rate_index=INDEX, spread=0.0020,
    )),
    "monthly_compounder_quarterly_pay": one(leg(
        payment_frequency=Frequency.Quarterly, reset_frequency=Frequency.Monthly,
        lookback_style=LookbackStyle.ObservationShift,
        coupon_type=CouponType.GeometricAveraged, rate_index=INDEX,
    )),
    "daily_compounder_monthly_pay": one(leg(
        payment_frequency=Frequency.Monthly, reset_frequency=Frequency.Daily,
        coupon_type=CouponType.GeometricAveraged, rate_index=INDEX, spread=0.0010,
    )),
    "daily_average_monthly_pay": one(leg(
        payment_frequency=Frequency.Monthly, reset_frequency=Frequency.Daily,
        coupon_type=CouponType.ArithmeticAveraged, rate_index=INDEX, spread=0.0025,
    )),
    "step_up_fixed": one(leg(
        payment_frequency=Frequency.SemiAnnually,
        coupon_schedule=CouponSchedule(events=[
            FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.03),
            FixedCouponEvent(start_date=Date(2027, 6, 3), coupon_rate=0.04),
            FixedCouponEvent(start_date=Date(2028, 6, 3), coupon_rate=0.05),
        ]),
    )),
    "float_capped_floored": one(leg(
        payment_frequency=Frequency.Quarterly, coupon_type=CouponType.Floating, rate_index=INDEX, spread=0.0030,
        index_floor=0.0, floor=0.02, cap=0.06,
    )),
    "fixed_then_compounded": one(leg(
        payment_frequency=Frequency.Quarterly, reset_frequency=Frequency.Daily, rate_index=INDEX,
        coupon_schedule=CouponSchedule(events=[
            FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.035),
            CompoundedCouponEvent(start_date=Date(2027, 6, 3), rate_index=INDEX, spread=0.0015),
        ]),
    )),
    "joined_monthly_then_quarterly": join_legs(
        leg(maturity=JOIN, payment_frequency=Frequency.Monthly, coupon_type=CouponType.Fixed, coupon_rate=0.04),
        leg(effective=JOIN, payment_frequency=Frequency.Quarterly, coupon_type=CouponType.Fixed, coupon_rate=0.04),
    ),
}


def _table(program, result) -> pd.DataFrame:
    ki = program.inputs
    cf = result.cashflows
    return pd.DataFrame({
        "period": np.arange(ki.n_flows),
        "pay_date": ki.pay_dates.astype(str),
        "rate_kind": [RateKind(int(k)).name for k in ki.rate_kind],
        "rate": cf.rate,
        "flow_pv": cf.flow_pv,
    })


@pytest.mark.regression
class TestLegShapes(UnitTest):
    COVERAGE = [
        "finance.instruments.common_instrument",
        "finance.instruments.join",
        "finance.instruments.schedules.payment_schedule",
        "finance.instruments.schedules.observation",
        "finance.pricing.pricers.legs",
        "finance.pricing.kernels.compiler",
    ]

    @classmethod
    def setUpClass(cls):
        cls.market = market()
        cls.programs = {name: LegPricer().compile([Legs(legs)]) for name, legs in SHAPES.items()}
        cls.results = {name: program.price(cls.market) for name, program in cls.programs.items()}
        cls.inputs = {name: program.inputs for name, program in cls.programs.items()}

    # -- structure ----------------------------------------------------------------------

    def test_period_counts_and_rate_kinds(self):
        expected = {
            "fixed_semi": (6, {"Fixed"}, 0),
            "float_quarterly": (12, {"Float"}, 0),
            "float_annual_reset_advance_monthly_pay": (36, {"Float"}, 0),
            "monthly_compounder_quarterly_pay": (12, {"Compounded"}, 12),
            "daily_compounder_monthly_pay": (36, {"Compounded"}, 36),
            "daily_average_monthly_pay": (36, {"Averaged"}, 36),
            "step_up_fixed": (6, {"Fixed"}, 0),
            "float_capped_floored": (12, {"Float"}, 0),
            "fixed_then_compounded": (12, {"Fixed", "Compounded"}, 8),
            "joined_monthly_then_quarterly": (28, {"Fixed"}, 0),
        }
        for name, (periods, kinds, obs_periods) in expected.items():
            with self.subTest(shape=name):
                ki = self.inputs[name]
                self.assertEqual(ki.n_flows, periods)
                self.assertEqual({RateKind(int(k)).name for k in ki.rate_kind}, kinds)
                self.assertEqual(ki.n_obs_periods, obs_periods)

    def test_observation_grids(self):
        ki = self.inputs["monthly_compounder_quarterly_pay"]
        self.assertEqual(ki.obs_w.size, 36)                                   # 12 quarters x 3 months
        self.assertEqual(ki.obs_ends[0] - ki.obs_starts[0], np.timedelta64(30, "D"))  # reads the month
        for name in ("daily_compounder_monthly_pay", "daily_average_monthly_pay"):
            ki = self.inputs[name]
            self.assertEqual(ki.obs_w.size, int(np.busday_count(EFFECTIVE.to_numpy(), MATURITY.to_numpy())))
            self.assertEqual(ki.obs_ends[0] - ki.obs_starts[0], np.timedelta64(1, "D"))  # overnight reads
        ki = self.inputs["fixed_then_compounded"]
        self.assertEqual(ki.obs_starts[0], Date(2027, 6, 3).to_numpy())                 # grid starts at the switch

    def test_in_advance_annual_reset_repeats_the_historical_fixing(self):
        ki = self.inputs["float_annual_reset_advance_monthly_pay"]
        rates = self.results["float_annual_reset_advance_monthly_pay"].cashflows.rate
        np.testing.assert_array_equal(ki.reset_starts[:12], Date(2025, 6, 3).to_numpy())
        np.testing.assert_allclose(rates[:12], FIXING_1Y + 0.0020, atol=1e-15)
        self.assertEqual(ki.reset_starts[12], EFFECTIVE.to_numpy())                     # year 2 projects from effective

    def test_step_up_rates(self):
        np.testing.assert_allclose(
            self.results["step_up_fixed"].cashflows.rate, [0.03, 0.03, 0.04, 0.04, 0.05, 0.05], atol=1e-15
        )

    def test_spread_rides_on_the_projected_rate(self):
        ki = self.inputs["float_quarterly"]
        projected = RateGenerator(self.market).simple_rate(CURVE, ki.reset_starts, ki.reset_ends)
        np.testing.assert_allclose(self.results["float_quarterly"].cashflows.rate, projected + 0.0015, atol=1e-15)

    def test_bounds_do_not_bind_at_these_rates(self):
        bounded = self.results["float_capped_floored"].cashflows.rate
        plain = self.results["float_quarterly"].cashflows.rate
        np.testing.assert_allclose(bounded, plain + 0.0015, atol=1e-15)   # 30bp vs 15bp spread, nothing clipped
        self.assertTrue(np.all((bounded > 0.02) & (bounded < 0.06)))

    def test_switch_period_and_segments(self):
        rates = self.results["fixed_then_compounded"].cashflows.rate
        kinds = self.inputs["fixed_then_compounded"].rate_kind
        np.testing.assert_allclose(rates[:4], 0.035, atol=1e-15)
        self.assertTrue(np.all(kinds[:4] == RateKind.Fixed) and np.all(kinds[4:] == RateKind.Compounded))

    def test_joined_legs_reconcile_to_the_pieces(self):
        joined = self.results["joined_monthly_then_quarterly"]
        self.assertEqual(self.inputs["joined_monthly_then_quarterly"].n_legs, 2)
        pieces = [
            LegPricer().compile([Legs(one(piece))]).price(self.market).pv
            for piece, _ in SHAPES["joined_monthly_then_quarterly"]
        ]
        np.testing.assert_allclose(joined.leg_pv, pieces, rtol=1e-12)
        self.assertAlmostEqual(joined.pv, sum(pieces), places=6)

    def test_join_must_abut(self):
        with self.assertRaisesRegex(ValueError, "abut"):
            join_legs(leg(maturity=JOIN, payment_frequency=Frequency.Monthly),
                      leg(effective=Date(2028, 7, 3), payment_frequency=Frequency.Quarterly))

    # -- numbers ------------------------------------------------------------------------

    def test_per_period_rates_and_pvs_match_expects(self):
        tables = {name: _table(self.programs[name], self.results[name]) for name in SHAPES}
        loader = ModuleLoader(EXPECTS_MODULE)
        if UPDATE:
            loader.update("leg_shapes", tables)
            self.skipTest("expects refreshed")
        expects = loader.load()
        for name, got in tables.items():
            with self.subTest(shape=name):
                self.assertIn(name, expects, "no stored expects; run with REGRESSION_UPDATE_EXPECTS=1")
                want = expects[name]
                self.assertEqual(list(got["pay_date"]), list(want["pay_date"]))
                self.assertEqual(list(got["rate_kind"]), list(want["rate_kind"]))
                np.testing.assert_allclose(got["rate"], want["rate"].to_numpy(), rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(got["flow_pv"], want["flow_pv"].to_numpy(), rtol=1e-10, atol=1e-6)
