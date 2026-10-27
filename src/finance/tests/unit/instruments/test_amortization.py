"""Amortisation: generated profiles (straight-line, level-pay), the custom schedule, the window fields,
and the principal flows the compiler lowers from the per-period notional."""

import numpy as np

from common.object import ValidationException
from common.testing import UnitTest
from finance.dates import BDC, Date, DayCountMethod, Frequency, Term
from finance.instruments import AmortizationType, CommonInstrument
from finance.instruments.enums import CouponType, PrincipalExchange
from finance.instruments.schedules import (
    CouponSchedule,
    FixedCouponEvent,
    FloatingCouponEvent,
    NotionalEvent,
    NotionalSchedule,
    build_payment_schedule,
    notional_profile,
)
from finance.markets.context import MarketContext
from finance.markets.curves import YieldCurve
from finance.pricing.kernels import compile_portfolio, reprice
from finance.pricing.pricers.legs import leg_spec

EFFECTIVE = Date(2026, 1, 15)
MATURITY = Date(2031, 1, 15)
FACE = 1e6
RATE = 0.04


def _leg(**overrides) -> CommonInstrument:
    fields = dict(
        effective=EFFECTIVE, maturity=MATURITY, currency="USD", notional=FACE,
        payment_frequency=Frequency.Annually, day_count_method=DayCountMethod.Actual360,
        business_day_convention=BDC.ModifiedFollowing, coupon_type=CouponType.Fixed, coupon_rate=RATE,
        principal_exchange=PrincipalExchange.Final,
    )
    fields.update(overrides)
    return CommonInstrument(**fields)


def _schedule(leg: CommonInstrument):
    return build_payment_schedule(
        leg.effective, leg.maturity, leg.payment_frequency, leg.day_count_method,
        leg.business_day_convention, leg.pay_calendar,
    )


def _flows(leg: CommonInstrument):
    """(schedule, interest per period, principal paid per payment date) from the compiled program."""
    spec = leg_spec(leg, 1.0, "USD.STDCSA", "USD.SOFR.1D", 0)
    ki = compile_portfolio([spec])
    n = spec.schedule.n_periods
    dates = np.array(["2026-01-15", "2031-01-15"], dtype="datetime64[D]")
    sofr = YieldCurve.build(dates, np.array([1.0, np.exp(-0.03 * 5)]), key="USD.SOFR.1D")
    result = reprice(ki, MarketContext(EFFECTIVE, {"USD.SOFR.1D": sofr, "USD.STDCSA": sofr}))
    cash = ki.notional * result.rate * ki.period_frac
    principal = np.zeros(n)
    principal[np.isin(spec.schedule.payment_dates, ki.pay_dates[n:])] = cash[n:]
    return spec.schedule, cash[:n], principal, result


class TestGeneratedProfiles(UnitTest):
    COVERAGE = ["finance.instruments.schedules.amortization", "finance.instruments.common_instrument"]

    def test_bullet_is_the_face_throughout(self):
        leg = _leg()
        self.assertIs(leg.amortization, AmortizationType.Bullet)
        np.testing.assert_array_equal(notional_profile(leg, _schedule(leg)), np.full(5, FACE))

    def test_straight_line_repays_equal_principal_each_period(self):
        leg = _leg(amortization=AmortizationType.StraightLine)
        np.testing.assert_allclose(notional_profile(leg, _schedule(leg)), [1e6, 8e5, 6e5, 4e5, 2e5])
        _, interest, principal, _ = _flows(leg)
        np.testing.assert_allclose(principal, np.full(5, 2e5))
        self.assertTrue(np.all(np.diff(interest) < 0))  # interest falls with the balance

    def test_level_pay_has_a_constant_total_payment(self):
        leg = _leg(amortization=AmortizationType.LevelPay)
        schedule, interest, principal, _ = _flows(leg)
        total = interest + principal
        np.testing.assert_allclose(total, total[0], rtol=1e-12)
        self.assertAlmostEqual(principal.sum(), FACE, places=6)
        self.assertTrue(np.all(np.diff(principal) > 0))  # principal share grows as interest shrinks
        # the closed form on the schedule's own fractions
        growth = 1.0 + RATE * schedule.period_fracs
        self.assertAlmostEqual(total[0], FACE / np.cumprod(1.0 / growth).sum(), places=6)

    def test_level_pay_is_resolved_when_the_fixed_rate_steps(self):
        step_up = CouponSchedule(events=[
            FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.04),
            FixedCouponEvent(start_date=Date(2028, 1, 15), coupon_rate=0.06),
        ])
        leg = _leg(amortization=AmortizationType.LevelPay, coupon_schedule=step_up)
        _, interest, principal, _ = _flows(leg)
        total = interest + principal
        np.testing.assert_allclose(total[:2], total[0], rtol=1e-12)   # held at 4%
        np.testing.assert_allclose(total[2:], total[2], rtol=1e-12)   # re-solved at 6%, then held
        self.assertGreater(total[2], total[0])
        self.assertAlmostEqual(principal.sum(), FACE, places=6)

    def test_level_pay_on_a_floating_leg_is_rate_dependent(self):
        leg = _leg(amortization=AmortizationType.LevelPay, coupon_type=CouponType.Floating, rate_index="USD.SOFR.1D")
        with self.assertRaises(NotImplementedError):
            notional_profile(leg, _schedule(leg))
        leg = _leg(amortization=AmortizationType.LevelPay, coupon_schedule=CouponSchedule(events=[
            FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.04),
            FloatingCouponEvent(start_date=Date(2029, 1, 15), rate_index="USD.SOFR.1D"),
        ]), rate_index="USD.SOFR.1D")
        with self.assertRaises(NotImplementedError):
            notional_profile(leg, _schedule(leg))


class TestAmortisationWindow(UnitTest):
    COVERAGE = ["finance.instruments.schedules.amortization", "finance.instruments.common_instrument"]

    def test_interest_only_then_level_pay_to_a_balloon_at_maturity(self):
        leg = _leg(amortization=AmortizationType.LevelPay, amort_start=Date(2028, 1, 15), amortization_term=Term.from_str("8Y"))
        schedule, interest, principal, _ = _flows(leg)
        profile = notional_profile(leg, schedule)
        np.testing.assert_array_equal(profile[:2], FACE)            # interest-only
        np.testing.assert_array_equal(principal[:2], 0.0)
        self.assertTrue(0 < profile[4] < profile[3] < profile[2] == FACE)
        total = interest + principal
        np.testing.assert_allclose(total[2:4], total[2], rtol=1e-12)  # level while amortising
        self.assertAlmostEqual(principal[4], profile[4], places=6)    # the balloon is what is left
        self.assertGreater(principal[4], total[2])
        self.assertAlmostEqual(principal.sum(), FACE, places=6)

    def test_balloon_date_before_maturity(self):
        leg = _leg(amortization=AmortizationType.StraightLine, amort_end=Date(2029, 1, 15), amortization_term=Term.from_str("6Y"))
        schedule, interest, principal, _ = _flows(leg)
        profile = notional_profile(leg, schedule)
        np.testing.assert_allclose(profile, [1e6, 1e6 - 1e6 / 6, 1e6 - 2e6 / 6, 0.0, 0.0])
        np.testing.assert_allclose(principal, [1e6 / 6, 1e6 / 6, 1e6 - 2e6 / 6, 0.0, 0.0])  # balloon on the amort_end date
        np.testing.assert_array_equal(interest[3:], 0.0)  # nothing outstanding afterwards

    def test_annual_principal_on_quarterly_coupons(self):
        leg = _leg(
            amortization=AmortizationType.StraightLine, payment_frequency=Frequency.Quarterly,
            principal_frequency=Frequency.Annually,
        )
        schedule, interest, principal, _ = _flows(leg)
        profile = notional_profile(leg, schedule)
        self.assertEqual(schedule.n_periods, 20)
        np.testing.assert_allclose(profile, np.repeat([1e6, 8e5, 6e5, 4e5, 2e5], 4))  # flat within each year
        paid = principal[principal > 0]
        self.assertEqual(paid.size, 5)
        np.testing.assert_allclose(paid, 2e5)
        np.testing.assert_array_equal(np.flatnonzero(principal), [3, 7, 11, 15, 19])  # on the anniversaries

    def test_level_pay_on_the_slower_principal_grid_is_level_per_year(self):
        leg = _leg(
            amortization=AmortizationType.LevelPay, payment_frequency=Frequency.Quarterly,
            principal_frequency=Frequency.Annually,
        )
        _, interest, principal, _ = _flows(leg)
        yearly = (interest + principal).reshape(5, 4).sum(axis=1)
        np.testing.assert_allclose(yearly, yearly[0], rtol=1e-12)
        self.assertAlmostEqual(principal.sum(), FACE, places=6)

    def test_principal_grid_must_nest_in_the_coupon_grid(self):
        with self.assertRaises(ValidationException):  # faster than the coupon: rejected at construction
            _leg(amortization=AmortizationType.StraightLine, principal_frequency=Frequency.Monthly)
        leg = _leg(amortization=AmortizationType.StraightLine, payment_frequency=Frequency.Quarterly,
                   principal_frequency=Frequency.Annually, amort_start=Date(2026, 2, 15))  # off the coupon grid
        with self.assertRaises(ValueError):
            notional_profile(leg, _schedule(leg))

    def test_window_must_lie_inside_the_contract(self):
        with self.assertRaises(ValidationException):
            _leg(amortization=AmortizationType.StraightLine, amort_start=Date(2025, 1, 15))
        with self.assertRaises(ValidationException):
            _leg(amortization=AmortizationType.StraightLine, amort_end=Date(2032, 1, 15))
        with self.assertRaises(ValidationException):
            _leg(amortization=AmortizationType.StraightLine, amort_start=Date(2029, 1, 15), amort_end=Date(2028, 1, 15))
        with self.assertRaises(ValidationException):
            _leg(amortization=AmortizationType.StraightLine, amortization_term=Term.from_str("0Y"))

    def test_seasoned_loan_solves_from_todays_balance(self):
        # originated earlier with curtailments we do not know: today's outstanding is the face,
        # amortising from today over the remaining term, so the payment is recomputed from it
        seasoned = _leg(notional=785_978.58, amortization=AmortizationType.LevelPay)
        _, interest, principal, _ = _flows(seasoned)
        self.assertAlmostEqual(principal.sum(), 785_978.58, places=6)
        np.testing.assert_allclose(interest + principal, (interest + principal)[0], rtol=1e-12)


class TestCustomSchedule(UnitTest):
    COVERAGE = ["finance.instruments.schedules.amortization", "finance.instruments.common_instrument"]

    def test_events_say_what_is_outstanding_from_when(self):
        schedule = NotionalSchedule(events=[
            NotionalEvent(start_date=Date(2029, 1, 15), notional=3e5),
            NotionalEvent(start_date=Date(2028, 1, 15), notional=6e5),  # out of order: sorted on construction
        ])
        leg = _leg(notional_schedule=schedule)
        self.assertIs(leg.amortization, AmortizationType.Custom)
        np.testing.assert_allclose(notional_profile(leg, _schedule(leg)), [1e6, 1e6, 6e5, 3e5, 3e5])
        _, _, principal, _ = _flows(leg)
        np.testing.assert_allclose(principal, [0.0, 4e5, 3e5, 0.0, 3e5])  # known curtailments, then the rest at maturity

    def test_custom_without_a_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Custom"):
            _leg(amortization=AmortizationType.Custom)

    def test_serialize_keeps_the_schedule_only_when_custom(self):
        self.assertNotIn("notional_schedule", _leg(amortization=AmortizationType.LevelPay).serialize())
        custom = _leg(notional_schedule=NotionalSchedule(events=[NotionalEvent(start_date=Date(2028, 1, 15), notional=5e5)]))
        self.assertEqual(len(custom.serialize()["notional_schedule"]["events"]), 1)

    def test_amortising_swap_leg_steps_its_notional_without_principal_flows(self):
        leg = _leg(amortization=AmortizationType.StraightLine, principal_exchange=PrincipalExchange.Unused)
        spec = leg_spec(leg, 1.0, "USD.STDCSA", "USD.SOFR.1D", 0)
        ki = compile_portfolio([spec])
        self.assertEqual(ki.n_flows, 5)  # coupons only
        np.testing.assert_allclose(ki.notional, [1e6, 8e5, 6e5, 4e5, 2e5])
