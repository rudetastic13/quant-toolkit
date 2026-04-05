"""Tests for instrument interface protocols — verifies structural typing via isinstance checks."""
from common.testing import UnitTest
from finance.dates import Date, Term, Frequency, DayCountMethod, BDC, Roll, Direction
from finance.instruments import CommonInstrument
from finance.instruments.enums import AmortizationType, CouponType, LegType
from finance.instruments.interfaces.traits import (
    HasScheduleParams,
    HasStubDates,
    HasFixedRate,
    HasFloatingRate,
    HasRateBounds,
    HasNotional,
    HasAmortization,
)
from finance.instruments.interfaces.products import Bond, Loan, FRA, Future, SwapLeg


class TestTraitProtocols(UnitTest):
    COVERAGE = ["finance.instruments.interfaces.traits"]

    def _make_common_instrument(self, **overrides):
        defaults = dict(
            effective=Date(2025, 1, 15),
            maturity=Date(2030, 1, 15),
            currency="USD",
            notional=1_000_000.0,
            payment_frequency=Frequency.SemiAnnually,
            day_count_method=DayCountMethod.Thirty360,
            business_day_convention=BDC.ModifiedFollowing,
            roll_convention=Roll.RollDay15,
            direction=Direction.Forward,
            coupon_rate=0.05,
            coupon_type=CouponType.Fixed,
        )
        defaults.update(overrides)
        return CommonInstrument(**defaults)

    def test_common_instrument_satisfies_has_schedule_params(self):
        ci = self._make_common_instrument()
        self.assertIsInstance(ci, HasScheduleParams)

    def test_common_instrument_satisfies_has_stub_dates(self):
        ci = self._make_common_instrument()
        self.assertIsInstance(ci, HasStubDates)

    def test_common_instrument_satisfies_has_fixed_rate(self):
        ci = self._make_common_instrument()
        self.assertIsInstance(ci, HasFixedRate)

    def test_common_instrument_satisfies_has_floating_rate(self):
        ci = self._make_common_instrument(
            coupon_type=CouponType.Floating,
            rate_index="SOFR",
            spread=0.01,
            reset_frequency=Frequency.Quarterly,
        )
        self.assertIsInstance(ci, HasFloatingRate)

    def test_common_instrument_satisfies_has_rate_bounds(self):
        ci = self._make_common_instrument(cap=0.08, floor=0.02)
        self.assertIsInstance(ci, HasRateBounds)

    def test_common_instrument_satisfies_has_notional(self):
        ci = self._make_common_instrument()
        self.assertIsInstance(ci, HasNotional)

    def test_common_instrument_satisfies_has_amortization(self):
        ci = self._make_common_instrument(
            amortization_type=AmortizationType.LevelPay,
            original_notional=1_000_000.0,
            amortization_start=Date(2025, 1, 15),
            amortization_end=Date(2030, 1, 15),
        )
        self.assertIsInstance(ci, HasAmortization)


class TestProductProtocols(UnitTest):
    COVERAGE = ["finance.instruments.interfaces.products"]

    def test_bond_composes_expected_traits(self):
        self.assertIn(HasScheduleParams, Bond.__mro__)
        self.assertIn(HasStubDates, Bond.__mro__)
        self.assertIn(HasNotional, Bond.__mro__)
        self.assertIn(HasFixedRate, Bond.__mro__)

    def test_loan_composes_expected_traits(self):
        self.assertIn(HasScheduleParams, Loan.__mro__)
        self.assertIn(HasStubDates, Loan.__mro__)
        self.assertIn(HasNotional, Loan.__mro__)
        self.assertIn(HasAmortization, Loan.__mro__)
        self.assertIn(HasFixedRate, Loan.__mro__)

    def test_swap_leg_composes_expected_traits(self):
        self.assertIn(HasScheduleParams, SwapLeg.__mro__)
        self.assertIn(HasStubDates, SwapLeg.__mro__)
        self.assertIn(HasNotional, SwapLeg.__mro__)
        self.assertIn(HasFixedRate, SwapLeg.__mro__)

    def test_fra_composes_has_notional(self):
        self.assertIn(HasNotional, FRA.__mro__)

    def test_future_composes_has_notional(self):
        self.assertIn(HasNotional, Future.__mro__)


class TestLegTypeEnum(UnitTest):
    COVERAGE = ["finance.instruments.enums.leg_type"]

    def test_leg_type_values(self):
        self.assertEqual(LegType.Pay, 1)
        self.assertEqual(LegType.Receive, 2)

    def test_leg_type_is_supported(self):
        self.assertTrue(LegType.Pay.is_supported())
        self.assertTrue(LegType.Receive.is_supported())
