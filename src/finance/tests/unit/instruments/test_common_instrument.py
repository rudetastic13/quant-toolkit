"""CommonInstrument validation tests (moved from the deleted test_interfaces.py)."""
from common.object import ValidationException
from common.testing import UnitTest
from finance.dates import Date, Frequency
from finance.instruments import CommonInstrument
from finance.instruments.enums import CouponType


class TestCommonInstrumentValidation(UnitTest):
    COVERAGE = ["finance.instruments.common_instrument"]

    def _make(self, **overrides):
        defaults = dict(
            effective=Date(2025, 1, 15),
            maturity=Date(2030, 1, 15),
            currency="USD",
            notional=1_000_000.0,
            payment_frequency=Frequency.SemiAnnually,
            coupon_type=CouponType.Fixed,
        )
        defaults.update(overrides)
        return CommonInstrument(**defaults)

    def test_valid_instrument_has_no_failures(self):
        ci = self._make()
        result = ci.validate(raise_on=set())
        self.assertFalse(result.has_failures)

    def test_maturity_before_effective_fails(self):
        ci = self._make(effective=Date(2030, 1, 15), maturity=Date(2025, 1, 15))
        with self.assertRaises(ValidationException):
            ci.validate()

    def test_negative_notional_adds_warning(self):
        ci = self._make(notional=-1_000_000.0)
        result = ci.validate(raise_on=set())
        self.assertTrue(result.has_warnings)
        self.assertFalse(result.has_failures)

    def test_floating_without_rate_index_fails(self):
        ci = self._make(coupon_type=CouponType.Floating, rate_index=None)
        with self.assertRaises(ValidationException):
            ci.validate()

    def test_cap_below_floor_fails(self):
        ci = self._make(cap=0.01, floor=0.05)
        with self.assertRaises(ValidationException):
            ci.validate()

    def test_valid_cap_and_floor_no_failure(self):
        ci = self._make(cap=0.08, floor=0.02)
        result = ci.validate(raise_on=set())
        self.assertFalse(result.has_failures)
