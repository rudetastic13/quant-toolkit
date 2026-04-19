"""Tests for ConventionSet and ConventionRegistry."""
from common.testing import UnitTest
from finance.dates import Term
from finance.dates.term.term_type import TermType
from finance.dates.enums import BDC, DayCountMethod, Frequency, Roll
from finance.pricing.conventions import ConventionSet, ConventionRegistry


def _make_convention(**overrides) -> ConventionSet:
    defaults = dict(
        day_count_method=DayCountMethod.ActualActual,
        payment_frequency=Frequency.SemiAnnually,
        reset_frequency=Frequency.Quarterly,
        business_day_convention=BDC.ModifiedFollowing,
        roll_convention=Roll.EOM,
        calendar="NYC",
        spot_lag=Term(2, TermType.Days),
    )
    defaults.update(overrides)
    return ConventionSet(**defaults)


class TestConventionSet(UnitTest):
    COVERAGE = ["finance.pricing.conventions.convention_set"]

    def test_frozen(self):
        c = _make_convention()
        with self.assertRaises(Exception):
            c.day_count_method = DayCountMethod.Thirty360

    def test_override_returns_new_instance(self):
        c = _make_convention()
        c2 = c.override(payment_frequency=Frequency.Annually)
        self.assertEqual(c2.payment_frequency, Frequency.Annually)
        self.assertEqual(c.payment_frequency, Frequency.SemiAnnually)

    def test_override_preserves_unspecified_fields(self):
        c = _make_convention()
        c2 = c.override(calendar="LON")
        self.assertEqual(c2.day_count_method, c.day_count_method)
        self.assertEqual(c2.business_day_convention, c.business_day_convention)

    def test_optional_fixed_day_count_defaults_none(self):
        c = _make_convention()
        self.assertIsNone(c.fixed_day_count_method)

    def test_optional_fixed_day_count_explicit(self):
        c = _make_convention(fixed_day_count_method=DayCountMethod.Thirty360)
        self.assertEqual(c.fixed_day_count_method, DayCountMethod.Thirty360)


class TestConventionRegistry(UnitTest):
    COVERAGE = ["finance.pricing.conventions.convention_registry"]

    def _fresh_registry(self) -> ConventionRegistry:
        from common.registry import Registry
        return ConventionRegistry(registry=Registry("TestConventionRegistry"))

    def test_register_and_get(self):
        reg = self._fresh_registry()
        conv = _make_convention()
        reg.register("USD", "SOFR", conv)
        self.assertEqual(reg.get("USD", "SOFR"), conv)

    def test_key_is_case_insensitive(self):
        reg = self._fresh_registry()
        conv = _make_convention()
        reg.register("usd", "sofr", conv)
        self.assertEqual(reg.get("USD", "SOFR"), conv)
        self.assertEqual(reg.get("Usd", "Sofr"), conv)

    def test_missing_key_raises(self):
        reg = self._fresh_registry()
        with self.assertRaises(KeyError):
            reg.get("EUR", "EURIBOR")

    def test_duplicate_raises_without_overwrite(self):
        reg = self._fresh_registry()
        conv = _make_convention()
        reg.register("USD", "SOFR", conv)
        with self.assertRaises(ValueError):
            reg.register("USD", "SOFR", conv)

    def test_overwrite_succeeds(self):
        reg = self._fresh_registry()
        c1 = _make_convention()
        c2 = _make_convention(calendar="LON")
        reg.register("USD", "SOFR", c1)
        reg.register("USD", "SOFR", c2, overwrite=True)
        self.assertEqual(reg.get("USD", "SOFR").calendar, "LON")

    def test_has(self):
        reg = self._fresh_registry()
        reg.register("USD", "SOFR", _make_convention())
        self.assertTrue(reg.has("USD", "SOFR"))
        self.assertFalse(reg.has("EUR", "EURIBOR"))
