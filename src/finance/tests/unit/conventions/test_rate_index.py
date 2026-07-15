"""Tests for RateIndex / FundingIndex convention descriptors."""
from common.testing import UnitTest
from finance.dates import Term
from finance.dates.enums import DayCountMethod, FixingType
from finance.conventions.rate_index import FundingIndex, RateIndex


def _sofr(**overrides) -> RateIndex:
    defaults = dict(
        currency="USD",
        name="SOFR",
        day_count_method=DayCountMethod.Actual360,
        fixing_calendar="NYC",
    )
    defaults.update(overrides)
    return RateIndex(**defaults)


class TestRateIndex(UnitTest):
    COVERAGE = ["finance.conventions.rate_index"]

    def test_overnight_when_no_tenor(self):
        idx = _sofr()
        self.assertIsNone(idx.tenor)
        self.assertTrue(idx.is_overnight)

    def test_term_index_when_tenor_given(self):
        idx = _sofr(name="TERMSOFR", tenor=Term.from_str("3M"))
        self.assertFalse(idx.is_overnight)
        self.assertEqual(idx.tenor, Term.from_str("3M"))

    def test_label(self):
        self.assertEqual(_sofr().label, "USD SOFR")

    def test_fixing_type_defaults_arrears(self):
        self.assertEqual(_sofr().fixing_type, FixingType.Arrears)

    def test_funding_index_config_is_real_dict(self):
        fi = FundingIndex(currency="USD", name="STDCSA")
        self.assertEqual(fi.config, {})
        fi.config["discount_index"] = "SOFR"
        self.assertEqual(fi["discount_index"], "SOFR")
