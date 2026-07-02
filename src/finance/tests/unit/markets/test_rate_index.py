"""Tests for RateIndex / FundingIndex convention descriptors."""
from common.testing import UnitTest
from finance.dates import Term
from finance.markets.conventions.rate_index import FundingIndex, RateIndex


class TestRateIndex(UnitTest):
    COVERAGE = ["finance.markets.conventions.rate_index"]

    def test_term_parsed_from_tenor(self):
        idx = RateIndex(currency="USD", name="SOFR", tenor="3M")
        self.assertEqual(idx.term, Term.from_str("3M"))

    def test_config_defaults_empty_dict(self):
        idx = RateIndex(currency="USD", name="SOFR", tenor="1D")
        self.assertEqual(idx.config, {})

    def test_getitem_reads_config(self):
        idx = RateIndex(currency="USD", name="SOFR", tenor="1D")
        idx.config["fixing_type"] = "Arrears"
        self.assertEqual(idx["fixing_type"], "Arrears")

    def test_funding_index_config_is_real_dict(self):
        fi = FundingIndex(currency="USD", name="STDCSA")
        self.assertEqual(fi.config, {})
        fi.config["discount_index"] = "SOFR"
        self.assertEqual(fi["discount_index"], "SOFR")
