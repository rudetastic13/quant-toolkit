"""Currency index convention data: typed, loadable, minimally complete."""
from common.testing import UnitTest
from finance.dates.enums import FixingType
from finance.markets.conventions.ccy_indices.amrs.usd import DEFINED


class TestUsdIndexDefinitions(UnitTest):
    COVERAGE = ["finance.markets.conventions.ccy_indices.amrs.usd"]

    def test_sofr_defined(self):
        self.assertIn("SOFR", DEFINED)

    def test_fixing_type_is_typed_enum(self):
        self.assertIs(DEFINED["SOFR"]["fixing_type"], FixingType.Arrears)

    def test_required_keys_present(self):
        for key in ("long_name", "rate_type", "rate_source", "fixing_type"):
            self.assertIn(key, DEFINED["SOFR"])
