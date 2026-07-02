"""Resolution-builder tests: funding alias, the proj/disc split label, Deposit/FRA builders."""
import pytest

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Deposit, Fra, Swap, curve_name, funding_curve_name


@pytest.mark.calibration
class TestResolutionBuilders(UnitTest):
    COVERAGE = [
        "finance.instruments.resolution.builders",
        "finance.instruments.resolution.resolver",
    ]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)

    def test_funding_alias_resolves_per_ccy(self):
        self.assertEqual(funding_curve_name("USD", "STDCSA"), "USD.SOFR")
        self.assertEqual(funding_curve_name("EUR", "STDCSA"), "EUR.ESTR")

    def test_funding_id_passthrough_for_concrete_index(self):
        # An unrecognised id is treated as a concrete index name.
        self.assertEqual(funding_curve_name("USD", "FEDFUNDS"), "USD.FEDFUNDS")

    def test_funding_alias_unknown_currency_raises(self):
        with self.assertRaises(KeyError):
            funding_curve_name("JPY", "STDCSA")

    def test_swap_carries_funding_id_and_iterates_legs(self):
        s = Swap(notional=100.0, rate_index="SOFR", fixed_rate=0.04, tenor="5Y", as_of=self.as_of)
        self.assertEqual(s.funding_id, "STDCSA")
        self.assertEqual(list(s), [s.receive_leg, s.pay_leg])

    def test_deposit_builder(self):
        d = Deposit(rate=0.04, tenor="3M", as_of=self.as_of)
        self.assertGreater(d.maturity.to_numpy(), d.effective.to_numpy())
        self.assertEqual(d.funding_id, "STDCSA")
        self.assertEqual(curve_name(d.currency, d.index_name), "USD.SOFR")

    def test_fra_builder(self):
        f = Fra(rate=0.04, start="3M", end="6M", as_of=self.as_of)
        self.assertGreater(f.maturity.to_numpy(), f.effective.to_numpy())
        self.assertEqual(f.rate_index, "USD SOFR")

    def test_fra_rejects_inverted_window(self):
        with self.assertRaises(ValueError):
            Fra(rate=0.04, start="6M", end="3M", as_of=self.as_of)
