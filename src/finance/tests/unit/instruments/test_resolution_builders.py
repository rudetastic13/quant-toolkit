"""Resolution-builder tests: curve keys, the proj/disc split, builders and overrides."""
import pytest

from common.testing import UnitTest
from finance.dates import Date
from finance.dates.enums import DayCountMethod, Frequency
from finance.instruments.resolution import Deposit, Fra, Swap, curve_name, funding_curve_name


@pytest.mark.calibration
class TestResolutionBuilders(UnitTest):
    COVERAGE = [
        "finance.instruments.resolution.builders",
        "finance.instruments.resolution.resolver",
    ]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)

    def test_funding_curve_name_is_the_discount_key(self):
        self.assertEqual(funding_curve_name("USD", "STDCSA"), "USD.STDCSA")
        self.assertEqual(funding_curve_name("gbp", "usdstdcsa"), "GBP.USDSTDCSA")
        self.assertEqual(funding_curve_name("USD", "UST"), "USD.UST")
        self.assertEqual(funding_curve_name("USD"), "USD.STDCSA")

    def test_funding_id_rejects_a_tenor(self):
        with self.assertRaises(ValueError):
            funding_curve_name("USD", "SOFR.1D")

    def test_curve_name_is_the_projection_key(self):
        self.assertEqual(curve_name("usd", "usd.sofr.1d"), "USD.SOFR.1D")
        self.assertEqual(curve_name("USD", "USD.CME-TERM-SOFR.3M"), "USD.CME-TERM-SOFR.3M")
        with self.assertRaises(ValueError):
            curve_name("USD", "EUR.ESTR.1D")  # another currency
        with self.assertRaises(ValueError):
            curve_name("USD", "USD.STDCSA")  # a discount key
        with self.assertRaises(ValueError):
            curve_name("USD", "SOFR")  # not a key

    def test_swap_carries_funding_id_and_iterates_legs(self):
        s = Swap.fixed_float_swap(
            notional=100.0, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=self.as_of
        )
        self.assertEqual(s.funding_id, "STDCSA")
        self.assertEqual(list(s), [s.receive_leg, s.pay_leg])

    def test_swap_overrides_route_per_leg(self):
        s = Swap.fixed_float_swap(
            notional=100.0, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=self.as_of,
            fixed_frequency=Frequency.SemiAnnually,
            fixed_day_count=DayCountMethod.Thirty360,
            spread=0.001, cap=0.06,
        )
        fixed, floating = s.receive_leg, s.pay_leg
        # fixed-leg overrides land on the fixed leg only
        self.assertEqual(fixed.payment_frequency, Frequency.SemiAnnually)
        self.assertEqual(fixed.day_count_method, DayCountMethod.Thirty360)
        self.assertEqual(floating.payment_frequency, Frequency.Annually)  # convention
        # float shaping lands on the float leg only
        self.assertEqual(floating.spread, 0.001)
        self.assertEqual(floating.cap, 0.06)
        self.assertIsNone(fixed.cap)

    def test_swap_float_day_count_aligns_with_index(self):
        s = Swap.fixed_float_swap(
            notional=100.0, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=self.as_of
        )
        floating = s.pay_leg
        self.assertEqual(floating.day_count_method, DayCountMethod.Actual360)
        self.assertEqual(floating.rate_index, "USD.SOFR.1D")

    def test_term_index_swap_matches_tenor(self):
        s = Swap.fixed_float_swap(
            notional=100.0, rate_index="USD.CME-TERM-SOFR.3M", fixed_rate=0.04, tenor="2Y", as_of=self.as_of
        )
        floating = s.pay_leg
        self.assertEqual(floating.rate_index, "USD.CME-TERM-SOFR.3M")
        self.assertEqual(floating.payment_frequency, Frequency.Quarterly)
        self.assertEqual(floating.reset_frequency, Frequency.Quarterly)
        self.assertEqual(s.index_name, "USD.CME-TERM-SOFR.3M")
        with self.assertRaises(KeyError):  # the family does not publish 2M
            Swap.fixed_float_swap(notional=100.0, rate_index="USD.CME-TERM-SOFR.2M", fixed_rate=0.04, tenor="2Y", as_of=self.as_of)

    def test_swap_currency_mismatch_raises(self):
        s = Swap.fixed_float_swap(
            notional=100.0, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=self.as_of
        )
        with self.assertRaises(ValueError):
            Swap(
                receive_leg=s.receive_leg, pay_leg=s.pay_leg,
                currency="EUR", index_name="USD.SOFR.1D",
            )

    def test_deposit_builder(self):
        d = Deposit.spot_deposit(rate=0.04, tenor="3M", as_of=self.as_of)
        self.assertGreater(d.maturity.to_numpy(), d.effective.to_numpy())
        self.assertEqual(d.funding_id, "STDCSA")
        self.assertEqual(curve_name(d.currency, d.index_name), "USD.SOFR.1D")
        self.assertEqual(d.day_count_method, DayCountMethod.Actual360)  # from the index

    def test_fra_builder(self):
        f = Fra.forward_starting(rate=0.04, start="3M", end="6M", as_of=self.as_of)
        self.assertGreater(f.maturity.to_numpy(), f.effective.to_numpy())
        self.assertEqual(f.rate_index, "USD.SOFR.1D")

    def test_fra_rejects_inverted_window(self):
        with self.assertRaises(ValueError):
            Fra.forward_starting(rate=0.04, start="6M", end="3M", as_of=self.as_of)
