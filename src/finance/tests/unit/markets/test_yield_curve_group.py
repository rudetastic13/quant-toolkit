"""YieldCurveGroup: one currency's curves by key, roles read off the key grammar."""

import numpy as np

from common.testing import UnitTest
from finance.markets.curves import YieldCurve, ZeroCurve, bp
from finance.markets.yield_curve_group import YieldCurveGroup

ORIGIN = np.datetime64("2026-06-01", "D")
SOFR = "USD.SOFR.1D"
STDCSA = "USD.STDCSA"


def _zero_curve(level: float = 0.04) -> ZeroCurve:
    x = np.array([0.0, 365.0, 1825.0])
    dfs = np.exp(-level * x / 365.0)
    dfs[0] = 1.0
    return ZeroCurve(x, dfs)


def _yield_curve(key: str = SOFR, level: float = 0.04) -> YieldCurve:
    return YieldCurve.from_registry(ORIGIN, _zero_curve(level), key=key)


class TestYieldCurveGroup(UnitTest):
    COVERAGE = ["finance.markets.yield_curve_group"]

    def setUp(self):
        self.sofr = _yield_curve()
        self.usd = YieldCurveGroup("usd", default_funding="stdcsa")

    def test_identity_and_defaults(self):
        self.assertEqual(self.usd.currency, "USD")
        self.assertEqual(self.usd.fx_rate, 1.0)
        self.assertEqual(self.usd.default_funding, "STDCSA")
        self.assertFalse(self.usd.frozen)
        self.assertEqual(len(self.usd), 0)
        gbp = YieldCurveGroup("GBP", fx_rate=1.27)
        self.assertEqual(gbp.fx_rate, 1.27)
        self.assertIsNone(gbp.default_funding)
        with self.assertRaises(ValueError):
            YieldCurveGroup("GBP", fx_rate=0.0)

    def test_add_registers_the_curve_under_its_key_and_its_funding_names(self):
        self.usd.add(self.sofr, funding_names=[STDCSA, "usd.usdsofr"])
        self.assertEqual(list(self.usd), [SOFR, STDCSA, "USD.USDSOFR"])
        self.assertIs(self.usd[SOFR], self.sofr)  # stored as given: conventions and fixings included
        alias = self.usd[STDCSA]
        self.assertIs(alias.zero_curve, self.sofr.zero_curve)  # one math object, one view per name
        self.assertIsNone(alias.conventions)
        self.assertIsNone(alias.historical_fixings)
        self.assertEqual(alias.name, STDCSA)
        self.assertIs(self.usd.projection("sofr"), self.sofr)
        self.assertIs(self.usd.discount(), alias)
        self.assertIs(self.usd.discount("USDSOFR").zero_curve, self.sofr.zero_curve)

    def test_roles_come_from_the_key_grammar(self):
        cme = _yield_curve("USD.CMECSA")
        ftp = _yield_curve("USD.FEDFUND.1D")
        self.usd.add(cme)
        self.usd.add(ftp)
        self.assertIs(self.usd.discount("CMECSA"), cme)
        self.assertIs(self.usd.projection("FEDFUND", "1D"), ftp)
        with self.assertRaises(KeyError):
            self.usd.projection("CMECSA")  # a discount key never projects
        with self.assertRaises(KeyError):
            self.usd.discount("FEDFUND")  # a projection key never discounts unless named as funding

    def test_lookup_normalises_keys_and_rejects_other_currencies(self):
        self.usd.add(self.sofr, funding_names=[STDCSA])
        self.assertIs(self.usd["usd.sofr.1d"], self.sofr)
        self.assertIn("usd.stdcsa", self.usd)
        self.assertNotIn("GBP.SONIA.1D", self.usd)
        self.assertNotIn(42, self.usd)
        with self.assertRaises(KeyError):
            self.usd["USD.FEDFUND.1D"]
        with self.assertRaises(ValueError):
            self.usd["GBP.SONIA.1D"]

    def test_add_validates_currency_and_funding_names(self):
        with self.assertRaises(ValueError):
            self.usd.add(_yield_curve("GBP.STDCSA"))  # a GBP key in the USD group
        with self.assertRaises(ValueError):
            self.usd.add(self.sofr, funding_names=["GBP.STDCSA"])
        with self.assertRaises(ValueError):
            self.usd.add(self.sofr, funding_names=["USD.FEDFUND.1D"])  # a funding name is a discount key
        with self.assertRaises(TypeError):
            self.usd.add(_zero_curve())
        self.assertEqual(len(self.usd), 0)  # nothing registered on failure

    def test_discount_without_a_default_needs_a_name(self):
        gbp = YieldCurveGroup("GBP")
        gbp.add(_yield_curve("GBP.STDCSA"))
        with self.assertRaises(ValueError):
            gbp.discount()
        self.assertEqual(gbp.discount("STDCSA").name, "GBP.STDCSA")

    def test_a_composite_registers_like_any_curve(self):
        self.usd.add(self.sofr, funding_names=[STDCSA])
        fedfund = YieldCurve.from_registry(ORIGIN, self.sofr.zero_curve - 5 * bp, key="USD.FEDFUND.1D")
        self.usd.add(fedfund)
        self.assertIs(self.usd.projection("FEDFUND").zero_curve["base"], self.sofr.zero_curve)

    def test_frozen_copy_keeps_settings_and_rejects_add(self):
        self.usd.add(self.sofr, funding_names=[STDCSA])
        copy = self.usd._rebound({SOFR: self.sofr})
        self.assertTrue(copy.frozen)
        self.assertEqual(copy.default_funding, "STDCSA")
        self.assertEqual(list(copy), [SOFR])
        with self.assertRaises(TypeError):
            copy.add(_yield_curve("USD.FEDFUND.1D"))
        self.usd._freeze()
        with self.assertRaises(TypeError):
            self.usd.add(_yield_curve("USD.FEDFUND.1D"))
        self.assertIn("frozen", repr(self.usd))
        self.assertIn(SOFR, repr(self.usd))
