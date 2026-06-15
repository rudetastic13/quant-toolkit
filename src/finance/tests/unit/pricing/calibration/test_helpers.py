"""Calibration-instrument (helper) unit tests.

Each helper computes the model value of its quoted measure off a market.  Deposits/FRAs are
checked against the hand-computed DF-ratio simple rate; the swap helper is checked by the
par identity (a swap struck at the implied par rate prices to zero).
"""
import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date, period_fractions
from finance.instruments.resolution import Swap
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, ZeroCurve
from finance.pricing.calibration.instruments import deposit_helper, fra_helper, swap_helper
from finance.pricing.pricers.swap import SwapPricer

CURVE = "USD.SOFR"


def _curve(origin: str) -> ZeroCurve:
    dates = np.array(
        [origin, "2027-06-01", "2028-06-01", "2029-06-01", "2031-06-01", "2036-06-01"],
        dtype="datetime64[D]",
    )
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    dfs = np.exp(-0.04 * t)
    dfs[0] = 1.0
    return ZeroCurve(dates, dfs, CurveInterpolator.LogLinearDF)


@pytest.mark.calibration
class TestHelpers(UnitTest):
    COVERAGE = ["finance.pricing.calibration.instruments"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.curve = _curve("2026-06-01")
        ns = CurveNamespace()
        ns.bind(CURVE, self.curve)
        self.mkt = MarketContext(as_of_date=self.as_of, curves=ns)

    def _df_ratio_rate(self, eff, mat, dcm) -> float:
        s = np.array([eff.to_numpy()], dtype="datetime64[D]")
        e = np.array([mat.to_numpy()], dtype="datetime64[D]")
        df_s = self.curve.discount_factor(s)
        df_e = self.curve.discount_factor(e)
        tau = period_fractions(dcm, s, e)
        return float(((df_s / df_e - 1.0) / tau)[0])

    def test_deposit_implied_matches_df_ratio(self):
        h = deposit_helper(rate=0.0, tenor="3M", as_of=self.as_of)
        expected = self._df_ratio_rate(h.deposit.effective, h.deposit.maturity, h.deposit.day_count_method)
        self.assertAlmostEqual(h.implied(self.mkt), expected, places=12)

    def test_fra_implied_matches_df_ratio(self):
        h = fra_helper(rate=0.0, start="3M", end="6M", as_of=self.as_of)
        expected = self._df_ratio_rate(h.fra.effective, h.fra.maturity, h.fra.day_count_method)
        self.assertAlmostEqual(h.implied(self.mkt), expected, places=12)

    def test_swap_implied_is_par_rate(self):
        h = swap_helper(rate=0.0, tenor="3Y", as_of=self.as_of)
        par = h.implied(self.mkt)
        # A swap struck at the implied par rate must price to ~0.
        swap = Swap(notional=100.0, rate_index="SOFR", fixed_rate=par, tenor="3Y", as_of=self.as_of)
        pv = SwapPricer().price([swap], self.mkt).pv
        self.assertAlmostEqual(float(pv), 0.0, places=6)

    def test_residual_is_implied_minus_quote(self):
        h = deposit_helper(rate=0.03, tenor="6M", as_of=self.as_of)
        self.assertAlmostEqual(h.residual(self.mkt), h.implied(self.mkt) - 0.03, places=14)

    def test_pillar_date_is_maturity(self):
        h = deposit_helper(rate=0.0, tenor="6M", as_of=self.as_of)
        self.assertEqual(h.pillar_date, np.datetime64(h.deposit.maturity.to_numpy(), "D"))
