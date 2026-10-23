"""The FTP chain: SOFR -> UST-OTR (asset-swap par spread) -> XXX-FTP (credit spread), built, calibrated, read back."""

import numpy as np
import pytest

from common.testing import UnitTest
from finance.conventions import default_registry
from finance.dates import Date
from finance.dates.enums import DayCountMethod
from finance.markets.context import MarketContext
from finance.markets.curves import YieldCurve, ZeroCurve, bp, dates_to_x
from finance.markets.rate_generator import RateGenerator
from finance.pricing.calibration import (
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    Quote,
    QuoteKind,
    SpreadHelper,
    spread_helper,
    swap_helper,
)

SOFR = "USD.SOFR.1D"
STDCSA = "USD.STDCSA"
UST = "USD.UST-OTR.1D"
FTP = "USD.XXX-FTP.1D"
TENORS = ("1Y", "2Y", "3Y", "5Y", "10Y")
ASW = 20 * bp  # UST over SOFR, flat
CREDIT = 35 * bp  # bank debt over UST, flat


@pytest.mark.calibration
class TestFtpChain(UnitTest):
    COVERAGE = [
        "finance.conventions.definitions",
        "finance.pricing.calibration.instruments",
        "finance.markets.rate_generator",
    ]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.origin = np.datetime64("2026-06-01", "D")
        self.ust_quotes = [spread_helper(spread=0.0, tenor=t, as_of=self.as_of) for t in TENORS]
        self.ftp_quotes = [spread_helper(spread=0.0, tenor=t, as_of=self.as_of, rate_index=FTP, over=UST) for t in TENORS]
        pillars = np.array(sorted(h.pillar_date for h in self.ust_quotes), dtype="datetime64[D]")
        x = np.concatenate([[0.0], dates_to_x(self.origin, pillars)])
        z = np.array([0.0, 0.035, 0.037, 0.039, 0.041, 0.044])
        dfs = np.exp(-z * x / 365.0)
        dfs[0] = 1.0
        self.sofr = ZeroCurve(x, dfs)
        truth = MarketContext(self.as_of)
        truth.add(YieldCurve.from_registry(self.origin, self.sofr, key=SOFR), funding_names=[STDCSA, "USD.USDSOFR"])
        truth[UST] = truth[SOFR] + ASW
        truth[FTP] = truth[UST] + CREDIT
        self.truth = truth.freeze()
        for h in (*self.ust_quotes, *self.ftp_quotes):
            h.quote = Quote(h.implied(self.truth), QuoteKind.ParSpread)

    def test_the_chain_families_are_overnight_act360(self):
        for key in (UST, FTP):
            family = default_registry.family(key)
            self.assertTrue(family.is_overnight)
            self.assertEqual(family.day_count_method, DayCountMethod.Actual360)
            self.assertEqual(default_registry.validate(key).name, family.name)
        self.assertEqual(list(self.truth), [SOFR, STDCSA, "USD.USDSOFR", UST, FTP])
        self.assertIs(self.truth.group("USD").discount("USDSOFR").zero_curve, self.sofr)

    def test_the_quote_is_the_bond_par_coupon_minus_the_swap_par_rate(self):
        helper = self.ust_quotes[2]
        self.assertIsInstance(helper, SpreadHelper)
        ust_par = swap_helper(rate=0.0, tenor="3Y", as_of=self.as_of, rate_index=UST).implied(self.truth)
        sofr_par = swap_helper(rate=0.0, tenor="3Y", as_of=self.as_of, rate_index=SOFR).implied(self.truth)
        self.assertAlmostEqual(helper.quote.value, ust_par - sofr_par, places=14)
        self.assertAlmostEqual(helper.quote.value, ASW, delta=1e-4)  # a flat 20bp log-DF spread read on the fixed leg's basis
        self.assertEqual(helper.curve, UST)
        self.assertEqual(helper.pillar_date, helper.swap.pillar_date)
        self.assertEqual(helper.quote.kind, QuoteKind.ParSpread)
        self.assertAlmostEqual(helper.residual(self.truth), 0.0, places=14)

    def test_sequential_calibration_recovers_both_spreads(self):
        market = MarketContext(self.as_of)
        market.add(YieldCurve.from_registry(self.origin, self.sofr, key=SOFR), funding_names=[STDCSA])
        ust_def = CurveDefinition(UST)
        market[UST] = market[SOFR] + ust_def.flat_on_pillars(market, self.ust_quotes)
        ust = CurveCalibrator(self.ust_quotes, GlobalSolver(), ust_def).calibrate(market)
        np.testing.assert_allclose(ust.zero_curve.params, ASW, atol=1e-9)
        self.assertLess(np.abs(ust.residuals).max(), 1e-10)

        market = MarketContext(self.as_of, ust.market)
        ftp_def = CurveDefinition(FTP)
        market[FTP] = market[UST] + ftp_def.flat_on_pillars(market, self.ftp_quotes)
        ftp = CurveCalibrator(self.ftp_quotes, GlobalSolver(), ftp_def).calibrate(market)
        np.testing.assert_allclose(ftp.zero_curve.params, CREDIT, atol=1e-9)
        self.assertLess(np.abs(ftp.residuals).max(), 1e-10)

        final = ftp.market
        self.assertIs(final[FTP]["base"], final[UST])  # FTP is still built on the UST object
        self.assertIs(final[UST]["base"], final[SOFR])  # ... which is still built on SOFR
        self.assertIs(final[STDCSA], final[SOFR])
        for h in self.ust_quotes:
            self.assertAlmostEqual(h.implied(final), h.quote.value, places=10)

    def test_bumps_follow_the_chain(self):
        at = np.array([1825.0])

        def moved(market, key):
            return market[key].zero_rate(at)[0] - self.truth[key].zero_rate(at)[0]

        up = self.truth.bump(SOFR, 1e-4)
        for key in (SOFR, STDCSA, UST, FTP):
            self.assertAlmostEqual(moved(up, key), 1e-4, places=12)
        self.assertIs(up[STDCSA], up[SOFR])

        up = self.truth.bump(UST, 1e-4)  # the ASW spread atom: UST and FTP move, SOFR does not
        self.assertIs(up[SOFR], self.sofr)
        self.assertAlmostEqual(moved(up, UST), 1e-4, places=12)
        self.assertAlmostEqual(moved(up, FTP), 1e-4, places=12)

        up = self.truth.bump(FTP, 1e-4)  # the credit spread atom: FTP alone
        self.assertIs(up[UST], self.truth[UST])
        self.assertAlmostEqual(moved(up, FTP), 1e-4, places=12)
        self.assertEqual(self.truth.owned(STDCSA), ())

    def test_spread_reads_the_chain_back_in_the_index_basis(self):
        rates = RateGenerator(self.truth)
        starts = np.array(["2026-12-01"], dtype="datetime64[D]")
        ends = np.array(["2027-03-01"], dtype="datetime64[D]")
        credit = rates.spread(FTP, UST, starts, ends)[0]
        self.assertEqual(credit, rates.simple_rate(FTP, starts, ends)[0] - rates.simple_rate(UST, starts, ends)[0])
        self.assertAlmostEqual(credit, CREDIT, delta=1.5e-4)  # the convention spread is near, not equal to, the flat log-DF spread
        over_sofr = rates.spread(FTP, SOFR, starts, ends)[0]
        self.assertAlmostEqual(over_sofr, ASW + CREDIT, delta=2.5e-4)
        self.assertAlmostEqual(over_sofr, credit + rates.spread(UST, SOFR, starts, ends)[0], places=15)  # additive
        self.assertEqual(rates.spread(STDCSA, SOFR, starts, ends, DayCountMethod.Actual360)[0], 0.0)  # same object
        with self.assertRaises(ValueError):
            rates.spread(STDCSA, SOFR, starts, ends)  # a discount key has no index basis
        with self.assertRaises(ValueError):
            rates.spread(FTP, "GBP.STDCSA", starts, ends)  # cross-currency
