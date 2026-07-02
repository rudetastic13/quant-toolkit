"""Priceable functor: standalone instrument pricing through the compiled kernel path."""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Swap, curve_name
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, ZeroCurve
from finance.pricing.pricers import SwapPricer


def _market(as_of: Date, level: float = 0.04) -> MarketContext:
    origin = np.datetime64(as_of.to_str(), "D")
    nd = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 1825, 3650)], dtype="datetime64[D]")
    t = (nd.astype(np.int64) - origin.astype(np.int64)) / 365.0
    dfs = np.exp(-level * t)
    dfs[0] = 1.0
    ns = CurveNamespace()
    ns.bind(curve_name("USD", "SOFR"), ZeroCurve(nd, dfs, CurveInterpolator.LogLinearDF))
    return MarketContext(as_of_date=as_of, curves=ns)


class TestPriceableFunctor(UnitTest):
    COVERAGE = ["finance.instruments.priceable"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.swap = Swap(notional=10e6, rate_index="SOFR", fixed_rate=0.04, tenor="5Y", as_of=self.as_of)

    def test_call_matches_pricer_path(self):
        standalone = self.swap(self.market)
        portfolio = SwapPricer().compile([self.swap]).price(self.market)
        self.assertEqual(standalone.pv, portfolio.pv)
        np.testing.assert_array_equal(standalone.leg_pv, portfolio.leg_pv)

    def test_compile_cached_across_markets(self):
        self.swap(self.market)
        program_first = self.swap.__dict__["_program"]
        bumped = _market(self.as_of, level=0.05)
        pv_bumped = self.swap(bumped).pv
        self.assertIs(self.swap.__dict__["_program"], program_first)  # no recompile
        self.assertNotEqual(pv_bumped, self.swap(self.market).pv)     # but market matters

    def test_requests_default_includes_cashflows(self):
        res = self.swap(self.market)
        self.assertIsNotNone(res.cashflows)

    def test_requests_pv_only_skips_cashflows(self):
        res = self.swap(self.market, requests=["pv"])
        self.assertIsNone(res.cashflows)
        self.assertEqual(res.instrument_pv.shape, (1,))

    def test_unknown_request_raises(self):
        with self.assertRaises(ValueError):
            self.swap(self.market, requests=["pv", "delta_gamma_vega"])

    def test_scenario_loop_reprices_consistently(self):
        pvs = [self.swap(_market(self.as_of, level=lv)).pv for lv in (0.03, 0.04, 0.05)]
        # receiving 4% fixed: PV falls as rates rise
        self.assertGreater(pvs[0], pvs[1])
        self.assertGreater(pvs[1], pvs[2])
