"""Priceable functor: standalone instrument pricing through the compiled kernel path."""
import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.dates import Date
from quant_toolkit.finance.instruments.resolution import Swap
from quant_toolkit.finance.markets.context import MarketContext
from quant_toolkit.finance.markets.curves import YieldCurve
from quant_toolkit.finance.pricing.pricers import SwapPricer


def _market(as_of: Date, level: float = 0.04) -> MarketContext:
    origin = np.datetime64(as_of.to_str(), "D")
    nd = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 1825, 3650)], dtype="datetime64[D]")
    t = (nd.astype(np.int64) - origin.astype(np.int64)) / 365.0
    dfs = np.exp(-level * t)
    dfs[0] = 1.0
    sofr = YieldCurve.build(nd, dfs, key="USD.SOFR.1D")
    return MarketContext(as_of=as_of, curves={sofr.name: sofr, "USD.STDCSA": sofr})


class TestPriceableFunctor(UnitTest):

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.swap = Swap.fixed_float_swap(notional=10e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=self.as_of)

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

    def test_request_for_another_product_reads_none(self):
        res = self.swap(self.market, requests=["pv", "vega", "pnl"])
        np.testing.assert_array_equal(res.get("pv"), res.instrument_pv)
        self.assertIsNone(res.get("vega"))
        self.assertIsNone(res.get("pnl"))
        self.assertIsNone(res.get("cashflows"))  # answered by swaps, but not requested

    def test_unknown_request_raises(self):
        with self.assertRaises(ValueError):
            self.swap(self.market, requests=["pv", "delta_gamma_vega"])

    def test_scenario_loop_reprices_consistently(self):
        pvs = [self.swap(_market(self.as_of, level=lv)).pv for lv in (0.03, 0.04, 0.05)]
        # receiving 4% fixed: PV falls as rates rise
        self.assertGreater(pvs[0], pvs[1])
        self.assertGreater(pvs[1], pvs[2])

    def test_ladders_are_on_demand_and_split_total(self):
        self.assertEqual(self.swap(self.market).extras, {})  # default requests run no adjoint
        res = self.swap(self.market, requests=["pv", "zero_ladder", "index_ladder", "funding_ladder"])
        zero, index, funding = (res.get(name) for name in ("zero_ladder", "index_ladder", "funding_ladder"))
        self.assertEqual(zero.values.shape[0], 1)
        np.testing.assert_allclose(zero.values, index.values + funding.values, rtol=1e-12, atol=1e-9)
        bumped = self.swap(_market(self.as_of, level=0.0401)).pv - self.swap(self.market).pv
        self.assertAlmostEqual(zero.dv01.sum() / bumped, 1.0, delta=0.02)
