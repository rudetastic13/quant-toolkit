"""MarketContext and RateGenerator responsibility tests."""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Swap
from finance.markets.context import MarketContext
from finance.markets.curves import CurveNamespace, YieldCurve, ZeroCurve
from finance.markets import HistoricalFixings
from finance.markets.rate_generator import RateGenerator
from finance.pricing.pricers import SwapPricer

CURVE = "USD.SOFR"


def _zero_curve(level: float = 0.04) -> ZeroCurve:
    x = np.array([0.0, 365.0, 1825.0])
    dfs = np.exp(-level * x / 365.0)
    dfs[0] = 1.0
    return ZeroCurve(x, dfs)


def _yield_curve(origin: np.datetime64, level: float = 0.04, index: str = "SOFR") -> YieldCurve:
    return YieldCurve.from_registry(
        origin,
        _zero_curve(level),
        currency="USD",
        index_name=index,
    )


class TestMarketContext(UnitTest):
    COVERAGE = ["finance.markets.context", "finance.markets.rate_generator"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.origin = np.datetime64("2026-06-01", "D")
        namespace = CurveNamespace()
        namespace.bind(_yield_curve(self.origin))
        self.market = MarketContext(as_of_date=self.as_of, curves=namespace)
        self.rates = RateGenerator(self.market)

    def test_market_resolves_yield_and_zero_curve(self):
        self.assertEqual(self.market.yield_curve(CURVE).name, CURVE)
        self.assertIs(self.market.zero_curve(CURVE), self.market.yield_curve(CURVE).zero_curve)

    def test_simple_rate_uses_index_convention(self):
        starts = np.array(["2026-12-01"], dtype="datetime64[D]")
        ends = np.array(["2027-03-01"], dtype="datetime64[D]")
        got = self.rates.simple_rate(CURVE, starts, ends)[0]
        curve = self.market.yield_curve(CURVE)
        df_start = curve.discount_factor(starts)[0]
        df_end = curve.discount_factor(ends)[0]
        tau = (ends[0] - starts[0]).astype(np.int64) / 360.0
        self.assertAlmostEqual(got, (df_start / df_end - 1.0) / tau, places=12)

    def test_zero_length_rate_window_is_zero(self):
        dates = np.array(["2027-06-01"], dtype="datetime64[D]")
        self.assertEqual(self.rates.simple_rate(CURVE, dates, dates)[0], 0.0)
        self.assertEqual(self.rates.continuous_forward_rate(CURVE, dates, dates)[0], 0.0)

    def test_rate_generator_overlays_fixings_before_curve(self):
        history = HistoricalFixings(
            dates=np.array(["2026-01-02", "2026-05-29"], dtype="datetime64[D]"),
            values=np.array([0.0311, 0.0311]),
        )
        market = self.market.with_curve(
            self.market.yield_curve(CURVE).with_historical_fixings(history)
        )
        starts = np.array(["2026-03-01", "2026-12-01"], dtype="datetime64[D]")
        ends = np.array(["2026-06-01", "2027-03-01"], dtype="datetime64[D]")
        got = RateGenerator(market).simple_rate(CURVE, starts, ends)
        self.assertAlmostEqual(got[0], 0.0311, places=12)
        self.assertAlmostEqual(got[1], self.rates.simple_rate(CURVE, starts[1:], ends[1:])[0], places=12)

    def test_historical_request_without_fixings_raises(self):
        starts = np.array(["2026-05-29"], dtype="datetime64[D]")
        ends = np.array(["2026-06-01"], dtype="datetime64[D]")
        with self.assertRaises(ValueError):
            self.rates.simple_rate(CURVE, starts, ends)

    def test_curve_origin_is_projected_even_when_history_contains_that_date(self):
        history = HistoricalFixings(
            dates=np.array(["2026-05-29", "2026-06-01"], dtype="datetime64[D]"),
            values=np.array([0.0311, 0.99]),
        )
        market = self.market.with_curve(
            self.market.yield_curve(CURVE).with_historical_fixings(history)
        )
        starts = np.array(["2026-06-01"], dtype="datetime64[D]")
        ends = np.array(["2026-06-02"], dtype="datetime64[D]")
        projected = RateGenerator(market).simple_rate(CURVE, starts, ends)[0]
        self.assertNotAlmostEqual(projected, 0.99)
        self.assertAlmostEqual(projected, self.rates.simple_rate(CURVE, starts, ends)[0])

    def test_with_curve_rebinds_without_mutating_original(self):
        bumped = self.market.with_curve(_yield_curve(self.origin, level=0.05))
        date = np.array(["2027-06-01"], dtype="datetime64[D]")
        self.assertAlmostEqual(bumped.yield_curve(CURVE).discount_factor(date)[0], np.exp(-0.05), places=10)
        self.assertAlmostEqual(self.market.yield_curve(CURVE).discount_factor(date)[0], np.exp(-0.04), places=10)

    def test_with_curve_binds_new_name(self):
        market = self.market.with_curve(_yield_curve(self.origin, level=0.045, index="FEDFUND"))
        self.assertIn("USD.FEDFUND", market.curves)
        self.assertIn(CURVE, market.curves)

    def test_continuous_forward_rate(self):
        starts = np.array(["2026-12-01"], dtype="datetime64[D]")
        ends = np.array(["2027-06-01"], dtype="datetime64[D]")
        self.assertAlmostEqual(
            self.rates.continuous_forward_rate(CURVE, starts, ends)[0],
            0.04,
            places=10,
        )

    def test_par_swap_rate_prices_unit_fixed_swap_to_zero(self):
        template = Swap.fixed_float_swap(
            notional=1e6,
            rate_index="SOFR",
            fixed_rate=0.0,
            tenor="3Y",
            as_of=self.as_of,
        )
        par = self.rates.par_swap_rate(template)
        swap = Swap.fixed_float_swap(
            notional=1e6,
            rate_index="SOFR",
            fixed_rate=par,
            tenor="3Y",
            as_of=self.as_of,
        )
        self.assertAlmostEqual(SwapPricer().price([swap], self.market).pv, 0.0, places=8)

    def test_convention_helper(self):
        conv = self.market.convention("USD", "SOFR")
        self.assertEqual(conv.index.label, "USD SOFR")
        self.assertEqual(conv.swap.spot_calendar, "no_holidays")
