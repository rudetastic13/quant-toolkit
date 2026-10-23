"""MarketContext (a Mapping of curves by key, with ownership and propagating scenarios) and RateGenerator."""

import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Swap
from finance.markets import HistoricalFixings
from finance.markets.context import MarketContext
from finance.markets.curves import CompositeCurve, YieldCurve, ZeroCurve, bp
from finance.markets.rate_generator import RateGenerator
from finance.pricing.pricers import SwapPricer

CURVE = "USD.SOFR.1D"
DISCOUNT = "USD.STDCSA"
FEDFUND = "USD.FEDFUND.1D"
TERM3M = "USD.CME-TERM-SOFR.3M"
TERM6M = "USD.CME-TERM-SOFR.6M"
AT = np.array([1825.0])


def _zero_curve(level: float = 0.04) -> ZeroCurve:
    x = np.array([0.0, 365.0, 1825.0])
    dfs = np.exp(-level * x / 365.0)
    dfs[0] = 1.0
    return ZeroCurve(x, dfs)


def _yield_curve(origin: np.datetime64, level: float = 0.04, key: str = CURVE) -> YieldCurve:
    return YieldCurve.from_registry(origin, _zero_curve(level), key=key)


class TestMarketContext(UnitTest):
    COVERAGE = ["finance.markets.context", "finance.markets.rate_generator"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.origin = np.datetime64("2026-06-01", "D")
        self.sofr = _zero_curve()
        self.market = MarketContext(self.as_of, {CURVE: self.sofr, DISCOUNT: self.sofr})
        self.rates = RateGenerator(self.market)

    # -- mapping ------------------------------------------------------------------------

    def test_market_is_a_mapping_of_discount_curves(self):
        self.assertEqual(list(self.market), [CURVE, DISCOUNT])
        self.assertEqual(len(self.market), 2)
        self.assertIs(self.market[CURVE], self.sofr)
        self.assertIs(self.market["usd.sofr.1d"], self.sofr)  # keys are normalised
        self.assertIn("usd.stdcsa", self.market)
        self.assertNotIn(FEDFUND, self.market)
        self.assertNotIn(42, self.market)
        with self.assertRaises(KeyError):
            self.market[FEDFUND]
        self.assertEqual(self.market.as_of, self.as_of)
        self.assertEqual(self.market.origin, self.origin)
        self.assertIn("USD.SOFR.1D", repr(self.market))

    def test_market_resolves_yield_and_zero_curve(self):
        view = self.market.yield_curve(CURVE)
        self.assertEqual(view.name, CURVE)
        self.assertEqual(view.origin, self.origin)
        self.assertIs(view.zero_curve, self.market[CURVE])
        self.assertIs(self.market[CURVE], self.sofr)
        self.assertIs(self.market.yield_curve(CURVE), view)  # cached until the curve or fixings change
        self.assertIsNone(self.market.yield_curve(DISCOUNT).conventions)

    def test_assignment_validates_the_key_against_the_conventions(self):
        with self.assertRaises(KeyError):
            self.market["USD.NOSUCHINDEX.1D"] = self.sofr
        with self.assertRaises(ValueError):
            self.market["USD"] = self.sofr
        with self.assertRaises(TypeError):
            self.market["USD.UST"] = object()
        self.assertEqual(len(self.market), 2)

    def test_a_yield_curve_registers_its_curve_and_its_fixings(self):
        history = HistoricalFixings(dates=np.array(["2026-05-29"], dtype="datetime64[D]"), values=np.array([0.0311]))
        seasoned = YieldCurve.from_registry(self.origin, _zero_curve(0.05), key=CURVE, historical_fixings=history)
        market = MarketContext(self.as_of, {CURVE: seasoned, DISCOUNT: seasoned})
        self.assertIs(market[CURVE], seasoned.zero_curve)
        self.assertIs(market.fixings[CURVE], history)
        self.assertIs(market.yield_curve(CURVE).historical_fixings, history)
        self.assertIsNone(market.yield_curve(DISCOUNT).historical_fixings)
        wrong_origin = YieldCurve.from_registry(np.datetime64("2026-06-02"), self.sofr, key=CURVE)
        with self.assertRaises(ValueError):
            self.market[CURVE] = wrong_origin

    def test_with_fixings_returns_a_copy(self):
        history = HistoricalFixings(dates=np.array(["2026-05-29"], dtype="datetime64[D]"), values=np.array([0.0311]))
        seasoned = self.market.with_fixings(CURVE, history)
        self.assertIs(seasoned.yield_curve(CURVE).historical_fixings, history)
        self.assertIsNone(self.market.yield_curve(CURVE).historical_fixings)
        self.assertIsNone(seasoned.with_fixings(CURVE, None).yield_curve(CURVE).historical_fixings)
        with self.assertRaises(KeyError):
            self.market.with_fixings(FEDFUND, history)

    # -- ownership ----------------------------------------------------------------------

    def test_an_atom_under_two_discount_keys_belongs_to_the_first_registered(self):
        ust = _zero_curve(0.03)
        market = MarketContext(self.as_of, {"USD.UST": ust, DISCOUNT: ust})
        self.assertEqual(market.leaf_owner()[id(ust)], "USD.UST")
        moved = _zero_curve(0.035)
        self.assertIs(market.with_curve("USD.UST", moved)[DISCOUNT], moved)   # the owner moved: the alias follows
        self.assertIs(market.with_curve(DISCOUNT, moved)["USD.UST"], ust)     # the alias rebinds alone

    def _chain(self) -> MarketContext:
        market = MarketContext(self.as_of, {CURVE: self.sofr, DISCOUNT: self.sofr})
        market[FEDFUND] = market[CURVE] - 5 * bp
        market[TERM3M] = market[CURVE] + 10 * bp
        market[TERM6M] = market[TERM3M] + 5 * bp
        return market

    def test_leaf_owner_is_the_shortest_path_then_projection_then_first_registered(self):
        market = self._chain()
        owner = market.leaf_owner()
        self.assertEqual(owner[id(self.sofr)], CURVE)
        self.assertEqual(owner[id(market[FEDFUND]["spread"])], FEDFUND)
        self.assertEqual(owner[id(market[TERM3M]["spread"])], TERM3M)
        self.assertEqual(owner[id(market[TERM6M]["spread"])], TERM6M)
        self.assertEqual(sorted(owner.values()), sorted([CURVE, FEDFUND, TERM3M, TERM6M]))
        self.assertEqual(market.owned(CURVE), (self.sofr,))
        self.assertEqual(market.owned(FEDFUND), (market[FEDFUND]["spread"],))
        self.assertEqual(market.owned(TERM6M), (market[TERM6M]["spread"],))
        self.assertEqual(market.owned(DISCOUNT), ())

    def test_a_key_owns_every_atom_registered_through_it_alone(self):
        turn = ZeroCurve.constant(0.001, 365.0)
        market = MarketContext(self.as_of, {CURVE: CompositeCurve({"smooth": self.sofr, "turn": turn})})
        self.assertEqual(market.owned(CURVE), (self.sofr, turn))

    def test_ambiguous_ownership_is_a_bind_time_error_and_leaves_the_market_as_it_was(self):
        market = MarketContext(self.as_of)
        market[FEDFUND] = self.sofr - 5 * bp
        with self.assertRaises(ValueError):
            market[TERM3M] = self.sofr + 10 * bp  # two projection keys reach the SOFR atom at depth 1
        self.assertEqual(list(market), [FEDFUND])
        market[CURVE] = self.sofr  # registering the atom resolves it
        market[TERM3M] = self.sofr + 10 * bp
        self.assertEqual(market.leaf_owner()[id(self.sofr)], CURVE)

    # -- scenarios ----------------------------------------------------------------------

    def test_with_curve_rebinds_without_mutating_original(self):
        bumped = self.market.with_curve(CURVE, _zero_curve(0.05))
        date = np.array(["2027-06-01"], dtype="datetime64[D]")
        self.assertAlmostEqual(bumped.yield_curve(CURVE).discount_factor(date)[0], np.exp(-0.05), places=10)
        self.assertAlmostEqual(self.market.yield_curve(CURVE).discount_factor(date)[0], np.exp(-0.04), places=10)
        self.assertTrue(bumped.frozen)
        self.assertFalse(self.market.frozen)

    def test_with_curve_accepts_a_yield_curve_under_its_own_key(self):
        market = self.market.with_curve(_yield_curve(self.origin, level=0.045, key=FEDFUND))
        self.assertIn(FEDFUND, market)
        self.assertNotIn(FEDFUND, self.market)
        with self.assertRaises(TypeError):
            self.market.with_curve(CURVE)

    def test_with_curve_on_the_owner_moves_every_key_holding_the_object(self):
        moved = _zero_curve(0.05)
        bumped = self.market.with_curve(CURVE, moved)
        self.assertIs(bumped[DISCOUNT], moved)
        self.assertIs(bumped[CURVE], moved)
        self.assertIs(self.market[DISCOUNT], self.sofr)

    def test_with_curve_on_an_alias_rebinds_only_that_key(self):
        ust = _zero_curve(0.03)
        market = self.market.with_curve(DISCOUNT, ust)
        self.assertIs(market[DISCOUNT], ust)
        self.assertIs(market[CURVE], self.sofr)

    def test_with_curve_propagates_through_every_composite_built_on_the_moved_curve(self):
        market = self._chain()
        moved = _zero_curve(0.05)
        out = market.with_curve(CURVE, moved)
        self.assertIs(out[FEDFUND]["base"], moved)
        self.assertIs(out[TERM6M]["base"]["base"], moved)
        self.assertIs(out[TERM6M]["spread"], market[TERM6M]["spread"])  # spreads untouched
        self.assertIs(market[FEDFUND]["base"], self.sofr)  # original untouched

    def test_redefining_a_key_in_terms_of_itself_is_rejected(self):
        with self.assertRaises(ValueError):
            self.market[CURVE] = self.market[CURVE] + 1 * bp
        with self.assertRaises(ValueError):
            self.market.with_curve(CURVE, self.market[CURVE] + 1 * bp)
        self.assertIs(self.market[CURVE], self.sofr)

    def test_bump_sofr_moves_fed_funds_and_leaves_the_basis_alone(self):
        market = self._chain()
        up = market.bump(CURVE, 1e-4)
        for key in (CURVE, DISCOUNT, FEDFUND, TERM3M, TERM6M):
            np.testing.assert_allclose(up[key].zero_rate(AT) - market[key].zero_rate(AT), 1e-4, rtol=1e-9)
        basis_before = market[FEDFUND].zero_rate(AT) - market[CURVE].zero_rate(AT)
        basis_after = up[FEDFUND].zero_rate(AT) - up[CURVE].zero_rate(AT)
        np.testing.assert_allclose(basis_after, basis_before, atol=1e-15)
        self.assertIs(up[DISCOUNT], up[CURVE])
        self.assertIs(market[CURVE], self.sofr)

    def test_bump_fed_funds_moves_the_spread_only(self):
        market = self._chain()
        up = market.bump(FEDFUND, 1e-4)
        np.testing.assert_allclose(up[FEDFUND].zero_rate(AT) - market[FEDFUND].zero_rate(AT), 1e-4, rtol=1e-9)
        self.assertIs(up[CURVE], self.sofr)
        self.assertIs(up[TERM3M], market[TERM3M])

    def test_bump_vector_lays_out_over_the_owned_parameters(self):
        shift = np.array([1e-4, 2e-4])
        up = self.market.bump(CURVE, shift)
        np.testing.assert_allclose(up[CURVE].params, self.sofr.params + shift)
        with self.assertRaises(ValueError):
            self.market.bump(CURVE, np.array([1e-4]))
        with self.assertRaises(ValueError):
            self.market.bump(DISCOUNT, 1e-4)  # owns nothing: the SOFR key owns the object

    def test_freeze_makes_the_market_read_only(self):
        self.assertIs(self.market.freeze(), self.market)
        self.assertTrue(self.market.frozen)
        with self.assertRaises(TypeError):
            self.market[FEDFUND] = self.sofr - 5 * bp
        with self.assertRaises(TypeError):
            self.market.fixings[CURVE] = None
        self.assertIn(FEDFUND, self.market.with_curve(FEDFUND, self.sofr - 5 * bp))

    def test_to_mermaid_lists_keys_components_and_aliases(self):
        text = self._chain().to_mermaid()
        self.assertTrue(text.startswith("flowchart LR"))
        for key in (CURVE, DISCOUNT, FEDFUND, TERM3M, TERM6M):
            self.assertIn(f'"{key}"', text)
        self.assertIn("same object", text)
        self.assertIn("base w=1", text)
        self.assertIn("spread w=1", text)

    # -- rates ----------------------------------------------------------------------------

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
        market = self.market.with_fixings(CURVE, history)
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
        market = self.market.with_fixings(CURVE, history)
        starts = np.array(["2026-06-01"], dtype="datetime64[D]")
        ends = np.array(["2026-06-02"], dtype="datetime64[D]")
        projected = RateGenerator(market).simple_rate(CURVE, starts, ends)[0]
        self.assertNotAlmostEqual(projected, 0.99)
        self.assertAlmostEqual(projected, self.rates.simple_rate(CURVE, starts, ends)[0])

    def test_continuous_forward_rate(self):
        starts = np.array(["2026-12-01"], dtype="datetime64[D]")
        ends = np.array(["2027-06-01"], dtype="datetime64[D]")
        self.assertAlmostEqual(self.rates.continuous_forward_rate(CURVE, starts, ends)[0], 0.04, places=10)

    def test_par_swap_rate_prices_unit_fixed_swap_to_zero(self):
        template = Swap.fixed_float_swap(notional=1e6, rate_index="USD.SOFR.1D", fixed_rate=0.0, tenor="3Y", as_of=self.as_of)
        par = self.rates.par_swap_rate(template)
        swap = Swap.fixed_float_swap(notional=1e6, rate_index="USD.SOFR.1D", fixed_rate=par, tenor="3Y", as_of=self.as_of)
        self.assertAlmostEqual(SwapPricer().price([swap], self.market).pv, 0.0, places=8)

    def test_simple_rate_on_a_discount_curve_needs_a_day_count(self):
        from finance.dates.enums import DayCountMethod

        market = self.market.with_curve("USD.UST", _zero_curve(0.03))
        starts = np.array(["2026-12-01"], dtype="datetime64[D]")
        ends = np.array(["2027-03-01"], dtype="datetime64[D]")
        with self.assertRaises(ValueError):
            RateGenerator(market).simple_rate("USD.UST", starts, ends)
        got = RateGenerator(market).simple_rate("USD.UST", starts, ends, DayCountMethod.Actual360)[0]
        self.assertGreater(got, 0.0)
