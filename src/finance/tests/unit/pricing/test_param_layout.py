"""ParamLayout: one parameter block per owning key, shared atoms once, through the Numba adjoint and bumps."""

import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Swap
from finance.markets.context import MarketContext
from finance.markets.curves import CompositeCurve, YieldCurve, ZeroCurve, bp
from finance.pricing.engines.layout import ParamLayout
from finance.pricing.pricers import SwapPricer
from finance.pricing.risk import RiskEngine, Sensitivities
from finance.pricing.types import Backend

SOFR = "USD.SOFR.ON"
STDCSA = "USD.STDCSA"
FEDFUND = "USD.FEDFUND.ON"
TERM3M = "USD.CME-TERM-SOFR.3M"
TERM6M = "USD.CME-TERM-SOFR.6M"


def _sofr(as_of: Date) -> YieldCurve:
    origin = as_of.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 182, 365, 730, 1825, 3650)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.032, 0.035, 0.039, 0.043, 0.046])
    return YieldCurve.build(dates, np.exp(-z * t), key=SOFR)


def _market(as_of: Date) -> MarketContext:
    market = MarketContext(as_of)
    sofr = _sofr(as_of)
    market[SOFR] = sofr
    market[STDCSA] = sofr
    market[FEDFUND] = market[SOFR] - 5 * bp
    market[TERM3M] = market[SOFR] + 10 * bp
    market[TERM6M] = market[TERM3M] + 5 * bp
    return market.freeze()


class TestParamLayout(UnitTest):
    COVERAGE = ["finance.pricing.engines.layout"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.n = self.market[SOFR].n_params

    def test_shared_atom_is_one_block_named_after_its_owner(self):
        layout = ParamLayout.build(self.market, [FEDFUND, STDCSA])
        self.assertEqual(layout.blocks, (SOFR, FEDFUND))
        self.assertEqual(layout.n_params, self.n + 1)
        self.assertEqual(layout.block_slice(SOFR), slice(0, self.n))
        self.assertEqual(layout.block_slice(STDCSA), slice(0, self.n))  # alias reports under its owner
        self.assertEqual(layout.block_slice(FEDFUND), slice(self.n, self.n + 1))
        np.testing.assert_array_equal(layout.column_maps[FEDFUND], np.arange(self.n + 1))
        np.testing.assert_array_equal(layout.column_maps[STDCSA], np.arange(self.n))

    def test_three_deep_chain_is_three_blocks(self):
        layout = ParamLayout.build(self.market, [TERM6M, STDCSA])
        self.assertEqual(layout.blocks, (SOFR, TERM3M, TERM6M))
        self.assertEqual(layout.n_params, self.n + 2)
        # 6M's own parameters are [sofr | 3M spread | 6M spread]; the first two live in other blocks
        np.testing.assert_array_equal(layout.column_maps[TERM6M], np.r_[np.arange(self.n), self.n, self.n + 1])
        np.testing.assert_allclose(
            layout.params(self.market), np.r_[self.market[SOFR].params, 0.0010, 0.0005], rtol=1e-12
        )

    def test_a_key_spanning_blocks_it_does_not_own_cannot_report(self):
        ust = ZeroCurve.constant(0.03, 3650.0)
        market = MarketContext(self.as_of, {SOFR: self.market[SOFR], "USD.UST": ust})
        market["USD.BLEND"] = CompositeCurve((market[SOFR], market["USD.UST"]), weights=(0.5, 0.5))
        layout = ParamLayout.build(market, ["USD.BLEND"])
        self.assertEqual(layout.blocks, (SOFR, "USD.UST"))
        with self.assertRaisesRegex(KeyError, "spans several blocks"):
            layout.block_of("USD.BLEND")
        with self.assertRaisesRegex(KeyError, "no curve named"):
            layout.block_of("USD.FEDFUND.ON")

    def test_pillars_follow_the_block(self):
        layout = ParamLayout.build(self.market, [FEDFUND, STDCSA])
        np.testing.assert_array_equal(layout.pillar_dates(STDCSA), self.market.yield_curve(SOFR).node_dates[1:])
        np.testing.assert_allclose(layout.pillar_years(FEDFUND), [3650.0 / 365.0])

    def test_compatible_tracks_atoms_weights_and_ownership(self):
        layout = ParamLayout.build(self.market, [FEDFUND, STDCSA])
        self.assertTrue(layout.compatible(self.market.bump(SOFR, 1e-4), [FEDFUND, STDCSA]))
        self.assertTrue(layout.compatible(self.market.bump(FEDFUND, 1e-4), [FEDFUND, STDCSA]))
        regeared = self.market.with_curve(FEDFUND, self.market[SOFR] - ZeroCurve.constant(0.0005, 365.0))
        self.assertFalse(layout.compatible(regeared, [FEDFUND, STDCSA]))  # spread pillar moved
        realiased = self.market.with_curve(STDCSA, ZeroCurve.constant(0.03, 3650.0))
        self.assertFalse(layout.compatible(realiased, [FEDFUND, STDCSA]))  # STDCSA no longer SOFR's atom


@pytest.mark.risk
class TestLayoutThroughTheEngines(UnitTest):
    COVERAGE = [
        "finance.pricing.engines.numba.program",
        "finance.pricing.risk.adjoint",
        "finance.pricing.risk.sensitivities",
    ]

    def setUp(self):
        pytest.importorskip("numba")
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.n = self.market[SOFR].n_params
        self.ff_swap = Swap.fixed_float_swap(
            notional=100e6, rate_index="FEDFUND", fixed_rate=0.04, tenor="5Y", as_of=self.as_of
        )
        self.sofr_swap = Swap.fixed_float_swap(
            notional=100e6, rate_index="SOFR", fixed_rate=0.04, tenor="5Y", as_of=self.as_of
        )

    def _risk(self, swaps, market=None):
        from finance.pricing.risk import NumbaRisk

        return NumbaRisk(SwapPricer().compile(swaps), market or self.market)

    def test_fed_funds_swap_reports_one_sofr_ladder_and_one_basis_number(self):
        risk = self._risk([self.ff_swap])
        sofr_ladder = risk.zero_ladder(SOFR)
        basis = risk.zero_ladder(FEDFUND)
        self.assertEqual(sofr_ladder.shape, (1, self.n))
        self.assertEqual(basis.shape, (1, 1))
        # the SOFR ladder is projection risk through the composite plus discounting risk under STDCSA
        np.testing.assert_allclose(sofr_ladder, risk.index_ladder(SOFR) + risk.funding_ladder(STDCSA))
        # a flat spread is a parallel move of the projection curve: the basis equals the summed index ladder
        np.testing.assert_allclose(basis[0, 0], risk.index_ladder(SOFR).sum(), rtol=1e-10)
        self.assertEqual(risk.funding_ladder(FEDFUND).sum(), 0.0)  # nothing discounts off Fed Funds

    def test_adjoint_matches_bump_on_every_block(self):
        program = SwapPricer().compile([self.ff_swap, self.sofr_swap])
        risk = self._risk([self.ff_swap, self.sofr_swap])
        bump = Sensitivities(program, self.market)
        for key in (SOFR, FEDFUND, STDCSA):
            with self.subTest(key=key):
                np.testing.assert_allclose(
                    risk.zero_ladder(key), bump.key_rate_durations(key).krd, rtol=2e-4, atol=1e-3
                )
        # the SOFR swap carries no basis risk
        self.assertAlmostEqual(risk.zero_ladder(FEDFUND)[1, 0], 0.0)

    def test_three_deep_chain_through_the_engine(self):
        swap_6m = Swap.fixed_float_swap(
            notional=50e6, rate_index="CME-TERM-SOFR.6M", fixed_rate=0.04, tenor="3Y", as_of=self.as_of
        )
        program = SwapPricer().compile([swap_6m])
        risk = self._risk([swap_6m])
        self.assertEqual(risk.base.layout.blocks, (SOFR, TERM3M, TERM6M))
        bump = Sensitivities(program, self.market)
        for key in (SOFR, TERM3M, TERM6M):
            with self.subTest(key=key):
                np.testing.assert_allclose(
                    risk.zero_ladder(key), bump.key_rate_durations(key).krd, rtol=2e-4, atol=1e-3
                )
        np.testing.assert_allclose(
            risk.zero_ladder(TERM3M), risk.zero_ladder(TERM6M), rtol=1e-10
        )  # both spreads are parallel moves

    def test_prepared_program_reprices_a_bump_and_rejects_a_regear(self):
        program = SwapPricer().compile([self.ff_swap])
        prepared = program.prepare(self.market, backend=Backend.Numba)
        moved = self.market.bump(FEDFUND, 0.001)
        self.assertTrue(prepared.compatible(moved))
        self.assertAlmostEqual(prepared.params_from_market(moved)[-1], -0.0005 + 0.001, places=15)
        np.testing.assert_allclose(
            prepared.reprice(moved).instrument_pv, program.reprice(moved).instrument_pv, rtol=1e-11
        )
        regeared = self.market.with_curve(FEDFUND, self.market[SOFR] - ZeroCurve.constant(0.0005, 365.0))
        with self.assertRaises(ValueError):
            prepared.params_from_market(regeared)

    def test_cross_block_gamma(self):
        risk = self._risk([self.ff_swap])
        own = risk.gamma(FEDFUND)
        cross = risk.gamma(keys=(SOFR, FEDFUND))
        self.assertEqual(own.shape, (1, 1))
        self.assertEqual(cross.shape, (self.n + 1, self.n + 1))
        np.testing.assert_allclose(cross[-1, -1], own[0, 0], rtol=1e-8)
        np.testing.assert_allclose(cross[: self.n, : self.n], risk.gamma(SOFR), rtol=1e-8)

    def test_risk_engine_bump_and_adjoint_agree_on_the_basis(self):
        program = SwapPricer().compile([self.ff_swap])
        adjoint = RiskEngine(program, self.market)
        bump = RiskEngine(program, self.market, method="bump")
        for key in (SOFR, FEDFUND):
            with self.subTest(key=key):
                np.testing.assert_allclose(adjoint.ladder(key).ladder, bump.ladder(key).ladder, rtol=5e-4, atol=1e-3)
                np.testing.assert_array_equal(adjoint.ladder(key).pillar_dates, bump.ladder(key).pillar_dates)

    def test_jax_oracle_refuses_a_composite_market(self):
        pytest.importorskip("jax")
        from finance.pricing.engines.jax.program import JaxProgram

        with self.assertRaises(NotImplementedError):
            JaxProgram(SwapPricer().compile([self.ff_swap]).inputs, self.market)
