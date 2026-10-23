"""R2: RiskTemplate — identity, bump oracle, hedge, futures on a swap template, refusals, bases, multi-curve, units."""

import numpy as np
import pytest

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import SofrFuture, Swap
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, YieldCurve, bp, dates_to_x
from finance.pricing.calibration import (
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    Quote,
    deposit_helper,
    fra_helper,
    futures_helper,
    swap_helper,
)
from finance.pricing.pricers import FuturesPricer, SwapPricer
from finance.pricing.risk import PillarRisk, pillar_risk
from finance.pricing.risk.basis import LeastSquaresBasis, SchemeBasis, TentBasis
from finance.pricing.risk.template import RiskTemplate, TemplateError, TemplateLimits

SOFR = "USD.SOFR.1D"
STDCSA = "USD.STDCSA"
FEDFUND = "USD.FEDFUND.1D"
AS_OF = Date(2026, 6, 1)

pytest.importorskip("numba")


def _quoted(helpers, market):
    for h in helpers:
        h.quote = Quote(float(h.implied(market)), h.quote.kind)
    return helpers


def _calibration_set():
    return [deposit_helper(rate=0.0, tenor="3M", as_of=AS_OF), deposit_helper(rate=0.0, tenor="6M", as_of=AS_OF)] + [
        swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y")
    ]


def _market() -> MarketContext:
    """SOFR on the calibration set's pillars, STDCSA aliased, Fed Funds 5bp under."""
    helpers = _calibration_set()
    origin = AS_OF.to_numpy()
    pillars = np.array(sorted(h.pillar_date for h in helpers), dtype="datetime64[D]")
    dates = np.concatenate([[origin], pillars])
    t = dates_to_x(origin, dates) / 365.0
    z = np.array([0.0, 0.041, 0.040, 0.039, 0.0385, 0.038, 0.039, 0.040, 0.042, 0.044, 0.0445])
    sofr = YieldCurve.build(dates, np.exp(-z * t), key=SOFR)
    market = MarketContext(AS_OF)
    market[SOFR] = sofr
    market[STDCSA] = sofr
    market[FEDFUND] = market[SOFR] - 5 * bp
    return market.freeze()


def _book():
    return [
        Swap.fixed_float_swap(notional=100e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=AS_OF),
        Swap.fixed_float_swap(notional=-40e6, rate_index="USD.SOFR.1D", fixed_rate=0.042, tenor="12Y", as_of=AS_OF),
        Swap.fixed_float_swap(notional=25e6, rate_index="USD.SOFR.1D", fixed_rate=0.041, tenor="18M", as_of=AS_OF),
    ]


@pytest.mark.risk
class TestRiskTemplate(UnitTest):
    COVERAGE = ["finance.pricing.risk.template", "finance.pricing.risk.basis"]

    def setUp(self):
        self.market = _market()
        self.program = SwapPricer().compile(_book())
        self.pillar = pillar_risk(self.program, self.market, labels=("5Y recv", "12Y pay", "18M recv"))
        self.identity = RiskTemplate.identity(_quoted(_calibration_set(), self.market), name="identity")

    # 1
    def test_identity_template_reproduces_partial_dv01(self):
        bound = self.identity.bind(self.market)
        risk = bound.quote_risk(self.pillar)
        G, J = self.pillar.block(SOFR), bound.jacobian[:, bound.slice(SOFR)]
        np.testing.assert_allclose(risk.values, G @ np.linalg.inv(J), rtol=1e-10, atol=1e-12 * np.abs(G).max())
        np.testing.assert_allclose(risk.unexplained, 0.0, atol=1e-9 * np.abs(G).max())
        np.testing.assert_allclose(risk.pillar_parallel, G.sum(axis=1))
        self.assertEqual(risk.curves, (SOFR,))
        self.assertEqual(risk.uncovered, {})
        self.assertEqual(risk.stamp.template, "identity")
        self.assertEqual(risk.stamp.pillars, (10,))

    # 2
    def test_template_bump_oracle(self):
        template = RiskTemplate(
            _quoted([swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in ("1Y", "3Y", "7Y", "20Y")], self.market)
        )
        bound = template.bind(self.market)
        risk = bound.quote_risk(self.pillar)
        eps = 1e-4  # a smaller step only amplifies roundoff in the compounded coupons; the adjoint is exact
        for k in range(4):
            up, dn = bound.apply(self.market, k, eps), bound.apply(self.market, k, -eps)
            pv = (self.program.reprice(up).instrument_pv - self.program.reprice(dn).instrument_pv) / (2 * eps)
            np.testing.assert_allclose(pv, risk.values[:, k], rtol=1e-6, atol=1e-3)
            moved = [(h.implied(up) - h.implied(dn)) / (2 * eps) for h in template.instruments]
            np.testing.assert_allclose(moved, np.eye(4)[k], atol=1e-7)  # J_T A = I on the real pricer

    # 3
    def test_hedge_flattens_template_risk(self):
        bound = RiskTemplate(
            _quoted([swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in ("2Y", "5Y", "10Y", "30Y")], self.market)
        ).bind(self.market)
        risk = bound.quote_risk(self.pillar)
        h = bound.hedge(risk)
        hedged_pillar = self.pillar.block(SOFR) + h @ bound.unit_risk
        hedged = PillarRisk(
            (SOFR,), np.array([0, 10]), {SOFR: self.pillar.pillar_dates[SOFR]}, hedged_pillar, self.pillar.pv
        )
        np.testing.assert_allclose(bound.quote_risk(hedged).values, 0.0, atol=1e-6 * np.abs(risk.values).max())
        np.testing.assert_allclose(hedged_pillar, risk.unexplained, rtol=1e-6, atol=1e-6 * np.abs(hedged_pillar).max())

    # 4
    def test_futures_strip_shows_as_swap_risk(self):
        base_helpers = (
            [deposit_helper(rate=0.0, tenor="3M", as_of=AS_OF), deposit_helper(rate=0.0, tenor="6M", as_of=AS_OF)]
            + [
                futures_helper(price=96.0, contract="SR3", month=m, as_of=AS_OF)
                for m in ("Z26", "H27", "M27", "U27", "Z27", "H28", "M28")
            ]
            + [swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in ("3Y", "4Y", "5Y", "7Y", "10Y", "20Y", "30Y")]
        )
        _quoted(base_helpers, self.market)
        base = (
            CurveCalibrator(
                base_helpers,
                GlobalSolver(),
                CurveDefinition(SOFR, CurveInterpolator.LogLinearDF, discount_keys=(STDCSA,)),
            )
            .calibrate(MarketContext(AS_OF))
            .market
        )
        tenors = ("3M", "6M", "9M", "12M", "15M", "18M", "21M", "24M", "3Y", "4Y", "5Y", "7Y", "10Y", "20Y", "30Y")
        template = RiskTemplate(
            _quoted([swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in tenors], base), labels=tenors, name="SOFR par"
        )
        bound = template.bind(base)
        strip = FuturesPricer().compile(
            [
                SofrFuture.from_contract_month(contract="SR3", month=m, as_of=AS_OF, contracts=100)
                for m in ("Z26", "H27", "M27", "U27", "Z27")
            ]
        )
        risk = bound.quote_risk(pillar_risk(strip, base))
        total = np.abs(risk.values).sum()
        beyond = np.abs(risk.values[:, tenors.index("3Y") :]).sum()
        self.assertLess(beyond, 0.01 * total)
        np.testing.assert_allclose(risk.template_parallel, risk.pillar_parallel, rtol=0.1)  # two different parallels
        self.assertGreater(np.abs(risk.unexplained).max(), 0.0)
        self.assertTrue((risk.dv01.sum(axis=1) < 0).all())  # long futures: rates up, P&L down
        self.assertEqual(bound.diagnostics.warnings, ())

    # 5
    def test_too_fine_template_is_refused(self):
        template = RiskTemplate(
            _quoted([swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in ("1M", "2M", "5Y")], self.market)
        )
        with self.assertRaises(TemplateError) as ctx:
            template.bind(self.market)
        self.assertEqual(ctx.exception.code, "RANK_DEFICIENT")
        self.assertEqual(ctx.exception.node, template.label(0))

    # 6
    def test_k_greater_than_p_is_refused(self):
        tenors = ("1M", "2M", "3M", "4M", "5M", "6M", "9M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y")
        template = RiskTemplate(_quoted([swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in tenors], self.market))
        with self.assertRaises(TemplateError) as ctx:
            template.bind(self.market)
        self.assertEqual(ctx.exception.code, "K_GT_P")

    def test_unsorted_nodes_and_missing_curve_are_refused(self):
        helpers = _quoted([swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in ("5Y", "2Y")], self.market)
        with self.assertRaises(TemplateError) as ctx:
            RiskTemplate(helpers).bind(self.market)
        self.assertEqual(ctx.exception.code, "UNSORTED_NODES")
        other = RiskTemplate([swap_helper(rate=0.0, tenor="5Y", as_of=AS_OF, rate_index="USD.CME-TERM-SOFR.3M")])
        with self.assertRaises(TemplateError) as ctx:
            other.bind(self.market)
        self.assertEqual(ctx.exception.code, "MISSING_CURVE")

    # 7
    def test_condition_and_leverage(self):
        bound = self.identity.bind(self.market)
        self.assertLessEqual(bound.diagnostics.leverage.max(), 2.0)  # a 1bp par move moves a long pillar ~1.6bp
        self.assertLess(bound.diagnostics.condition, 1e3)
        self.assertEqual(bound.diagnostics.rank, {SOFR: 10})
        with self.assertRaises(TemplateError) as ctx:
            RiskTemplate(self.identity.instruments, limits=TemplateLimits(max_leverage=0.5)).bind(self.market)
        self.assertEqual(ctx.exception.code, "LEVERAGE")
        with self.assertRaises(TemplateError) as ctx:
            RiskTemplate(self.identity.instruments, limits=TemplateLimits(max_condition=1.0)).bind(self.market)
        self.assertEqual(ctx.exception.code, "ILL_CONDITIONED")
        warned = RiskTemplate(
            self.identity.instruments, limits=TemplateLimits(warn_leverage=0.5, warn_condition=1.0)
        ).bind(self.market)
        self.assertTrue(any("leverage" in w for w in warned.diagnostics.warnings))
        self.assertTrue(any("cond(M)" in w for w in warned.diagnostics.warnings))
        off_quote = RiskTemplate(
            [swap_helper(rate=0.05, tenor="5Y", as_of=AS_OF)], limits=TemplateLimits(max_abs_residual=1e-4)
        )
        with self.assertRaises(TemplateError) as ctx:
            off_quote.bind(self.market)
        self.assertEqual(ctx.exception.code, "RESIDUAL")

    # 8
    def test_least_squares_basis_matches_the_july_harness(self):
        fras = _quoted(
            [
                fra_helper(rate=0.0, start=a, end=b, as_of=AS_OF)
                for a, b in (("3M", "6M"), ("6M", "1Y"), ("1Y", "2Y"), ("2Y", "3Y"), ("3Y", "5Y"), ("5Y", "7Y"))
            ],
            self.market,
        )
        bound = RiskTemplate(fras, basis=LeastSquaresBasis()).bind(self.market)
        risk = bound.quote_risk(self.pillar)
        g, J = self.pillar.block(SOFR), bound.jacobian
        partials = g @ np.linalg.pinv(J)  # research/risk_reconciliation.reconcile
        np.testing.assert_allclose(risk.values, partials, rtol=1e-8, atol=1e-10 * np.abs(partials).max())
        np.testing.assert_allclose(
            np.abs(risk.unexplained).max(axis=1), np.abs(g - partials @ J).max(axis=1), rtol=1e-8, atol=1e-9
        )

    # 9
    def test_tent_basis(self):
        base_x = np.array([30.0, 91.0, 182.0, 365.0, 730.0, 1825.0, 3650.0])
        node_x = np.array([91.0, 365.0, 1825.0])
        B = TentBasis().matrix(base_x, node_x, jacobian_block=np.zeros((3, 7)))
        np.testing.assert_allclose(B.sum(axis=1), 1.0)
        self.assertTrue((B >= 0).all())
        np.testing.assert_allclose(B[0], [1.0, 0.0, 0.0])  # flat before the first node
        np.testing.assert_allclose(B[-1], [0.0, 0.0, 1.0])  # flat after the last
        np.testing.assert_allclose(B[2, :2], [(365 - 182) / (365 - 91), (182 - 91) / (365 - 91)])
        self.assertEqual(TentBasis.unsupported_nodes(base_x, node_x), [])
        self.assertEqual(TentBasis.unsupported_nodes(np.array([91.0, 3650.0]), np.array([30.0, 60.0, 91.0])), [0, 1])
        scheme = SchemeBasis().matrix(base_x, node_x, jacobian_block=np.zeros((3, 7)))
        self.assertEqual(scheme.shape, (7, 3))
        np.testing.assert_allclose(scheme.sum(axis=1), 1.0, atol=1e-12)  # rate-linear bumps also partition unity

    # 11
    def test_multi_curve_block_structure(self):
        sofr_nodes = _quoted(
            [swap_helper(rate=0.0, tenor=t, as_of=AS_OF) for t in ("1Y", "3Y", "5Y", "10Y")], self.market
        )
        ff_nodes = _quoted([swap_helper(rate=0.0, tenor="5Y", as_of=AS_OF, rate_index="USD.FEDFUND.1D")], self.market)
        template = RiskTemplate(sofr_nodes + ff_nodes, labels=("1Y", "3Y", "5Y", "10Y", "FF 5Y"))
        bound = template.bind(self.market)
        self.assertEqual(bound.curves, (SOFR, FEDFUND))
        M = bound.jacobian @ bound.basis
        np.testing.assert_array_equal(M[:4, 4], 0.0)  # SOFR quotes ignore the basis node
        self.assertNotEqual(M[4, :4].tolist(), [0.0] * 4)  # the FF par rate depends on SOFR nodes
        ff_swap = Swap.fixed_float_swap(notional=100e6, rate_index="USD.FEDFUND.1D", fixed_rate=0.04, tenor="5Y", as_of=AS_OF)
        program = SwapPricer().compile([ff_swap])
        risk = bound.quote_risk(pillar_risk(program, self.market))
        self.assertLess(risk.values[0, 4], 0.0)  # receiver: basis up, PV down
        eps = 1e-5  # the basis node is a parallel move of the projection curve: a larger step shows its convexity
        for k in (2, 4):
            up, dn = bound.apply(self.market, k, eps), bound.apply(self.market, k, -eps)
            pv = (program.reprice(up).instrument_pv - program.reprice(dn).instrument_pv) / (2 * eps)
            np.testing.assert_allclose(pv, risk.values[:, k], rtol=2e-6, atol=1e-3)

    def test_uncovered_curves_are_reported_not_dropped(self):
        ff_swap = Swap.fixed_float_swap(notional=100e6, rate_index="USD.FEDFUND.1D", fixed_rate=0.04, tenor="5Y", as_of=AS_OF)
        pillar = pillar_risk(SwapPricer().compile([ff_swap]), self.market)
        risk = self.identity.bind(self.market).quote_risk(pillar)
        self.assertEqual(list(risk.uncovered), [FEDFUND])
        np.testing.assert_array_equal(risk.uncovered[FEDFUND], pillar.block(FEDFUND))

    # 12
    def test_signs_units_and_report(self):
        bound = self.identity.bind(self.market)
        risk = bound.quote_risk(self.pillar)
        self.assertLess(risk.dv01[0].sum(), 0.0)  # receiver
        self.assertGreater(risk.dv01[1].sum(), 0.0)  # payer
        np.testing.assert_array_equal(risk.dv01, risk.values * 1e-4)
        frame = risk.to_pandas()
        self.assertEqual(list(frame.index), ["5Y recv", "12Y pay", "18M recv"])
        self.assertIn("template_parallel_dv01", frame.columns)
        self.assertEqual(len(risk.kinds), 10)
        np.testing.assert_allclose(bound.rebucket(self.pillar), self.pillar.block(SOFR) @ bound.basis)
        self.assertTrue(bound.matches(self.market))
        self.assertFalse(bound.matches(self.market.bump(SOFR, 1e-4)))
        self.assertTrue(bound.matches(self.market.bump(FEDFUND, 1e-4)))  # the SOFR atom did not move

    def test_from_spec_round_trip(self):
        spec = {
            "name": "SOFR par v1",
            "basis": "tent",
            "limits": {"warn_condition": 1e5},
            "curves": {SOFR: {"deposit": ["3M"], "future": [["SR3", "U26"]], "swap": ["2Y", "5Y", "10Y"]}},
        }
        template = RiskTemplate.from_spec(spec, AS_OF)
        self.assertEqual(
            template.labels,
            (f"{SOFR} deposit 3M", f"{SOFR} future SR3 U26", f"{SOFR} swap 2Y", f"{SOFR} swap 5Y", f"{SOFR} swap 10Y"),
        )
        self.assertEqual(template.to_spec(), spec)
        self.assertEqual(template.limits.warn_condition, 1e5)
        bound = template.bind(self.market)
        self.assertTrue(np.isnan(bound.diagnostics.residuals).all())  # no quotes carried
        self.assertEqual(len(template.digest), 12)
        self.assertNotEqual(template.digest, self.identity.digest)
        with self.assertRaises(ValueError):
            self.identity.to_spec()
