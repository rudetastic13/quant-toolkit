"""RiskEngine dispatch: adjoint vs bump agreement, method resolution, and guardrails."""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Swap, SofrFuture, Swaption, SwaptionModel
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, YieldCurve
from finance.markets.vols import FlatVolSurface, VolNamespace, VolUnits
from finance.pricing.pricers import FuturesPricer, SwapPricer, SwaptionPricer
from finance.pricing.risk import RiskEngine, Sensitivities

CURVE = "USD.SOFR"


def _market(as_of, *, interpolation=CurveInterpolator.LogLinearDF):
    origin = as_of.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 3 * 365, 7 * 365, 12 * 365)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.035, 0.04, 0.045, 0.047])
    curves = CurveNamespace()
    space, interpolator = interpolation.resolve()
    curves.bind(
        YieldCurve.build(dates, np.exp(-z * t), currency="USD", index_name="SOFR", space=space, interpolator=interpolator)
    )
    vols = VolNamespace()
    vols.bind(CURVE, FlatVolSurface(0.01, VolUnits.Normal))
    return MarketContext(as_of, curves, vols=vols)


class TestRiskEngine(UnitTest):
    COVERAGE = ["finance.pricing.risk.engine"]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.book = [
            Swap.fixed_float_swap(notional=100e6, rate_index="SOFR", fixed_rate=0.041, tenor="5Y", as_of=self.as_of),
            Swap.fixed_float_swap(notional=-25e6, rate_index="SOFR", fixed_rate=0.040, tenor="10Y", as_of=self.as_of),
        ]

    def test_swap_book_adjoint_matches_bump(self):
        program = SwapPricer().compile(self.book)
        adjoint = RiskEngine(program, self.market)
        bump = RiskEngine(program, self.market, method="bump")
        self.assertEqual(adjoint.method, "adjoint")
        np.testing.assert_allclose(adjoint.ladder(CURVE).ladder, bump.ladder(CURVE).ladder, rtol=5e-4, atol=1e-6)
        np.testing.assert_allclose(adjoint.dv01(CURVE), bump.dv01(CURVE), rtol=5e-4)
        np.testing.assert_allclose(bump.dv01(CURVE), Sensitivities(program, self.market).dv01(CURVE), rtol=1e-12)

    def test_non_loglinear_curve_resolves_to_bump(self):
        market = _market(self.as_of, interpolation=CurveInterpolator.RateLinear)
        engine = RiskEngine(SwapPricer().compile(self.book), market)
        self.assertEqual(engine.method, "bump")
        self.assertEqual(engine.ladder(CURVE).method, "bump")
        self.assertTrue(np.isfinite(engine.dv01(CURVE)).all())

    def test_swaption_adjoint_matches_bump(self):
        option = Swaption.european(
            notional=1e6, strike=0.04, expiry="1Y", swap_tenor="5Y", as_of=self.as_of,
            payer=True, model=SwaptionModel.Bachelier,
        )
        program = SwaptionPricer().compile([option])
        adjoint = RiskEngine(program, self.market)
        bump = RiskEngine(program, self.market, method="bump")
        self.assertEqual(adjoint.method, "adjoint")
        np.testing.assert_allclose(adjoint.ladder(CURVE).ladder, bump.ladder(CURVE).ladder, rtol=5e-4, atol=1e-6)

    def test_gamma_requires_adjoint_and_runs_on_linear(self):
        program = SwapPricer().compile(self.book)
        gamma = RiskEngine(program, self.market).gamma(CURVE)
        self.assertEqual(gamma.shape, (4, 4))
        self.assertTrue(np.isfinite(gamma).all())
        with self.assertRaisesRegex(ValueError, "adjoint"):
            RiskEngine(program, self.market, method="bump").gamma(CURVE)

    def test_futures_program_is_rejected(self):
        future = SofrFuture.from_contract_month(contract="SR3", month="U26", as_of=self.as_of)
        with self.assertRaisesRegex(TypeError, "FuturesProgram"):
            RiskEngine(FuturesPricer().compile([future]), self.market)
