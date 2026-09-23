"""SOFR future and European swaption product wrappers over the Numba engine."""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import SofrFuture, Swaption, SwaptionModel, imm_date, next_quarterly_imm
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, YieldCurve, ZeroCurve
from finance.markets.vols import FlatVolSurface, VolNamespace, VolUnits
from finance.pricing.calibration import (
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    Quote,
    deposit_helper,
    futures_helper,
)
from finance.pricing.pricers import FuturesPricer, SwaptionPricer
from finance.pricing.risk.sensitivities import bumped_curve
from finance.pricing.types import Backend

CURVE = "USD.SOFR"


def _market(as_of, *, vol=0.01):
    origin = as_of.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 3 * 365, 7 * 365, 12 * 365)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.035, 0.04, 0.045, 0.047])
    curves = CurveNamespace()
    zero_curve = ZeroCurve(dates, np.exp(-z * t), CurveInterpolator.LogLinearDF)
    curves.bind(YieldCurve.from_registry(zero_curve, currency="USD", index_name="SOFR"))
    vols = VolNamespace()
    vols.bind(CURVE, FlatVolSurface(vol, VolUnits.Normal))
    return MarketContext(as_of, curves, vols=vols)


class TestSofrFutures(UnitTest):
    COVERAGE = [
        "finance.instruments.resolution.futures",
        "finance.pricing.pricers.futures",
    ]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)

    def test_imm_contract_dates(self):
        self.assertEqual(imm_date(2026, 9), Date(2026, 9, 16))
        self.assertEqual(next_quarterly_imm(Date(2026, 9, 16)), Date(2026, 12, 16))

    def test_sr3_model_price_and_pnl_identity(self):
        future = SofrFuture.from_contract_month(
            contract="SR3", month="U26", as_of=self.as_of, entry_price=95.75, contracts=3
        )
        result = FuturesPricer().price([future], self.market)
        np.testing.assert_allclose(result.model_price, 100.0 * (1.0 - result.model_rate))
        expected_pnl = (result.model_price[0] - 95.75) / 0.01 * 25.0 * 3
        self.assertAlmostEqual(result.pnl[0], expected_pnl)

    def test_rate_adjoint_matches_zero_rate_bump(self):
        future = SofrFuture.from_contract_month(contract="SR3", month="Z26", as_of=self.as_of)
        program = FuturesPricer().compile([future])
        analytic = program.risk(self.market, CURVE).rate_gradient[0]
        base = self.market.zero_curve(CURVE)
        h = 1e-6
        for pillar in range(1, base.node_dates.size):
            yc = self.market.yield_curve(CURVE)
            up = self.market.with_curve(yc.with_zero_curve(bumped_curve(base, h, pillar=pillar)))
            down = self.market.with_curve(yc.with_zero_curve(bumped_curve(base, -h, pillar=pillar)))
            finite_difference = (program.reprice(up).model_rate[0] - program.reprice(down).model_rate[0]) / (2 * h)
            self.assertAlmostEqual(analytic[pillar - 1], finite_difference, places=7)

    def test_future_participates_in_calibration_and_numba_jacobian(self):
        dep = deposit_helper(rate=0.0, tenor="3M", as_of=self.as_of)
        fut = futures_helper(price=96.0, contract="SR3", month="U26", as_of=self.as_of)
        true_market = self.market
        dep.quote = Quote(dep.implied(true_market), dep.quote.kind)
        fut.quote = Quote(fut.implied(true_market), fut.quote.kind)
        base = MarketContext(self.as_of, CurveNamespace())
        result = CurveCalibrator(
            [dep, fut], GlobalSolver(), CurveDefinition("USD", "SOFR", CurveInterpolator.LogLinearDF)
        ).calibrate(base, jacobian=True)
        self.assertLess(np.max(np.abs(result.residuals)), 1e-10)
        self.assertEqual(result.jacobian.shape, (2, 2))
        self.assertTrue(np.isfinite(result.jacobian).all())


class TestSwaptions(UnitTest):
    COVERAGE = [
        "finance.instruments.resolution.swaption",
        "finance.pricing.pricers.swaption",
    ]

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)

    def _option(self, payer=True, model=SwaptionModel.Bachelier, vol_name=None):
        return Swaption.european(
            notional=1e6,
            strike=0.04,
            expiry="1Y",
            swap_tenor="5Y",
            as_of=self.as_of,
            payer=payer,
            model=model,
            vol_name=vol_name,
        )

    def test_payer_receiver_parity(self):
        payer = SwaptionPricer().price([self._option(True)], self.market)
        receiver = SwaptionPricer().price([self._option(False)], self.market)
        expected = payer.annuity[0] * (payer.forward[0] - 0.04)
        self.assertAlmostEqual(payer.pv[0] - receiver.pv[0], expected, places=7)

    def test_curve_adjoint_matches_bump(self):
        program = SwaptionPricer().compile([self._option(True)])
        analytic = program.risk(self.market, CURVE).curve_gradient[0]
        base = self.market.zero_curve(CURVE)
        h = 1e-6
        for pillar in range(1, base.node_dates.size):
            yc = self.market.yield_curve(CURVE)
            up = self.market.with_curve(yc.with_zero_curve(bumped_curve(base, h, pillar=pillar)))
            down = self.market.with_curve(yc.with_zero_curve(bumped_curve(base, -h, pillar=pillar)))
            finite_difference = (program.reprice(up).pv[0] - program.reprice(down).pv[0]) / (2 * h)
            np.testing.assert_allclose(analytic[pillar - 1], finite_difference, rtol=2e-7, atol=1e-3)

    def test_monetary_vega_is_positive(self):
        result = SwaptionPricer().price([self._option(True)], self.market)
        self.assertGreater(result.greeks.vega[0], 0.0)

    def test_numpy_and_numba_backends_agree(self):
        options = [self._option(True), self._option(False)]
        via_numpy = SwaptionPricer().price(options, self.market, backend=Backend.Numpy)
        via_numba = SwaptionPricer().price(options, self.market, backend=Backend.Numba)
        np.testing.assert_allclose(via_numba.pv, via_numpy.pv, rtol=1e-12)
        for field in ("value", "delta", "gamma", "vega", "vanna", "volga"):
            np.testing.assert_allclose(
                getattr(via_numba.greeks, field), getattr(via_numpy.greeks, field), rtol=1e-9
            )

    def test_black_model_payer_receiver_parity(self):
        self.market.vols.bind("USD.SOFR.LN", FlatVolSurface(0.25, VolUnits.Lognormal))
        payer = SwaptionPricer().price([self._option(True, SwaptionModel.Black, "USD.SOFR.LN")], self.market)
        receiver = SwaptionPricer().price([self._option(False, SwaptionModel.Black, "USD.SOFR.LN")], self.market)
        expected = payer.annuity[0] * (payer.forward[0] - 0.04)
        self.assertAlmostEqual(payer.pv[0] - receiver.pv[0], expected, places=7)

    def test_vol_units_mismatch_raises(self):
        # default vol name resolves the Normal surface; Black requires Lognormal
        with self.assertRaisesRegex(ValueError, "Lognormal"):
            SwaptionPricer().price([self._option(True, SwaptionModel.Black)], self.market)
