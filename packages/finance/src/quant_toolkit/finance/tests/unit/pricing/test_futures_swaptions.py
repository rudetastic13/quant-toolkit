"""SOFR future and European swaption product wrappers over the Numba engine."""
import dataclasses

import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.dates import Date
from quant_toolkit.finance.instruments.resolution import SofrFuture, Swaption, imm_date, next_quarterly_imm
from quant_toolkit.finance.markets.context import MarketContext
from quant_toolkit.finance.markets.curves import CurveInterpolator, YieldCurve
from quant_toolkit.finance.markets.vols import FlatVolSurface, GridVolSurface, ModelVolCube, VolNamespace, VolUnits
from quant_toolkit.finance.pricing.calibration import (
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    Quote,
    deposit_helper,
    futures_helper,
)
from quant_toolkit.finance.pricing.engines.numpy.option import black_greeks
from quant_toolkit.finance.pricing.pricers import FuturesPricer, SwaptionPricer
from quant_toolkit.finance.pricing.types import Backend

CURVE = "USD.SOFR.1D"


def _market(as_of, *, vol=0.01):
    origin = as_of.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 3 * 365, 7 * 365, 12 * 365)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.035, 0.04, 0.045, 0.047])
    sofr = YieldCurve.build(dates, np.exp(-z * t), key="USD.SOFR.1D")
    vols = VolNamespace()
    vols.bind(CURVE, ModelVolCube(FlatVolSurface(vol), VolUnits.Normal))
    return MarketContext(as_of, {sofr.name: sofr, "USD.STDCSA": sofr}, vols=vols)


class TestSofrFutures(UnitTest):

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
        h = 1e-6
        for pillar in range(self.market[CURVE].zero_curve.n_params):
            shift = np.zeros(self.market[CURVE].zero_curve.n_params)
            shift[pillar] = h
            up, down = self.market.bump(CURVE, shift), self.market.bump(CURVE, -shift)
            finite_difference = (program.price(up).model_rate[0] - program.price(down).model_rate[0]) / (2 * h)
            self.assertAlmostEqual(analytic[pillar], finite_difference, places=7)

    def test_future_participates_in_calibration_and_numba_jacobian(self):
        dep = deposit_helper(rate=0.0, tenor="3M", as_of=self.as_of)
        fut = futures_helper(price=96.0, contract="SR3", month="U26", as_of=self.as_of)
        true_market = self.market
        dep.quote = Quote(dep.implied(true_market), dep.quote.kind)
        fut.quote = Quote(fut.implied(true_market), fut.quote.kind)
        base = MarketContext(self.as_of)
        result = CurveCalibrator(
            [dep, fut], GlobalSolver(), CurveDefinition("USD.SOFR.1D", CurveInterpolator.LogLinearDF, discount_keys=("USD.STDCSA",))
        ).calibrate(base, jacobian=True)
        self.assertLess(np.max(np.abs(result.residuals)), 1e-10)
        self.assertEqual(result.jacobian.shape, (2, 2))
        self.assertTrue(np.isfinite(result.jacobian).all())


class TestSwaptions(UnitTest):

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)

    def _option(self, payer=True, vol_name=None):
        return Swaption.european(
            notional=1e6,
            strike=0.04,
            expiry="1Y",
            swap_tenor="5Y",
            as_of=self.as_of,
            payer=payer,
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
        h = 1e-6
        for pillar in range(self.market[CURVE].zero_curve.n_params):
            shift = np.zeros(self.market[CURVE].zero_curve.n_params)
            shift[pillar] = h
            up, down = self.market.bump(CURVE, shift), self.market.bump(CURVE, -shift)
            finite_difference = (program.price(up).pv[0] - program.price(down).pv[0]) / (2 * h)
            np.testing.assert_allclose(analytic[pillar], finite_difference, rtol=2e-7, atol=1e-3)

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
        self.market.vols.bind("USD.SOFR.LN", ModelVolCube(FlatVolSurface(0.25), VolUnits.Lognormal))
        payer = SwaptionPricer().price([self._option(True, "USD.SOFR.LN")], self.market)
        receiver = SwaptionPricer().price([self._option(False, "USD.SOFR.LN")], self.market)
        expected = payer.annuity[0] * (payer.forward[0] - 0.04)
        self.assertAlmostEqual(payer.pv[0] - receiver.pv[0], expected, places=7)

    def test_vol_gradient_matches_a_node_bump(self):
        grid = GridVolSurface(
            np.array([0.5, 2.0, 5.0]), np.array([2.0, 10.0]),
            np.array([[0.010, 0.012], [0.011, 0.013], [0.009, 0.0095]]),
        )
        cube = ModelVolCube(grid, VolUnits.Normal)
        self.market.vols.rebind(CURVE, cube)
        options = [
            self._option(True),
            Swaption.european(notional=2e6, strike=0.045, expiry="3Y", swap_tenor="7Y", as_of=self.as_of, payer=False),
        ]
        program = SwaptionPricer().compile(options)
        analytic = program.risk(self.market).vol_gradient[CURVE]
        self.assertEqual(analytic.shape, (2, 6))
        h = 1e-7
        for node in range(cube.params.size):
            shift = np.zeros(cube.params.size)
            shift[node] = h
            self.market.vols.rebind(CURVE, cube.with_params(cube.params + shift))
            up = program.price(self.market).pv
            self.market.vols.rebind(CURVE, cube.with_params(cube.params - shift))
            down = program.price(self.market).pv
            np.testing.assert_allclose(analytic[:, node], (up - down) / (2 * h), rtol=1e-6, atol=1e-4)
        # every option's node vegas add up to its own vega: the weights sum to one
        self.market.vols.rebind(CURVE, cube)
        np.testing.assert_allclose(analytic.sum(axis=1), program.price(self.market).greeks.vega, rtol=1e-12)

    def test_vol_gradient_is_reported_per_cube(self):
        self.market.vols.bind("USD.SOFR.LN", ModelVolCube(FlatVolSurface(0.25), VolUnits.Lognormal))
        options = [self._option(True), self._option(True, "USD.SOFR.LN")]
        risk = SwaptionPricer().compile(options).risk(self.market)
        self.assertEqual(set(risk.vol_gradient), {CURVE, "USD.SOFR.LN"})
        self.assertGreater(risk.vol_gradient[CURVE][0, 0], 0.0)
        self.assertEqual(risk.vol_gradient[CURVE][1, 0], 0.0)
        self.assertEqual(risk.vol_gradient["USD.SOFR.LN"][0, 0], 0.0)
        self.assertGreater(risk.vol_gradient["USD.SOFR.LN"][1, 0], 0.0)

    def test_cube_units_pick_the_option_model(self):
        # the same contract prices Bachelier off a Normal cube and Black off a Lognormal one
        option = self._option(True)
        normal = SwaptionPricer().price([option], self.market)
        self.market.vols.rebind(CURVE, ModelVolCube(FlatVolSurface(0.25), VolUnits.Lognormal))
        lognormal = SwaptionPricer().price([option], self.market)
        unit = black_greeks(lognormal.forward, np.array([0.04]), np.array([1.0]), np.array([0.25]), is_call=True)
        np.testing.assert_allclose(lognormal.pv, lognormal.annuity * unit.value, rtol=1e-12)
        self.assertNotAlmostEqual(float(normal.pv[0]), float(lognormal.pv[0]), places=2)

    def test_swaption_and_future_are_priceable(self):
        option = self._option(True)
        np.testing.assert_array_equal(
            option(self.market, ["pv", "vega", "cashflows"]).get("pv"), SwaptionPricer().price([option], self.market).pv
        )
        self.assertIsNone(option(self.market, ["pv", "cashflows"]).get("cashflows"))
        future = SofrFuture.from_contract_month(contract="SR3", month="U26", as_of=self.as_of, convexity=0.0002)
        result = future(self.market, ["price", "convexity", "forward_rate", "vega"])
        np.testing.assert_array_equal(result.get("price"), FuturesPricer().price([future], self.market).model_price)
        np.testing.assert_allclose(result.get("rate") - result.get("forward_rate"), result.get("convexity"), atol=1e-15)
        self.assertIsNone(result.get("vega"))

    def test_direction_comes_from_the_underlying(self):
        self.assertTrue(self._option(True).payer)
        self.assertFalse(self._option(False).payer)
        with self.assertRaisesRegex(ValueError, "one fixed leg"):
            underlying = self._option(True).underlying
            Swaption(
                underlying=dataclasses.replace(underlying, pay_leg=underlying.receive_leg),
                expiry=self._option(True).expiry,
            )

    def test_sold_option_negates_value_and_risk(self):
        common = dict(notional=1e6, strike=0.04, expiry="1Y", swap_tenor="5Y", as_of=self.as_of)
        program = SwaptionPricer().compile(
            [Swaption.european(**common), Swaption.european(**common, position=-1.0)]
        )
        result = program.price(self.market)
        self.assertGreater(result.pv[0], 0.0)
        np.testing.assert_allclose(result.pv[1], -result.pv[0], rtol=1e-14)
        np.testing.assert_allclose(result.greeks.vega[1], -result.greeks.vega[0], rtol=1e-14)
        risk = program.risk(self.market)
        np.testing.assert_allclose(risk.curve_gradient[1], -risk.curve_gradient[0], rtol=1e-14)
        np.testing.assert_allclose(risk.vol_gradient[CURVE][1], -risk.vol_gradient[CURVE][0], rtol=1e-14)

    def test_every_product_lists_its_requests(self):
        future = SofrFuture.from_contract_month(contract="SR3", month="U26", as_of=self.as_of)
        swap = self._option(True).underlying
        self.assertLessEqual({"pv", "vega", "zero_ladder"}, set(self._option(True).requests()))
        self.assertLessEqual({"price", "convexity", "zero_ladder"}, set(future.requests()))
        self.assertLessEqual({"pv", "cashflows", "index_ladder"}, set(swap.requests()))
        self.assertNotIn("vega", swap.requests())

    def test_as_dict_reports_none_for_another_products_measure(self):
        out = self._option(True)(self.market, ["pv", "vega", "pnl"]).as_dict(["pv", "vega", "pnl"])
        self.assertEqual(list(out), ["pv", "vega", "pnl"])
        self.assertIsNone(out["pnl"])
        self.assertGreater(out["vega"][0], 0.0)
