"""pv01, annuity and par_coupon: fixed coupon flows only, principal excluded."""
import dataclasses

import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.dates import Date
from quant_toolkit.finance.instruments.enums import PrincipalExchange
from quant_toolkit.finance.instruments.resolution import Deposit, SofrFuture, Swap, Swaption
from quant_toolkit.finance.markets.context import MarketContext
from quant_toolkit.finance.markets.curves import YieldCurve
from quant_toolkit.finance.markets.vols import FlatVolSurface, ModelVolCube, VolNamespace, VolUnits
from quant_toolkit.finance.pricing.pricers import LegPricer

CURVE = "USD.SOFR.1D"
BP = 1e-4
REQUESTS = ["pv", "pv01", "annuity", "par_coupon", "cashflows"]


def _market(as_of: Date) -> MarketContext:
    origin = as_of.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 3 * 365, 7 * 365, 12 * 365)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.035, 0.04, 0.045, 0.047])
    sofr = YieldCurve.build(dates, np.exp(-z * t), key="USD.SOFR.1D")
    vols = VolNamespace()
    vols.bind(CURVE, ModelVolCube(FlatVolSurface(0.01), VolUnits.Normal))
    return MarketContext(as_of, {sofr.name: sofr, "USD.STDCSA": sofr}, vols=vols)


class TestCouponMeasures(UnitTest):

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)

    def _swap(self, notional=10e6, rate=0.04):
        return Swap.fixed_float_swap(
            notional=notional, rate_index="USD.SOFR.1D", fixed_rate=rate, tenor="5Y", as_of=self.as_of
        )

    def _with_fixed_leg(self, swap, **changes):
        if swap.receive_leg.is_floating:
            return dataclasses.replace(swap, pay_leg=swap.pay_leg.replace(**changes))
        return dataclasses.replace(swap, receive_leg=swap.receive_leg.replace(**changes))

    def test_pv01_is_the_pv_change_for_one_bp_of_coupon(self):
        for notional in (10e6, -10e6):
            with self.subTest(notional=notional):
                base = self._swap(notional)(self.market, REQUESTS)
                bumped = self._swap(notional, rate=0.0401)(self.market).pv
                np.testing.assert_allclose(base.get("pv01")[0], bumped - base.pv, rtol=1e-9)
                self.assertEqual(np.sign(base.get("pv01")[0]), np.sign(notional))
                np.testing.assert_allclose(base.get("annuity")[0], abs(base.get("pv01")[0]) / 1e-4, rtol=1e-12)

    def test_par_coupon_zeroes_the_swap(self):
        par = self._swap()(self.market, REQUESTS).get("par_coupon")[0]
        self.assertAlmostEqual(self._swap(rate=par)(self.market).pv, 0.0, delta=1e-6)

    def test_principal_exchange_does_not_enter_the_annuity(self):
        plain = self._swap()
        exchanged = self._with_fixed_leg(plain, principal_exchange=PrincipalExchange.Final)
        a, b = plain(self.market, REQUESTS), exchanged(self.market, REQUESTS)
        self.assertNotAlmostEqual(a.pv, b.pv, delta=1.0)                      # the principal flow is priced
        np.testing.assert_allclose(b.get("pv01"), a.get("pv01"), rtol=1e-12)  # but is not a coupon
        self.assertEqual(int(b.cashflows.is_principal.sum()), 1)
        self.assertEqual(int(a.cashflows.is_principal.sum()), 0)
        par = b.get("par_coupon")[0]
        self.assertAlmostEqual(self._with_fixed_leg(exchanged, coupon_rate=par)(self.market).pv, 0.0, delta=1e-6)

    def test_deposit_reports_its_coupon_not_its_principal(self):
        deposit = Deposit.spot_deposit(notional=1e6, rate=0.05, tenor="3M", as_of=self.as_of)
        result = deposit(self.market, REQUESTS)
        self.assertLess(result.get("annuity")[0], 0.3 * 1e6)   # ~ notional * 0.25 * df, not notional * df

    def test_floating_only_instrument_has_zero_pv01_and_no_par_coupon(self):
        swap = self._swap()
        floating = swap.receive_leg if swap.receive_leg.is_floating else swap.pay_leg

        class FloatOnly:
            currency, index_name, funding_id = swap.currency, swap.index_name, swap.funding_id
            legs = ((floating, 1.0),)

        result = LegPricer().compile([FloatOnly()]).price(self.market, REQUESTS)
        self.assertEqual(result.get("pv01")[0], 0.0)
        self.assertEqual(result.get("annuity")[0], 0.0)
        self.assertTrue(np.isnan(result.get("par_coupon")[0]))

    def test_coupon_measures_are_on_request_only(self):
        self.assertEqual(self._swap()(self.market).extras, {})
        self.assertLessEqual({"pv01", "annuity", "par_coupon"}, set(self._swap().requests()))

    def test_replace_keeps_a_flat_leg_flat(self):
        # a non-coupon replace must not freeze the derived schedule: a later coupon change still applies
        leg = self._swap().receive_leg.replace(principal_exchange=PrincipalExchange.Final)
        self.assertEqual(leg.coupon_type, self._swap().receive_leg.coupon_type)
        self.assertEqual(leg.replace(coupon_rate=0.05).coupon_schedule.events[0].coupon_rate, 0.05)


class TestDv01(UnitTest):

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.swap = Swap.fixed_float_swap(
            notional=10e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=self.as_of
        )

    def _bumped(self, instrument, measure):
        up = instrument(self.market.bump(CURVE, BP), [measure]).get(measure)
        down = instrument(self.market.bump(CURVE, -BP), [measure]).get(measure)
        return (up - down) / 2.0

    def test_swap_dv01_matches_a_parallel_bump_and_the_ladder(self):
        result = self.swap(self.market, ["dv01", "zero_ladder"])
        np.testing.assert_allclose(result.get("dv01"), self._bumped(self.swap, "pv"), rtol=1e-6)
        np.testing.assert_allclose(
            result.get("dv01"), result.get("zero_ladder").block(CURVE).sum(axis=1) * BP, rtol=1e-12
        )
        self.assertLess(result.get("dv01")[0], 0.0)   # receiving fixed loses when rates rise

    def test_curve_setting_picks_the_block(self):
        default = self.swap(self.market, ["dv01"]).get("dv01")
        # discounting is the SOFR curve under another key, so it reports under the same block
        np.testing.assert_allclose(self.swap(self.market, ["dv01"], curve="USD.STDCSA").get("dv01"), default)
        with self.assertRaises(KeyError):
            self.swap(self.market, ["dv01"], curve="USD.NOPE.1D")

    def test_unknown_setting_raises(self):
        with self.assertRaisesRegex(TypeError, "curve"):
            self.swap(self.market, ["dv01"], crv=CURVE)

    def test_fixed_only_instrument_reports_against_its_discount_curve(self):
        deposit = Deposit.spot_deposit(notional=1e6, rate=0.05, tenor="3M", as_of=self.as_of)
        np.testing.assert_allclose(
            deposit(self.market, ["dv01"]).get("dv01"), self._bumped(deposit, "pv"), rtol=1e-6
        )

    def test_swaption_and_future_answer_dv01(self):
        option = Swaption.european(notional=1e6, strike=0.04, expiry="1Y", swap_tenor="5Y", as_of=self.as_of)
        np.testing.assert_allclose(option(self.market, ["dv01"]).get("dv01"), self._bumped(option, "pv"), rtol=1e-5)
        future = SofrFuture.from_contract_month(contract="SR3", month="U26", as_of=self.as_of, entry_price=96.0)
        np.testing.assert_allclose(future(self.market, ["dv01"]).get("dv01"), self._bumped(future, "pnl"), rtol=1e-5)

    def test_template_re_expresses_risk_and_leaves_the_rest_alone(self):
        from quant_toolkit.finance.pricing.calibration import Quote, swap_helper
        from quant_toolkit.finance.pricing.risk import RiskEngine
        from quant_toolkit.finance.pricing.risk.template import QuoteRisk, RiskTemplate

        helpers = [swap_helper(rate=0.0, tenor=t, as_of=self.as_of) for t in ("1Y", "3Y", "7Y", "12Y")]
        for h in helpers:
            h.quote = Quote(float(h.implied(self.market)), h.quote.kind)
        template = RiskTemplate.identity(helpers, name="par 1Y-12Y")

        plain = self.swap(self.market, ["pv", "pv01", "dv01", "zero_ladder"])
        viewed = self.swap(self.market, ["pv", "pv01", "dv01", "zero_ladder"], template=template)
        self.assertEqual(viewed.pv, plain.pv)                                  # a template changes no PV
        np.testing.assert_array_equal(viewed.get("pv01"), plain.get("pv01"))

        ladder = viewed.get("zero_ladder")
        self.assertIsInstance(ladder, QuoteRisk)
        expected = RiskEngine(self.swap._compiled_program(), self.market, template=template).quote_risk()
        np.testing.assert_allclose(ladder.values, expected.values, rtol=1e-12)
        np.testing.assert_allclose(ladder.unexplained, expected.unexplained, atol=1e-9)

        self.assertEqual(viewed.get("dv01").shape, (1,))                       # a number per position
        np.testing.assert_allclose(viewed.get("dv01"), ladder.dv01.sum(axis=1), rtol=1e-12)
        self.assertLess(viewed.get("dv01")[0], 0.0)
        # a bound template is accepted too, and gives the same answer
        bound = template.bind(self.market)
        np.testing.assert_allclose(
            self.swap(self.market, ["dv01"], template=bound).get("dv01"), viewed.get("dv01"), rtol=1e-12
        )


class TestParallelGamma(UnitTest):

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of)
        self.swap = Swap.fixed_float_swap(
            notional=10e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=self.as_of
        )
        self.option = Swaption.european(notional=10e6, strike=0.046, expiry="1Y", swap_tenor="5Y", as_of=self.as_of)

    def test_gamma_1bp_is_the_change_in_dv01_per_bp(self):
        for instrument in (self.swap, self.option):
            with self.subTest(type(instrument).__name__):
                result = instrument(self.market, ["dv01", "gamma_1bp", "gamma_ladder"])
                up = instrument(self.market.bump(CURVE, BP), ["dv01"]).get("dv01")
                down = instrument(self.market.bump(CURVE, -BP), ["dv01"]).get("dv01")
                np.testing.assert_allclose(result.get("gamma_1bp"), (up - down) / 2.0, rtol=1e-3)
                np.testing.assert_allclose(
                    result.get("gamma_ladder").block(CURVE).sum(axis=1), result.get("gamma_1bp"), rtol=1e-12
                )

    def test_second_order_explain_of_a_parallel_move(self):
        m = 25.0   # bp
        result = self.option(self.market, ["pv", "dv01", "gamma_1bp"])
        actual = self.option(self.market.bump(CURVE, m * BP), ["pv"]).get("pv") - result.get("pv")
        first = result.get("dv01") * m
        second = first + 0.5 * result.get("gamma_1bp") * m * m
        self.assertLess(abs(second[0] - actual[0]), 0.2 * abs(first[0] - actual[0]))   # gamma closes most of the gap

    def test_reconciles_with_the_engine_cross_gamma(self):
        from quant_toolkit.finance.pricing.risk import RiskEngine

        matrix = RiskEngine(self.swap._compiled_program(), self.market).gamma(CURVE)   # 0.5 * H * bp**2, one swap
        np.testing.assert_allclose(self.swap(self.market, ["gamma_1bp"]).get("gamma_1bp")[0], 2.0 * matrix.sum(), rtol=1e-5)

    def test_swaption_curve_gamma_is_not_its_forward_gamma(self):
        result = self.option(self.market, ["gamma", "gamma_1bp"])
        self.assertGreater(result.get("gamma_1bp")[0], 0.0)         # long option: long curve gamma
        self.assertNotAlmostEqual(result.get("gamma")[0], result.get("gamma_1bp")[0])

    def test_bad_step_is_refused(self):
        with self.assertRaisesRegex(ValueError, "gamma_step"):
            self.swap(self.market, ["gamma_1bp"], gamma_step=0.0)

    def _template(self):
        from quant_toolkit.finance.pricing.calibration import Quote, swap_helper
        from quant_toolkit.finance.pricing.risk.template import RiskTemplate

        helpers = [swap_helper(rate=0.0, tenor=t, as_of=self.as_of) for t in ("1Y", "3Y", "7Y", "12Y")]
        for h in helpers:
            h.quote = Quote(float(h.implied(self.market)), h.quote.kind)
        return RiskTemplate.identity(helpers, name="par 1Y-12Y")

    def _quotes_moved(self, bound, size):
        market = self.market
        for k in range(len(bound.template.instruments)):
            market = bound.apply(market, k, size)
        return market

    def test_quote_gamma_frozen_is_the_hessian_through_the_template(self):
        from quant_toolkit.finance.pricing.risk import RiskEngine
        from quant_toolkit.finance.pricing.risk.template import QuoteRisk

        bound = self._template().bind(self.market)
        result = self.swap(self.market, ["dv01", "gamma_1bp", "gamma_ladder"], template=bound)
        matrix = RiskEngine(self.swap._compiled_program(), self.market).gamma(CURVE)   # 0.5 * H * bp**2
        move = bound.transfer.sum(axis=1)                                             # pillar move per unit of all quotes
        np.testing.assert_allclose(result.get("gamma_1bp")[0], 2.0 * move @ matrix @ move, rtol=1e-4)
        ladder = result.get("gamma_ladder")
        self.assertIsInstance(ladder, QuoteRisk)
        np.testing.assert_allclose(ladder.values.sum(axis=1), result.get("gamma_1bp"), rtol=1e-12)

    def test_quote_gamma_rebound_is_the_change_in_quote_dv01(self):
        template = self._template()
        bound = template.bind(self.market)
        for instrument in (self.swap, self.option):
            with self.subTest(type(instrument).__name__):
                rebound = instrument(self.market, ["gamma_1bp"], template=bound, rebind_template=True).get("gamma_1bp")
                up = instrument(self._quotes_moved(bound, BP), ["dv01"], template=template).get("dv01")
                down = instrument(self._quotes_moved(bound, -BP), ["dv01"], template=template).get("dv01")
                np.testing.assert_allclose(rebound, (up - down) / 2.0, rtol=1e-3)
                frozen = instrument(self.market, ["gamma_1bp"], template=bound).get("gamma_1bp")
                self.assertTrue(np.isfinite(frozen).all())
                self.assertNotEqual(float(frozen[0]), float(rebound[0]))   # the map's curvature is a real term
