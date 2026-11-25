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
