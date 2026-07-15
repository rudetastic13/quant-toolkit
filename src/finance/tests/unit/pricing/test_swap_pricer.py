"""End-to-end: trader Swap(...) -> SwapPricer -> PV, the anti-fixedfloatswap path.

The instrument carries no curve; the market is passed to price(). Compile once, reprice
against any market. A par swap prices to ~0; negative notional flips the sign.
"""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date, Frequency, DayCountMethod
from finance.instruments.enums import CouponType
from finance.markets.curves import ZeroCurve
from finance.markets.context import MarketContext
from finance.markets.curves import CurveNamespace
from finance.instruments.resolution import Swap
from finance.pricing.pricers import SwapPricer

AS_OF = Date(2026, 6, 1)


def _market():
    origin = AS_OF.to_numpy()
    dates = np.array([origin, origin + np.timedelta64(2 * 365, "D"),
                      origin + np.timedelta64(5 * 365, "D"),
                      origin + np.timedelta64(12 * 365, "D")], dtype="datetime64[D]")
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    ns = CurveNamespace()
    ns.bind("USD.SOFR", ZeroCurve(dates, np.exp(-0.04 * t)))
    return MarketContext(as_of_date=AS_OF, curves=ns)


class TestSwapResolution(UnitTest):
    COVERAGE = ["finance.instruments.resolution.builders", "finance.instruments.resolution.resolver"]

    def test_minimal_input_resolves_conventions(self):
        swap = Swap.fixed_float_swap(notional=100, rate_index="SOFR", fixed_rate=0.045, tenor="10Y", as_of=AS_OF)
        # effective = as_of + 2 business days (SOFR spot lag), maturity = effective + 10Y
        expected_eff = np.busday_offset(AS_OF.to_numpy(), 2, roll="following")
        self.assertEqual(swap.receive_leg.effective.to_numpy(), expected_eff)
        eff = swap.receive_leg.effective
        self.assertEqual(swap.receive_leg.maturity, Date(eff.year + 10, eff.month, eff.day))
        # conventions filled from the registry, not by the trader
        self.assertEqual(swap.receive_leg.payment_frequency, Frequency.Annually)
        self.assertEqual(swap.receive_leg.day_count_method, DayCountMethod.Actual360)

    def test_receive_fixed_is_default(self):
        swap = Swap.fixed_float_swap(notional=100, rate_index="SOFR", fixed_rate=0.045, tenor="5Y", as_of=AS_OF)
        self.assertEqual(swap.receive_leg.coupon_type, CouponType.Fixed)
        self.assertEqual(swap.pay_leg.coupon_type, CouponType.GeometricAveraged)

    def test_negative_notional_pays_fixed(self):
        swap = Swap.fixed_float_swap(notional=-100, rate_index="SOFR", fixed_rate=0.045, tenor="5Y", as_of=AS_OF)
        # paying fixed: the fixed leg is now the pay leg
        self.assertEqual(swap.pay_leg.coupon_type, CouponType.Fixed)
        self.assertEqual(swap.receive_leg.coupon_type, CouponType.GeometricAveraged)
        self.assertEqual(swap.pay_leg.notional, 100.0)  # magnitude positive; sign is structural


class TestSwapPricing(UnitTest):
    COVERAGE = ["finance.pricing.pricers.swap", "finance.pricing.pricers.base"]

    def test_par_swap_prices_to_zero(self):
        mkt = _market()
        pricer = SwapPricer()
        # price at an off-par rate, derive par from the two leg PVs, reprice at par
        r0 = 0.04
        res0 = pricer.price([Swap.fixed_float_swap(notional=100, rate_index="SOFR", fixed_rate=r0, tenor="10Y", as_of=AS_OF)], mkt)
        fixed_pv, float_pv = res0.leg_pv[0], res0.leg_pv[1]  # receive=fixed(+), pay=float(-)
        par = r0 * (-float_pv) / fixed_pv

        res = pricer.price([Swap.fixed_float_swap(notional=100, rate_index="SOFR", fixed_rate=par, tenor="10Y", as_of=AS_OF)], mkt)
        self.assertEqual(res.instrument_pv.shape[0], 1)
        self.assertEqual(res.leg_pv.shape[0], 2)
        self.assertAlmostEqual(res.pv, 0.0, places=8)

    def test_negative_notional_flips_pv(self):
        mkt = _market()
        pricer = SwapPricer()
        recv = pricer.price([Swap.fixed_float_swap(notional=100, rate_index="SOFR", fixed_rate=0.06, tenor="10Y", as_of=AS_OF)], mkt)
        pay = pricer.price([Swap.fixed_float_swap(notional=-100, rate_index="SOFR", fixed_rate=0.06, tenor="10Y", as_of=AS_OF)], mkt)
        self.assertAlmostEqual(recv.pv, -pay.pv, places=10)
        self.assertGreater(recv.pv, 0.0)  # receiving 6% above par is positive PV

    def test_compile_once_reprice_many(self):
        # one compiled portfolio, repriced against two different curves -> no rebuild
        mkt = _market()
        compiled = SwapPricer().compile([Swap.fixed_float_swap(notional=100, rate_index="SOFR", fixed_rate=0.045, tenor="10Y", as_of=AS_OF)])
        pv1 = compiled.price(mkt).pv

        ns2 = CurveNamespace()
        o = AS_OF.to_numpy()
        d = np.array([o, o + np.timedelta64(12 * 365, "D")], dtype="datetime64[D]")
        ns2.bind("USD.SOFR", ZeroCurve(d, np.array([1.0, np.exp(-0.05 * 12)])))  # 5% flat
        mkt2 = MarketContext(as_of_date=AS_OF, curves=ns2)
        pv2 = compiled.price(mkt2).pv
        self.assertNotAlmostEqual(pv1, pv2, places=4)  # different curve -> different PV, same compiled form

    def test_population_two_swaps(self):
        mkt = _market()
        swaps = [
            Swap.fixed_float_swap(notional=100, rate_index="SOFR", fixed_rate=0.04, tenor="10Y", as_of=AS_OF),
            Swap.fixed_float_swap(notional=50, rate_index="SOFR", fixed_rate=0.05, tenor="5Y", as_of=AS_OF),
        ]
        res = SwapPricer().price(swaps, mkt)
        self.assertEqual(res.instrument_pv.shape[0], 2)
        self.assertEqual(res.leg_pv.shape[0], 4)
        self.assertTrue(np.all(np.isfinite(res.instrument_pv)))

    def test_cashflow_report(self):
        mkt = _market()
        res = SwapPricer().price([Swap.fixed_float_swap(notional=100, rate_index="SOFR", fixed_rate=0.045, tenor="5Y", as_of=AS_OF)], mkt)
        cf = res.cashflows
        self.assertIsNotNone(cf)
        # fixed leg flows carry the fixed rate
        fixed_flows = cf.leg == 0
        np.testing.assert_allclose(cf.rate[fixed_flows], 0.045, atol=1e-12)
