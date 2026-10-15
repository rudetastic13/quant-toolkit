"""Deposits, FRAs and futures price through the same leg lowering as swaps."""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.instruments.resolution import Deposit, Fra, SofrFuture
from finance.markets.context import MarketContext
from finance.markets.curves import CurveNamespace, YieldCurve
from finance.markets.rate_generator import RateGenerator
from finance.pricing.pricers import FuturesPricer, LegPricer
from finance.pricing.types import RateKind

AS_OF = Date(2026, 6, 1)
CURVE = "USD.SOFR.ON"


def _market() -> MarketContext:
    origin = AS_OF.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 90, 365, 3 * 365, 7 * 365)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.041, 0.040, 0.042, 0.045])
    ns = CurveNamespace()
    sofr = YieldCurve.build(dates, np.exp(-z * t), currency="USD", index_name="SOFR")
    ns.bind(sofr)
    ns.bind(sofr, "USD.STDCSA")
    return MarketContext(as_of_date=AS_OF, curves=ns)


def _window(instrument):
    return (
        np.array([instrument.effective.to_numpy()], dtype="datetime64[D]"),
        np.array([instrument.maturity.to_numpy()], dtype="datetime64[D]"),
    )


class TestDeposit(UnitTest):
    COVERAGE = ["finance.pricing.pricers.legs", "finance.instruments.resolution.builders", "finance.pricing.kernels.compiler"]

    def setUp(self):
        self.market = _market()

    def _par(self, deposit) -> float:
        starts, ends = _window(deposit)
        return float(RateGenerator(self.market).simple_rate(CURVE, starts, ends, deposit.day_count_method)[0])

    def test_lowers_to_coupon_plus_principal_out_and_back(self):
        deposit = Deposit.spot_deposit(rate=0.04, tenor="6M", as_of=AS_OF, notional=100.0)
        ki = LegPricer().compile([deposit]).inputs
        self.assertEqual(ki.n_flows, 3)
        self.assertTrue(np.all(ki.rate_kind == RateKind.Fixed))
        np.testing.assert_array_equal(ki.notional, [100.0, -100.0, 100.0])
        np.testing.assert_array_equal(ki.fixed_rate, [0.04, 1.0, 1.0])
        np.testing.assert_array_equal(ki.period_frac[1:], [1.0, 1.0])
        self.assertEqual(ki.pay_dates[1], deposit.effective.to_numpy())
        self.assertEqual(ki.pay_dates[2], deposit.maturity.to_numpy())

    def test_prices_to_zero_at_the_curve_simple_rate(self):
        probe = Deposit.spot_deposit(rate=0.0, tenor="6M", as_of=AS_OF)
        par = Deposit.spot_deposit(rate=self._par(probe), tenor="6M", as_of=AS_OF, notional=1e6)
        self.assertAlmostEqual(par(self.market).pv, 0.0, places=6)

    def test_lending_above_market_has_positive_pv(self):
        probe = Deposit.spot_deposit(rate=0.0, tenor="6M", as_of=AS_OF)
        rich = Deposit.spot_deposit(rate=self._par(probe) + 0.01, tenor="6M", as_of=AS_OF, notional=1e6)
        self.assertGreater(rich(self.market).pv, 0.0)


class TestFra(UnitTest):
    COVERAGE = ["finance.pricing.pricers.legs", "finance.instruments.resolution.builders"]

    def setUp(self):
        self.market = _market()

    def _forward(self, fra) -> float:
        starts, ends = _window(fra)
        return float(RateGenerator(self.market).simple_rate(CURVE, starts, ends, fra.day_count_method)[0])

    def test_lowers_to_fixed_against_single_fixing_float(self):
        fra = Fra.forward_starting(rate=0.04, start="3M", end="6M", as_of=AS_OF)
        ki = LegPricer().compile([fra]).inputs
        self.assertEqual(ki.n_flows, 2)
        np.testing.assert_array_equal(ki.rate_kind, [RateKind.Fixed, RateKind.Float])
        np.testing.assert_array_equal(ki.sign, [1.0, -1.0])
        self.assertEqual(ki.reset_starts[1], fra.effective.to_numpy())
        self.assertEqual(ki.reset_ends[1], fra.maturity.to_numpy())

    def test_prices_to_zero_at_the_forward(self):
        probe = Fra.forward_starting(rate=0.0, start="3M", end="6M", as_of=AS_OF)
        par = Fra.forward_starting(rate=self._forward(probe), start="3M", end="6M", as_of=AS_OF, notional=1e6)
        self.assertAlmostEqual(par(self.market).pv, 0.0, places=6)

    def test_notional_sign_sets_direction(self):
        probe = Fra.forward_starting(rate=0.0, start="3M", end="6M", as_of=AS_OF)
        rate = self._forward(probe) + 0.01
        receive = Fra.forward_starting(rate=rate, start="3M", end="6M", as_of=AS_OF, notional=1e6)
        pay = Fra.forward_starting(rate=rate, start="3M", end="6M", as_of=AS_OF, notional=-1e6)
        self.assertGreater(receive(self.market).pv, 0.0)
        self.assertAlmostEqual(receive(self.market).pv, -pay(self.market).pv, places=10)


class TestFuturesLeg(UnitTest):
    COVERAGE = ["finance.pricing.pricers.futures", "finance.instruments.resolution.futures"]

    def test_reference_period_is_one_leg_over_every_weekday(self):
        future = SofrFuture.from_contract_month(contract="SR3", month="U26", as_of=AS_OF)
        [(leg, sign)] = future.legs
        self.assertEqual((leg.effective, leg.maturity, sign), (future.ref_start, future.ref_end, 1.0))
        ki = FuturesPricer().compile([future]).program.inputs
        self.assertEqual(ki.n_flows, 1)
        self.assertEqual(ki.n_obs_periods, 1)
        expected = int(np.busday_count(future.ref_start.to_numpy(), future.ref_end.to_numpy()))
        self.assertEqual(ki.obs_w.size, expected)
