"""The real observation grid feeds project + compounded kernel and telescopes to DF.

This closes the loop: build_observation_grid -> RateGenerator -> compounded kernel,
across a multi-period leg, must reproduce the curve's DF ratios period-by-period.
"""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date, DayCountMethod, Frequency, BDC
from finance.markets.curves import YieldCurve, ZeroCurve
from finance.markets.context import MarketContext
from finance.markets.curves import CurveNamespace
from finance.markets.rate_generator import RateGenerator
from finance.instruments.schedules.payment_schedule import build_payment_schedule
from finance.instruments.schedules.observation import build_observation_grid
from finance.pricing.engines.numpy.rates import compounded

ACT360 = DayCountMethod.Actual360
CAL = "no_holidays"
CURVE = "USD.SOFR"


def _curve():
    dates = np.array(["2026-06-01", "2028-06-01", "2030-06-01"], dtype="datetime64[D]")
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    return ZeroCurve(dates, np.exp(-0.04 * t))


class TestRealGridTelescopes(UnitTest):
    COVERAGE = [
        "finance.instruments.schedules.observation",
        "finance.pricing.engines.numpy.rates",
    ]

    def setUp(self):
        ns = CurveNamespace()
        ns.bind(
            YieldCurve.from_registry(
                _curve(),
                currency="USD",
                index_name="SOFR",
            )
        )
        self.mkt = MarketContext(as_of_date=Date(2026, 6, 1), curves=ns)
        self.curve = self.mkt.zero_curve(CURVE)
        self.rates = RateGenerator(self.mkt)
        self.ps = build_payment_schedule(
            Date(2026, 6, 1), Date(2028, 6, 1), Frequency.SemiAnnually,
            ACT360, BDC.ModifiedFollowing, CAL, build_observations=True,
        )

    def test_compounded_period_rates_telescope(self):
        grid = build_observation_grid(self.ps.accrual_starts, self.ps.accrual_ends, CAL, ACT360)
        # read overnight simple rates over each sub-period (lookback=0 => value_date == sub_start)
        obs_rate = self.rates.simple_rate(CURVE, grid.sub_starts, grid.sub_ends, ACT360)
        rates = compounded(obs_rate, grid.weights, grid.offsets)

        # each period's compounded rate * period_frac == DF(start)/DF(end) - 1
        df_start = self.curve.discount_factor(self.ps.accrual_starts)
        df_end = self.curve.discount_factor(self.ps.accrual_ends)
        expected = (df_start / df_end - 1.0) / self.ps.period_fracs
        np.testing.assert_allclose(rates, expected, atol=1e-10)
