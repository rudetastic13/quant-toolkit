"""End-to-end proof: a par compounded-SOFR swap prices to ~0 using only real components.

Chain exercised: ZeroCurve -> YieldCurve -> RateGenerator (overnight simple rates) ->
compounded_rate kernel (flatten-and-reduceat) -> dcf kernel (portfolio reduceat).

No resolution/schedule plumbing here — this validates the numeric core. The schedule
observation-grid builder and trader resolution layer wire these same pieces together.
"""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date, DayCountMethod
from finance.markets.curves import YieldCurve, ZeroCurve
from finance.markets.context import MarketContext
from finance.markets.curves import CurveNamespace
from finance.markets.rate_generator import RateGenerator
from finance.pricing.engines.numpy.rates import compounded
from finance.pricing.engines.numpy.dcf import dcf

ACT360 = DayCountMethod.Actual360
CURVE = "USD.SOFR"


def _curve(origin: str) -> ZeroCurve:
    # Smooth descending DFs ~ 4% continuously compounded over 3y of pillars.
    dates = np.array([origin, "2027-06-01", "2028-06-01", "2029-06-01"], dtype="datetime64[D]")
    t = (dates.astype("datetime64[D]").astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    dfs = np.exp(-0.04 * t)
    return ZeroCurve(dates, dfs)


def _obs_windows(period_start, period_end):
    """Business-day observation windows tiling [period_start, period_end) exactly.

    no_holidays calendar == Mon-Fri, so np.is_busday (default weekmask) matches it.
    """
    days = np.arange(period_start, period_end, dtype="datetime64[D]")
    pts = days[np.is_busday(days)]
    if pts.size == 0 or pts[0] != period_start:
        pts = np.concatenate([[period_start], pts])
    obs_start = pts
    obs_end = np.concatenate([pts[1:], [period_end]])
    return obs_start, obs_end


class TestParSwapPricesToZero(UnitTest):
    COVERAGE = [
        "finance.pricing.engines.numpy.rates",
        "finance.pricing.engines.numpy.dcf",
        "finance.markets.context",
    ]

    def setUp(self):
        self.origin = "2026-06-01"
        ns = CurveNamespace()
        ns.bind(
            YieldCurve.from_registry(
                _curve(self.origin),
                currency="USD",
                index_name="SOFR",
            )
        )
        self.mkt = MarketContext(as_of_date=Date(2026, 6, 1), curves=ns)
        self.curve = self.mkt.zero_curve(CURVE)
        self.rates = RateGenerator(self.mkt)
        # 2y annual swap, effective = curve origin
        self.boundaries = np.array([self.origin, "2027-06-01", "2028-06-01"], dtype="datetime64[D]")
        self.notional = 100.0

    def _float_cashflows(self):
        """Per-period float cashflows via the real project + compounded kernel."""
        starts = self.boundaries[:-1]
        ends = self.boundaries[1:]
        cashflows = []
        for ps, pe in zip(starts, ends):
            obs_s, obs_e = _obs_windows(ps, pe)
            w = (obs_e.astype(np.int64) - obs_s.astype(np.int64)) / 360.0
            r = self.rates.simple_rate(CURVE, obs_s, obs_e, ACT360)
            rate = compounded(r, w, np.array([0]))[0]
            period_frac = w.sum()  # Act/360 of the whole period = sum of sub-windows
            cashflows.append(self.notional * rate * period_frac)
        return np.array(cashflows), ends

    def test_float_leg_telescopes_to_df(self):
        cash, ends = self._float_cashflows()
        df_b = self.curve.discount_factor(self.boundaries)
        # cashflow_i should equal notional*(DF(start_i)/DF(end_i) - 1)
        expected = self.notional * (df_b[:-1] / df_b[1:] - 1.0)
        np.testing.assert_allclose(cash, expected, atol=1e-10)

    def test_par_swap_pv_zero(self):
        float_cash, ends = self._float_cashflows()
        df_ends = self.curve.discount_factor(ends)
        period_fracs = np.array(
            [
                (self.boundaries[i + 1].astype(np.int64) - self.boundaries[i].astype(np.int64)) / 360.0
                for i in range(len(ends))
            ]
        )
        annuity = float(np.sum(period_fracs * df_ends))
        df_n = float(df_ends[-1])
        par_rate = (1.0 - df_n) / annuity  # OIS par rate

        fixed_cash = self.notional * par_rate * period_fracs

        # Receive fixed (+), pay float (-): one dcf call over the packed two-leg portfolio.
        cash = np.concatenate([fixed_cash, float_cash])
        df = np.concatenate([df_ends, df_ends])
        sign = np.concatenate([np.ones_like(fixed_cash), -np.ones_like(float_cash)])
        row_offsets = np.array([0, len(fixed_cash)])  # leg 0 fixed, leg 1 float

        pv = dcf(cash, df, sign, row_offsets)
        net = float(pv.sum())
        self.assertAlmostEqual(net, 0.0, places=10)

    def test_off_par_pv_sign(self):
        float_cash, ends = self._float_cashflows()
        df_ends = self.curve.discount_factor(ends)
        period_fracs = np.array(
            [
                (self.boundaries[i + 1].astype(np.int64) - self.boundaries[i].astype(np.int64)) / 360.0
                for i in range(len(ends))
            ]
        )
        annuity = float(np.sum(period_fracs * df_ends))
        par_rate = (1.0 - float(df_ends[-1])) / annuity

        # Receiving ABOVE par should be positive PV to the receiver.
        high = par_rate + 0.01
        fixed_cash = self.notional * high * period_fracs
        cash = np.concatenate([fixed_cash, float_cash])
        df = np.concatenate([df_ends, df_ends])
        sign = np.concatenate([np.ones_like(fixed_cash), -np.ones_like(float_cash)])
        pv = dcf(cash, df, sign, np.array([0, len(fixed_cash)]))
        self.assertGreater(float(pv.sum()), 0.0)
