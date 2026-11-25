"""The Numba program's historical-fixing overlay: skipped when unseasoned, reused across scenarios."""
import numpy as np

from quant_toolkit.common.testing import UnitTest
from quant_toolkit.finance.dates import Date
from quant_toolkit.finance.instruments.resolution import Swap
from quant_toolkit.finance.markets.context import MarketContext
from quant_toolkit.finance.markets.curves import YieldCurve
from quant_toolkit.finance.markets.fixings import HistoricalFixings
from quant_toolkit.finance.pricing.pricers import LegPricer
from quant_toolkit.finance.pricing.types import Backend

CURVE = "USD.SOFR.1D"


def _fixings(as_of: Date, level: float) -> HistoricalFixings:
    origin = as_of.to_numpy()
    dates = np.array([origin - np.timedelta64(d, "D") for d in range(400, 0, -1)])
    return HistoricalFixings(dates, np.full(dates.size, level))


def _market(as_of: Date, fixings: HistoricalFixings) -> MarketContext:
    origin = as_of.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 3 * 365, 7 * 365, 12 * 365)])
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    z = np.array([0.0, 0.035, 0.04, 0.045, 0.047])
    sofr = YieldCurve.build(dates, np.exp(-z * t), key=CURVE, historical_fixings=fixings)
    return MarketContext(as_of, {sofr.name: sofr, "USD.STDCSA": sofr})


class TestFixingsOverlay(UnitTest):

    def setUp(self):
        self.as_of = Date(2026, 6, 1)
        self.market = _market(self.as_of, _fixings(self.as_of, 0.04))

    def _swap(self, start: Date):
        return Swap.fixed_float_swap(
            notional=10e6, rate_index=CURVE, fixed_rate=0.04, effective=start, tenor="5Y", as_of=self.as_of
        )

    def test_unseasoned_book_skips_the_scan(self):
        program = LegPricer().compile([self._swap(Date(2026, 6, 3))], backend=Backend.Numba)
        prepared = program.prepare(self.market)
        self.assertFalse(prepared._requires_historical_fixings)
        self.assertIs(prepared._fixings_from_market(self.market), prepared._no_fixings)
        np.testing.assert_allclose(
            program.reprice(self.market).instrument_pv,
            program.reprice(self.market, backend=Backend.Numpy).instrument_pv, rtol=1e-12,
        )

    def test_seasoned_book_reuses_the_overlay_across_rate_scenarios(self):
        program = LegPricer().compile([self._swap(Date(2026, 1, 5))], backend=Backend.Numba)
        prepared = program.prepare(self.market)
        self.assertTrue(prepared._requires_historical_fixings)
        first = prepared._fixings_from_market(self.market)
        bumped = self.market.bump(CURVE, 1e-4)
        self.assertIs(prepared._fixings_from_market(bumped), first)          # same fixings object: reused
        for market in (self.market, bumped):
            np.testing.assert_allclose(
                program.reprice(market).instrument_pv,
                program.reprice(market, backend=Backend.Numpy).instrument_pv, rtol=1e-12,
            )

    def test_new_fixings_recompute(self):
        program = LegPricer().compile([self._swap(Date(2026, 1, 5))], backend=Backend.Numba)
        base = program.reprice(self.market).instrument_pv
        shocked = self.market.with_fixings(CURVE, _fixings(self.as_of, 0.06))
        moved = program.reprice(shocked).instrument_pv
        self.assertNotAlmostEqual(float(base[0]), float(moved[0]), delta=1.0)
        np.testing.assert_allclose(
            moved, program.reprice(shocked, backend=Backend.Numpy).instrument_pv, rtol=1e-12
        )
        np.testing.assert_allclose(program.reprice(self.market).instrument_pv, base, rtol=1e-15)   # and back
