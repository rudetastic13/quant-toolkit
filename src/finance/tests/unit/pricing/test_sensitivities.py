"""Sensitivities: DV01 and key-rate durations by bump-and-reprice (no rebuild)."""
import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from finance.markets.curves import ZeroCurve
from finance.markets.context import MarketContext
from finance.markets.curves import CurveNamespace
from finance.instruments.resolution import Swap
from finance.pricing.pricers import SwapPricer
from finance.pricing.risk import Sensitivities

AS_OF = Date(2026, 6, 1)


def _market():
    o = AS_OF.to_numpy()
    dates = np.array([o, o + np.timedelta64(2 * 365, "D"), o + np.timedelta64(5 * 365, "D"),
                      o + np.timedelta64(12 * 365, "D")], dtype="datetime64[D]")
    t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
    ns = CurveNamespace()
    ns.bind("USD.SOFR", ZeroCurve(dates, np.exp(-0.04 * t)))
    return MarketContext(as_of_date=AS_OF, curves=ns)


class TestDv01(UnitTest):
    COVERAGE = ["finance.pricing.risk.sensitivities"]

    def setUp(self):
        self.mkt = _market()
        self.program = SwapPricer().compile(
            [Swap(notional=100, rate_index="SOFR", fixed_rate=0.04, tenor="10Y", as_of=AS_OF)]
        )
        self.sens = Sensitivities(self.program, self.mkt)

    def test_receiver_dv01_is_negative(self):
        # receive fixed: PV falls when rates rise -> negative DV01
        dv01 = self.sens.dv01("USD.SOFR")
        self.assertEqual(dv01.shape[0], 1)
        self.assertLess(dv01[0], 0.0)

    def test_dv01_does_not_mutate_base_market(self):
        before = self.mkt.discount("USD.SOFR").node_dfs.copy()
        _ = self.sens.dv01("USD.SOFR")
        after = self.mkt.discount("USD.SOFR").node_dfs
        np.testing.assert_array_equal(before, after)  # bumping built a fresh market

    def test_dv01_scale_reasonable(self):
        # ~ notional * duration * 1bp; 10y receiver on 100 notional -> O(0.05-0.1)
        dv01 = abs(self.sens.dv01("USD.SOFR")[0])
        self.assertGreater(dv01, 0.001)
        self.assertLess(dv01, 1.0)


class TestKeyRateDurations(UnitTest):
    COVERAGE = ["finance.pricing.risk.sensitivities"]

    def setUp(self):
        self.mkt = _market()
        self.program = SwapPricer().compile(
            [Swap(notional=100, rate_index="SOFR", fixed_rate=0.04, tenor="10Y", as_of=AS_OF)]
        )
        self.sens = Sensitivities(self.program, self.mkt)

    def test_shape_matches_pillars(self):
        ladder = self.sens.key_rate_durations("USD.SOFR")
        # 4-node curve -> 3 non-origin pillars
        self.assertEqual(ladder.krd.shape, (1, 3))
        self.assertEqual(ladder.pillar_years.shape[0], 3)

    def test_additivity_sum_equals_parallel_dv01(self):
        # key-rate additivity: sum of pillar bumps ~ parallel bump (to first order)
        ladder = self.sens.key_rate_durations("USD.SOFR")
        dv01 = self.sens.dv01("USD.SOFR")
        np.testing.assert_allclose(ladder.total, dv01, rtol=1e-3, atol=1e-6)

    def test_two_instruments_independent_ladders(self):
        program = SwapPricer().compile([
            Swap(notional=100, rate_index="SOFR", fixed_rate=0.04, tenor="10Y", as_of=AS_OF),
            Swap(notional=100, rate_index="SOFR", fixed_rate=0.04, tenor="2Y", as_of=AS_OF),
        ])
        ladder = Sensitivities(program, self.mkt).key_rate_durations("USD.SOFR")
        self.assertEqual(ladder.krd.shape, (2, 3))
        # the 2y swap should carry ~no risk at the 12y pillar; the 10y swap carries more
        self.assertGreater(abs(ladder.krd[0, -1]), abs(ladder.krd[1, -1]))
