"""Document load and dump against the bare ``json`` calls on the same text.

The base case is ``json.loads``: the floor for reading a document.  ``loads`` adds building the
objects (hints, dates, enums, nested documents); it should stay a small multiple of the parse.

    pytest -m performance --no-cov -s packages/finance/src/quant_toolkit/finance/tests/performance/test_serialization_perf.py
    BENCHMARK_UPDATE_EXPECTS=1 pytest -m performance --no-cov -s packages/finance/src/quant_toolkit/finance/tests/performance/test_serialization_perf.py
"""

import json

from quant_toolkit.common.object import dumps, loads
from quant_toolkit.common.testing import PerformanceTest
from quant_toolkit.common.testing.benchmark import Benchmark, BenchmarkSuite, expects_dir_for
from quant_toolkit.finance.dates import BDC, Date, DayCountMethod, Frequency
from quant_toolkit.finance.instruments import CommonInstrument
from quant_toolkit.finance.instruments.resolution import Swap
from quant_toolkit.finance.instruments.schedules import CouponSchedule, FixedCouponEvent

AS_OF = Date(2026, 6, 1)
SWAP = Swap.fixed_float_swap(notional=1e6, rate_index="USD.SOFR.1D", fixed_rate=0.04, tenor="5Y", as_of=AS_OF)
SCHEDULED = CommonInstrument(
    effective=AS_OF,
    maturity=Date(2056, 6, 1),
    currency="USD",
    notional=1e6,
    payment_frequency=Frequency.Quarterly,
    day_count_method=DayCountMethod.Actual360,
    business_day_convention=BDC.ModifiedFollowing,
    rate_index="USD.SOFR.1D",
    coupon_schedule=CouponSchedule(
        events=[FixedCouponEvent(start_date=Date(2027 + i // 12, i % 12 + 1, 1), coupon_rate=0.03) for i in range(118)]
    ),
)

EXPECTS_DIR = expects_dir_for(__package__)


class TestSerializationPerformance(PerformanceTest):

    def test_load_and_dump_vs_json(self):
        suite = BenchmarkSuite("serialization", f"{__package__}.expects.serialization", EXPECTS_DIR)
        results = {}
        for label, obj, reps in (("swap", SWAP, 500), ("instrument, 118 coupon events", SCHEDULED, 50)):
            text = dumps(obj)
            self.assertEqual(loads(text), obj)
            results[label] = suite.run(
                Benchmark(
                    f"document: {label}", "json.loads", lambda text=text: json.loads(text), reps=reps, compare=None
                )
                .add("loads", lambda text=text: loads(text))
                .add("dumps", lambda obj=obj: dumps(obj))
            )
        # Before hints were cached per class and dates parsed as ISO, loads was 40x-140x the parse.
        for result in results.values():
            self.assertLess(result.ratio("loads"), 20.0)
        suite.finish()
