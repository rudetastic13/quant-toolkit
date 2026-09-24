"""ZeroCurve / YieldCurve against raw numpy base cases.

Bulk query is the pricing shape: 40 pillars, 100k dates, one call.  Rebinding is the
calibration and bump-and-reprice shape: new discount factors on fixed pillars.

    pytest -m performance --no-cov -s src/finance/tests/performance/test_curve_perf.py
    BENCHMARK_UPDATE_EXPECTS=1 pytest -m performance --no-cov -s src/finance/tests/performance/test_curve_perf.py
"""

import numpy as np

from common.math.interpolation import Cubic, Linear
from common.testing import PerformanceTest
from common.testing.benchmark import Benchmark, BenchmarkSuite, expects_dir_for
from finance.markets.curves import CurveSpace, YieldCurve, ZeroCurve, dates_to_x

ORIGIN = np.datetime64("2026-06-01", "D")
X = np.concatenate([[0.0], np.sort(np.random.default_rng(3).uniform(30.0, 40.0 * 365.0, 39))]).round()
Z = 0.03 + 0.015 * (1.0 - np.exp(-X / 2000.0))
DFS = np.exp(-Z * X / 365.0)
DFS[0] = 1.0
XQ = np.sort(np.random.default_rng(5).uniform(0.0, X[-1], 100_000)).round()
DATES = ORIGIN + XQ.astype(np.int64).astype("timedelta64[D]")
LOG_DF = np.log(DFS)

EXPECTS_DIR = expects_dir_for(__package__)


def _suite(name: str) -> BenchmarkSuite:
    return BenchmarkSuite(name, f"{__package__}.expects.{name}", EXPECTS_DIR)


class TestCurvePerformance(PerformanceTest):
    COVERAGE = ["finance.markets.curves._curve_impl.zero_curve", "finance.markets.curves.yield_curve"]

    def test_bulk_discount_factor_query(self):
        log_linear = ZeroCurve(X, DFS)
        log_cubic = ZeroCurve(X, DFS, interpolator=Cubic())
        rate_linear = ZeroCurve(X, DFS, space=CurveSpace.ZeroRate, interpolator=Linear())
        yc = YieldCurve.from_registry(ORIGIN, log_linear, currency="USD", index_name="SOFR")
        suite = _suite("curve_query")

        result = suite.run(
            Benchmark(
                "DF query 100k, 40 pillars",
                "np.exp(np.interp(xq, x, ln df))",
                lambda: np.exp(np.interp(XQ, X, LOG_DF)),
                reps=20,
            )
            .add("ZeroCurve LogDF/Linear", lambda: log_linear.discount_factor(XQ))
            .add("YieldCurve (dates in)", lambda: yc.discount_factor(DATES))
            .add("dates_to_x alone", lambda: dates_to_x(ORIGIN, DATES), compare=None)
            .add("ZeroCurve LogDF/Cubic", lambda: log_cubic.discount_factor(XQ), compare=None)
            .add("ZeroCurve ZeroRate/Linear", lambda: rate_linear.discount_factor(XQ), compare=None)
            .add("ZeroCurve zero_rate LogDF/Linear", lambda: log_linear.zero_rate(XQ), compare=None)
        )
        self.assertLess(result.ratio("ZeroCurve LogDF/Linear"), 6.0)
        # the dates layer is one int64 view + subtraction; it must not dominate the query
        self.assertLess(result["YieldCurve (dates in)"].per_call, 2.0 * result["ZeroCurve LogDF/Linear"].per_call)
        suite.finish()

    def test_rebinding_hot_path(self):
        t = X[1:] / 365.0
        z2 = Z[1:] + 1e-4
        linear = ZeroCurve(X, DFS)
        cubic = ZeroCurve(X, DFS, interpolator=Cubic())
        suite = _suite("curve_rebind")

        def dfs_from_z(z):
            return np.concatenate([[1.0], np.exp(-z * t)])

        new_dfs = dfs_from_z(z2)
        result = suite.run(
            Benchmark(
                "rebind 40 pillars (calibration step)", "np.exp(-z * t)", lambda: dfs_from_z(z2), reps=200, compare=None
            )
            .add("ZeroCurve(LogDF/Linear) fresh", lambda: ZeroCurve(X, new_dfs))
            .add("ZeroCurve(LogDF/Linear).with_dfs", lambda: linear.with_dfs(new_dfs))
            .add("ZeroCurve(LogDF/Cubic) fresh", lambda: ZeroCurve(X, new_dfs, interpolator=Cubic()))
            .add("ZeroCurve(LogDF/Cubic).with_dfs", lambda: cubic.with_dfs(new_dfs))
        )
        self.assertLessEqual(
            result.ratio("ZeroCurve(LogDF/Linear).with_dfs"), result.ratio("ZeroCurve(LogDF/Linear) fresh") * 1.1
        )
        suite.finish()
