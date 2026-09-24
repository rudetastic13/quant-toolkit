"""Line1d and the interpolation schemes against raw numpy / scipy base cases.

Query benchmarks use a 50-node line and 100k in-range queries: the shape the pricing
engines hit (one call per curve with a large deduplicated date array).  Construction
benchmarks are the calibration hot path: 50 nodes, refit per solver iteration.

    pytest -m performance --no-cov -s src/common/tests/performance
    BENCHMARK_UPDATE_EXPECTS=1 pytest -m performance --no-cov -s src/common/tests/performance
"""

import numpy as np
from scipy.interpolate import CubicSpline

from common.math.interpolation import Cubic, Flat, Linear, Quadratic
from common.math.line import Extrapolation, Line1d
from common.testing import PerformanceTest
from common.testing.benchmark import Benchmark, BenchmarkSuite, expects_dir_for

RNG = np.random.default_rng(11)
X = np.cumsum(RNG.uniform(20.0, 400.0, 50))
X -= X[0]
Y = np.cumsum(RNG.normal(0.0, 0.01, 50))
XQ = np.sort(RNG.uniform(X[0], X[-1], 100_000))

EXPECTS_DIR = expects_dir_for(__package__)


def _suite(name: str) -> BenchmarkSuite:
    return BenchmarkSuite(name, f"{__package__}.expects.{name}", EXPECTS_DIR)


class TestLine1dPerformance(PerformanceTest):
    COVERAGE = ["common.math.line", "common.math.interpolation"]

    def test_query_vs_numpy(self):
        linear = Line1d(X, Y, Linear())
        flat = Line1d(X, Y, Flat())
        cubic = Line1d(X, Y, Cubic())
        spline = CubicSpline(X, Y, bc_type="natural")
        suite = _suite("line1d_query")

        q_linear = suite.run(
            Benchmark("query 100k: Linear", "np.interp", lambda: np.interp(XQ, X, Y), reps=20).add(
                "Line1d(Linear)", lambda: linear(XQ)
            )
        )
        q_flat = suite.run(
            Benchmark(
                "query 100k: Flat (previous-hold)",
                "y[searchsorted(x, xq, 'right') - 1]",
                lambda: Y[np.searchsorted(X, XQ, side="right") - 1],
                reps=20,
            ).add("Line1d(Flat)", lambda: flat(XQ))
        )
        q_cubic = suite.run(
            Benchmark("query 100k: Cubic", "scipy CubicSpline.__call__", lambda: spline(XQ), reps=20)
            .add("Line1d(Cubic)", lambda: cubic(XQ))
            .add("Line1d(Cubic).derivative", lambda: cubic.derivative(XQ), compare=None)
        )
        # Line1d adds a bounds split on top of the raw call; anything beyond a small multiple
        # means the wrapper, not the math, is the cost.  These hold on any machine.
        self.assertLess(q_linear.ratio("Line1d(Linear)"), 6.0)
        self.assertLess(q_flat.ratio("Line1d(Flat)"), 6.0)
        self.assertLess(q_cubic.ratio("Line1d(Cubic)"), 3.0)
        suite.finish()

    def test_construction_vs_numpy(self):
        y2 = Y + 0.001
        base_linear = Line1d(X, Y, Linear())
        base_cubic = Line1d(X, Y, Cubic())
        base_quadratic = Line1d(X, Y, Quadratic())
        suite = _suite("line1d_construction")

        build = suite.run(
            Benchmark(
                "construct 50 nodes", "np.diff(y) / np.diff(x)", lambda: np.diff(Y) / np.diff(X), reps=200, compare=None
            )
            .add("Line1d(Linear) fresh", lambda: Line1d(X, Y, Linear()))
            .add("Line1d(Linear).with_y", lambda: base_linear.with_y(y2))
            .add("Line1d(Cubic) fresh", lambda: Line1d(X, Y, Cubic()))
            .add("Line1d(Cubic).with_y", lambda: base_cubic.with_y(y2))
            .add("scipy CubicSpline alone", lambda: CubicSpline(X, Y, bc_type="natural"))
            .add("Line1d(Quadratic) fresh", lambda: Line1d(X, Y, Quadratic()))
            .add("Line1d(Quadratic).with_y", lambda: base_quadratic.with_y(y2))
        )
        # with_y skips the x checks and must never cost more than a fresh construction.
        self.assertLessEqual(build.ratio("Line1d(Linear).with_y"), build.ratio("Line1d(Linear) fresh") * 1.1)
        self.assertLessEqual(build.ratio("Line1d(Cubic).with_y"), build.ratio("Line1d(Cubic) fresh") * 1.1)
        suite.finish()

    def test_extrapolation_split(self):
        line = Line1d(X, Y, Linear(), left=Extrapolation.Flat, right=Extrapolation.Linear)
        xq_in = np.tile(XQ, 3)
        xq_out = np.concatenate([XQ - 1000.0, XQ, XQ + 5000.0])
        suite = _suite("line1d_extrapolation")
        result = suite.run(
            Benchmark("query 300k", "Line1d, all in range", lambda: line(xq_in), reps=10, compare=None).add(
                "Line1d, 2/3 out of range", lambda: line(xq_out)
            )
        )
        self.assertLess(result.ratio("Line1d, 2/3 out of range"), 3.0)
        suite.finish()
