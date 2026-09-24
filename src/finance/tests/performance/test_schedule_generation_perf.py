"""Performance comparison of the schedule-generation implementations.

For each frequency the base case is the cheapest possible np.arange date grid
over 2025-2030, and every implementation is reported relative to it. Schedule
endpoints are chosen so each contender must reproduce the base grid exactly
(correctness gate) before it is timed.  Ratios are checked against the stored
expects in ``expects/schedule_generation.py``.

Run with coverage disabled so instrumentation does not distort the pure-Python
contenders relative to numba/C++:

    pytest -m performance --no-cov -s
    BENCHMARK_UPDATE_EXPECTS=1 pytest -m performance --no-cov -s   # refresh expects
"""

import numpy as np
import pytest

from common.testing import PerformanceTest
from common.testing.benchmark import Benchmark, BenchmarkSuite, array_equal, expects_dir_for
from finance._core import is_available as core_available
from finance.dates.date import Date
from finance.dates.enums import Frequency, Roll
from finance.dates.schedules import generate_schedule
from finance.dates.schedules.compiled import generate_schedule_cpp
from finance.dates.schedules.loop_based import generate_schedule_loop
from finance.dates.schedules.numba_based import generate_schedule_numba, is_available as numba_available


def _arange_months(step: int) -> np.ndarray:
    m0, m1 = np.datetime64("2025-01", "M"), np.datetime64("2030-01", "M")
    return np.arange(m0, m1, np.timedelta64(step, "M")).astype("datetime64[D]")


def _arange_days(step: int) -> np.ndarray:
    d0, d1 = np.datetime64("2025-01-01", "D"), np.datetime64("2030-01-01", "D")
    return np.arange(d0, d1, np.timedelta64(step, "D"))


# frequency -> (base-case grid builder, matching schedule end date, timing reps)
# schedule start is always 2025-01-01 and the end date is the last base grid
# point, so inclusive-endpoint schedules reproduce the half-open arange exactly
_CASES = {
    Frequency.Annually: (lambda: _arange_months(12), Date(2029, 1, 1), 2_000),
    Frequency.SemiAnnually: (lambda: _arange_months(6), Date(2029, 7, 1), 2_000),
    Frequency.Quarterly: (lambda: _arange_months(3), Date(2029, 10, 1), 2_000),
    Frequency.Monthly: (lambda: _arange_months(1), Date(2029, 12, 1), 2_000),
    Frequency.Weekly: (lambda: _arange_days(7), Date(2029, 12, 26), 1_000),
    Frequency.Daily: (lambda: _arange_days(1), Date(2029, 12, 31), 200),
}


@pytest.mark.datemath
class TestScheduleGenerationPerformance(PerformanceTest):
    COVERAGE = [
        "finance.dates.schedules.vectorized",
        "finance.dates.schedules.compiled",
        "finance.dates.schedules.loop_based",
        "finance.dates.schedules.numba_based",
    ]

    def test_schedules_vs_arange_base_cases(self):
        start = Date(2025, 1, 1)
        suite = BenchmarkSuite(
            "schedule_generation", f"{__package__}.expects.schedule_generation", expects_dir_for(__package__)
        )
        for frequency, (base_fn, end, reps) in _CASES.items():
            args = (start, end, frequency)
            kwargs = {"roll_convention": Roll.RollDay1}
            bench = Benchmark(
                f"{frequency.name}: 2025-01-01 -> {end.to_str()}, {base_fn().size} dates",
                "base np.arange",
                base_fn,
                reps=reps,
                compare=array_equal,
            )
            bench.add("while-loop (py)", lambda: generate_schedule_loop(*args, **kwargs))
            bench.add("vectorized (np)", lambda: generate_schedule(*args, **kwargs))
            if numba_available():
                bench.add("numba @njit", lambda: generate_schedule_numba(*args, **kwargs))
            if core_available():
                bench.add("C++ pybind11", lambda: generate_schedule_cpp(*args, **kwargs))
            result = suite.run(bench)

            # only assert orderings that are robust across machines; the rest is reported
            if numba_available():
                self.assertLess(result["numba @njit"].per_call, result["while-loop (py)"].per_call)
            if core_available():
                self.assertLess(result["C++ pybind11"].per_call, result["while-loop (py)"].per_call)
        suite.finish()
