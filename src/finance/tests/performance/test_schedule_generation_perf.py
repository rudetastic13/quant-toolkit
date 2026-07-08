"""Performance comparison of the schedule-generation implementations.

For each frequency the base case is the cheapest possible np.arange date grid
over 2025-2030, and every implementation is reported relative to it. Schedule
endpoints are chosen so each contender must reproduce the base grid exactly
(correctness gate) before it is timed.

Run with coverage disabled so instrumentation does not distort the pure-Python
contenders relative to numba/C++:

    pytest -m performance --no-cov -s
"""
import time

import numpy as np
import pytest

from common.testing import PerformanceTest
from finance._core import is_available as core_available
from finance.dates.date import Date
from finance.dates.enums import Frequency, Roll
from finance.dates.schedules import generate_schedule
from finance.dates.schedules.compiled import generate_schedule_cpp
from finance.dates.schedules.loop_based import generate_schedule_loop
from finance.dates.schedules.numba_based import generate_schedule_numba, is_available as numba_available

_REPEATS = 5


def _time_per_call(fn, reps: int) -> float:
    """Best-of-_REPEATS seconds per call over reps calls"""
    best = float("inf")
    for _ in range(_REPEATS):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


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
        lines = [""]
        for frequency, (base_fn, end, reps) in _CASES.items():
            base_case = base_fn()
            args = (start, end, frequency)
            kwargs = {"roll_convention": Roll.RollDay1}

            contenders = [("base np.arange", base_fn)]
            contenders.append(("while-loop (py)", lambda: generate_schedule_loop(*args, **kwargs)))
            contenders.append(("vectorized (np)", lambda: generate_schedule(*args, **kwargs)))
            if numba_available():
                contenders.append(("numba @njit", lambda: generate_schedule_numba(*args, **kwargs)))
            if core_available():
                contenders.append(("C++ pybind11", lambda: generate_schedule_cpp(*args, **kwargs)))

            # correctness gate + warm-up (numba JIT / cache load happens here, outside timing)
            warmup_times = {}
            for name, fn in contenders:
                t0 = time.perf_counter()
                result = fn()
                warmup_times[name] = time.perf_counter() - t0
                np.testing.assert_array_equal(
                    base_case, result, err_msg=f"{frequency.name}/{name} does not reproduce the base case"
                )

            timings = {name: _time_per_call(fn, reps) for name, fn in contenders}

            base_time = timings["base np.arange"]
            lines += [
                f"{frequency.name}: 2025-01-01 -> {end.to_str()}, {base_case.size} dates",
                f"{'impl':<18}{'per-call':>12}{'vs base':>10}{'first-call':>13}",
                "-" * 53,
            ]
            for name, _ in contenders:
                per_call = timings[name]
                lines.append(
                    f"{name:<18}{per_call * 1e6:>10.1f} us{per_call / base_time:>9.1f}x"
                    f"{warmup_times[name] * 1e3:>10.2f} ms"
                )
            lines.append("")

            # only assert orderings that are robust across machines; the rest is reported
            if numba_available():
                self.assertLess(timings["numba @njit"], timings["while-loop (py)"])
            if core_available():
                self.assertLess(timings["C++ pybind11"], timings["while-loop (py)"])
        print("\n".join(lines))
