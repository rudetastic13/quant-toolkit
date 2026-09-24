"""Relative benchmarks with stored expectations: the building block for performance tests.

Absolute timings are machine noise; **ratios** to a base case that does the same amount of
work with the simplest possible numpy call are stable enough to store and assert on.  The
pattern every performance test follows:

1. ``Benchmark``: one base case (``np.interp``, ``np.exp(-z * t)``, ...) and contenders.
   Each contender's first call is a *correctness gate* against the base result and doubles
   as the warm-up, so JIT compilation and cache loads sit outside the timing window.
   Timing is best-of-``repeats`` wall-clock per call, ``reps`` calls per sample.
2. ``BenchmarkSuite``: collects results, writes a markdown report to
   ``pytest-reports/benchmarks/<suite>.md``, and checks every contender's ratio against
   the **stored expects** (a ``ratios`` table in an expects module, same markdown-in-module
   format as ``common.testing.expects_loader``).  A contender slower than
   ``expected * (1 + tolerance)`` fails the test; one faster than
   ``expected / (1 + tolerance)`` is reported as an improvement so the expects get refreshed.
3. ``BENCHMARK_UPDATE_EXPECTS=1`` rewrites the expects module from the current run instead
   of checking.  Commit the result.

Run with coverage disabled so instrumentation does not skew pure-Python contenders::

    pytest -m performance --no-cov -s
    BENCHMARK_UPDATE_EXPECTS=1 pytest -m performance --no-cov -s   # refresh expects

Assert only on ratios (and on orderings that hold on any machine), never on absolutes.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common.testing.expects_loader import ModuleLoader

Compare = Callable[[object, object], None]

UPDATE_ENV = "BENCHMARK_UPDATE_EXPECTS"
REPORT_DIR = Path("pytest-reports/benchmarks")


def allclose(expected: object, actual: object) -> None:
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-10, atol=1e-12)


def array_equal(expected: object, actual: object) -> None:
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@dataclass(frozen=True)
class Timing:
    name: str
    per_call: float  # seconds, best of ``repeats`` samples
    first_call: float  # seconds, includes any JIT / cache warm-up (0.0 for the base)
    ratio: float  # per_call / base per_call


@dataclass
class Benchmark:
    """One base case and the contenders measured against it.

    ``compare(expected, actual)`` is the correctness gate applied to each contender's first
    result; pass ``compare=None`` (on the benchmark or a contender) when results are not
    comparable to the base, e.g. a constructed object.
    """

    title: str
    base_name: str
    base: Callable[[], object]
    reps: int
    repeats: int = 5
    compare: Compare | None = allclose
    _contenders: list[tuple[str, Callable[[], object], Compare | None]] = field(default_factory=list, repr=False)

    def add(self, name: str, fn: Callable[[], object], *, compare: Compare | None | str = "base") -> Benchmark:
        """Register a contender.  ``compare`` defaults to the benchmark's; ``None`` skips the gate."""
        gate = self.compare if isinstance(compare, str) else compare
        self._contenders.append((name, fn, gate))
        return self

    def run(self) -> BenchmarkResult:
        expected = self.base()
        base_per_call = self._time(self.base)
        timings = [Timing(self.base_name, base_per_call, 0.0, 1.0)]
        for name, fn, compare in self._contenders:
            t0 = time.perf_counter()
            actual = fn()
            first_call = time.perf_counter() - t0
            if compare is not None:
                try:
                    compare(expected, actual)
                except AssertionError as exc:
                    raise AssertionError(f"{self.title}/{name} does not reproduce the base case: {exc}") from None
            per_call = self._time(fn)
            timings.append(Timing(name, per_call, first_call, per_call / base_per_call))
        return BenchmarkResult(self.title, self.reps, tuple(timings))

    def _time(self, fn: Callable[[], object]) -> float:
        best = float("inf")
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            for _ in range(self.reps):
                fn()
            best = min(best, (time.perf_counter() - t0) / self.reps)
        return best


@dataclass(frozen=True)
class BenchmarkResult:
    title: str
    reps: int
    timings: tuple[Timing, ...]

    def __getitem__(self, name: str) -> Timing:
        for timing in self.timings:
            if timing.name == name:
                return timing
        raise KeyError(name)

    def ratio(self, name: str) -> float:
        return self[name].ratio

    def table(self) -> str:
        lines = [
            f"{self.title}  ({self.reps} calls/sample, best of samples)",
            f"{'impl':<36}{'per-call':>12}{'vs base':>10}{'first-call':>13}",
            "-" * 71,
        ]
        for t in self.timings:
            first = f"{t.first_call * 1e3:>10.2f} ms" if t.first_call else f"{'-':>13}"
            lines.append(f"{t.name:<36}{t.per_call * 1e6:>10.1f} us{t.ratio:>9.1f}x{first}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        rows = [f"### {self.title}", "", "| impl | per-call (us) | vs base | first-call (ms) |", "|---|---:|---:|---:|"]
        for t in self.timings:
            first = f"{t.first_call * 1e3:.2f}" if t.first_call else "-"
            rows.append(f"| {t.name} | {t.per_call * 1e6:.1f} | {t.ratio:.1f}x | {first} |")
        return "\n".join(rows) + "\n"


@dataclass
class BenchmarkSuite:
    """Run benchmarks, write the report, and check ratios against the stored expects.

    ``expects_module`` is a dotted module path (``"common.tests.performance.expects.line1d"``)
    whose ``ratios`` table has columns ``benchmark``, ``impl``, ``ratio``.  It is created on
    the first ``BENCHMARK_UPDATE_EXPECTS=1`` run next to ``expects_dir``.
    """

    name: str
    expects_module: str
    expects_dir: Path
    tolerance: float = 0.5  # +50%: laptop turbo / thermal noise on a ratio is well inside this
    results: list[BenchmarkResult] = field(default_factory=list)

    def run(self, benchmark: Benchmark) -> BenchmarkResult:
        result = benchmark.run()
        self.results.append(result)
        print("\n" + result.table())
        return result

    def finish(self) -> None:
        """Write the report; then update the expects or check the run against them."""
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = REPORT_DIR / f"{self.name}.md"
        report.write_text(f"# {self.name}\n\n" + "\n".join(r.to_markdown() for r in self.results))
        print(f"\nreport: {report}")
        if os.environ.get(UPDATE_ENV, "") not in ("", "0", "false"):
            self._update_expects()
        elif _coverage_active():
            # instrumentation slows pure-Python wrappers relative to numpy base cases, so the
            # ratios are not comparable to expects recorded with --no-cov: report only
            print("coverage is active: benchmark expects not checked (run with --no-cov)")
        else:
            self._check_expects()

    # -- expects ----------------------------------------------------------------------

    def _current(self) -> pd.DataFrame:
        rows = [
            {"benchmark": r.title, "impl": t.name, "ratio": t.ratio}
            for r in self.results
            for t in r.timings
            if t.first_call  # skip the base row (ratio 1 by definition)
        ]
        return pd.DataFrame(rows, columns=["benchmark", "impl", "ratio"])

    def _update_expects(self) -> None:
        path = self.expects_dir / (self.expects_module.rsplit(".", 1)[-1] + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "__init__.py").touch()
        with open(path, "w") as stream:
            ModuleLoader(self.expects_module).update(self.name, {"ratios": self._current()}, stream=stream)
        print(f"expects updated: {path}")

    def _check_expects(self) -> None:
        try:
            stored = ModuleLoader(self.expects_module).load()["ratios"]
        except (ModuleNotFoundError, KeyError):
            raise AssertionError(
                f"no stored expects for benchmark suite '{self.name}' "
                f"({self.expects_module}); run once with {UPDATE_ENV}=1 and commit the result"
            ) from None
        expected = {(row.benchmark, row.impl): float(row.ratio) for row in stored.itertuples()}
        regressions, improvements, missing = [], [], []
        for row in self._current().itertuples():
            key = (row.benchmark, row.impl)
            if key not in expected:
                missing.append(f"{key[0]} / {key[1]}")
                continue
            bound = expected[key] * (1.0 + self.tolerance)
            if row.ratio > bound:
                regressions.append(
                    f"{key[0]} / {key[1]}: {row.ratio:.2f}x vs expected {expected[key]:.2f}x (limit {bound:.2f}x)"
                )
            elif row.ratio < expected[key] / (1.0 + self.tolerance):
                improvements.append(f"{key[0]} / {key[1]}: {row.ratio:.2f}x vs expected {expected[key]:.2f}x")
        if improvements:
            print("\nfaster than expects (refresh with " + UPDATE_ENV + "=1):\n  " + "\n  ".join(improvements))
        problems = []
        if missing:
            problems.append("no stored expect for:\n  " + "\n  ".join(missing))
        if regressions:
            problems.append("slower than stored expects:\n  " + "\n  ".join(regressions))
        if problems:
            raise AssertionError(
                f"benchmark suite '{self.name}':\n"
                + "\n".join(problems)
                + f"\n(refresh deliberately with {UPDATE_ENV}=1)"
            )


def _coverage_active() -> bool:
    cov = sys.modules.get("coverage")
    return cov is not None and cov.Coverage.current() is not None


def expects_dir_for(module_name: str) -> Path:
    """``expects/`` directory next to the test module that calls this."""
    return Path(importlib.import_module(module_name).__file__).parent / "expects"


__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkSuite",
    "Timing",
    "allclose",
    "array_equal",
    "expects_dir_for",
    "UPDATE_ENV",
]
