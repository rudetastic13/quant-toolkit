# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**rates-lib** (published as `quant-toolkit`) is a Python quantitative finance library focused on interest rate analytics. It provides foundational classes for date/time arithmetic, financial calendars, term calculations, and rate curve modeling.
The goal is to design interest rate analytics levering composition over inheritance often and making use of patterns defined in the **src/common** layer often.  We will leverage numpy, scipy, numba and cython where to speed up computations.

## Commands

### Environment Setup
```bash
# Activate conda environment
source bin/use_env.sh pricing
```

### Running Tests
```bash
# Run all tests with coverage
pytest

# Run only unit tests
pytest -m unit

# Run a specific test file
pytest src/finance/tests/unit/dates/test_date.py

# Run by marker (available: unit, hypothesis, datemath, market_data, instruments, risk, calculators, calibration, slow, integration, regression, performance)
pytest -m "unit and datemath"
```

### Linting & Formatting
```bash
ruff check src/       # lint
ruff format src/      # format
mypy src/             # type check
```

## Architecture

> **Pricing layer:** see [docs/pricing_architecture.md](docs/pricing_architecture.md) for the
> end-to-end flow (data row → instrument → `PricingProgram` → reprice), Mermaid diagrams,
> design decisions, and patterns (Registry/Factory, columnar IR, compile-once/reprice-many,
> flatten-and-`reduceat`).

### Package Layout
- `src/common/` — shared utilities (registry, singleton, array buffers, curve container, test base classes)
- `src/finance/` — core financial domain (dates, calendars, terms, markets/curves)
- `src/quant_toolkit/` — public API re-exporting from `finance`
- `src/finance/tests/` — test suite mirroring the source structure

### Core Design Patterns

**Registry + Factory** (`common/registry.py`): Generic plugin architecture used throughout. Supports multi-key registration and decorator-based registration via `@register_with()`. Used for term math operations and calendar management.

**Singleton** (`common/singleton.py`): Metaclass-based. Used for `Calendars()` — the global calendar registry.

**Protocol** (`Calendar`): Runtime-checkable protocol defines the calendar interface without requiring inheritance.

### Key Modules

**`finance/dates/date/`** — Immutable `Date` dataclass with conversions to/from YMD tuples, strings, `int` (YYYYMMDD), `numpy.datetime64`, `pandas.Timestamp`, and Python `datetime`. Ordinal is days since 1970-01-01.

**`finance/dates/term/`** — `Term` class for time periods (Days, Weeks, Months, Quarters, Years, BusinessDays). Implements `NDArrayOperatorsMixin` for native NumPy ufunc support. String parsing: `"5D"`, `"3M"`, `"1Y"`. Date arithmetic is dispatched through `term_math_registry.py`.

**`finance/dates/calendars/`** — `StaticCalendar` (weekday masks + holiday sets, wraps `numpy.busdaycalendar`). Calendars can be combined with `+`. `Calendars()` singleton is the global registry of available calendars.

**`finance/dates/enums/`** — `Frequency`, `BDC` (Business Day Convention), `Direction`, `Roll`, `TermType`.

**`finance/markets/curves/`** — `ZeroCurve` stores log discount factors internally (`-ln(DF_t)`). Constructed from discount factors, log DFs, or zero rates. Supports linear/flat interpolation and flat-forward or no extrapolation.

**`common/containers/curve1d.py`** — Generic 1D curve with pluggable interpolation/extrapolation, used as the backbone for `ZeroCurve`.

**`common/array_buffer.py`** — Pre-allocated NumPy array buffers (`ArrayBuffer`, `ExpandableArrayBuffer`, `PaginatedArrayBuffer`) for performance-sensitive calculation loops.

### Test Infrastructure

Test base classes in `common/testing/__init__.py`:
- `UnitTest(TestCase)` — marks tests `@pytest.mark.unit`
- `IntegrationTest(TestCase)` — marks tests `@pytest.mark.integration`
- `PerformanceTest(TestCase)` — marks tests `@pytest.mark.performance`

Each declares a `COVERAGE = [...]` class attribute listing the modules it targets.

Custom assertions: `common/testing/numpy_array_asserts.py`. Expected data loading: `common/testing/expects_loader.py`.

Coverage minimum is 80% (`fail_under = 80`). Reports are written to `pytest-reports/` (HTML, XML, JUnit).

### Style & Tooling
- Line length: 120, double quotes, 4-space indent (enforced by ruff)
- Target: Python 3.11+; environment runs Python 3.14
- Type hints are used extensively; `NDArray`, `DTypeLike`, and custom aliases (`DateType`, `FloatArray`, etc.) are defined in type modules within each package