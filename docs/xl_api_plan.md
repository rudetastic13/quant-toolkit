# Excel add-in: the function surface and how it ships

> **Status: DRAFT FOR REVIEW (2026-11-11).** Written after the first end-to-end run of
> `quant_toolkit_xl` in desktop Excel (demo workbook prices the README swap in-cell). Covers the
> function catalogue to expose next, the `api/` package layout behind it, marshalling rules, the
> `.xlam` add-in and the `qtk-xl` CLI. Companion to [src/quant_toolkit_xl/README.md](../src/quant_toolkit_xl/README.md)
> (user-facing install and troubleshooting) and [pricing_architecture.md](pricing_architecture.md).

## 0. Decisions, so they are not re-litigated

1. **Names are `QTK_<AREA>_<Name>`.** Upper-case vendor and area prefix, PascalCase name:
   `QTK_DATE_AddTerm`, `QTK_CURVE_Build`, `QTK_MKT_Build`. xlwings turns the Python function name
   into a VBA `Function`, and VBA identifiers cannot contain `.`, so the dotted `QTK.DATE.AddTerm`
   is not available on this stack. The area prefix is the namespace: `=QTK_D` in a cell lists only
   the date functions. Excel is case-insensitive on entry and displays the name as defined.
2. **Six areas, not sixteen.** `DATE` (dates, terms, calendars, schedules), `CURVE`, `MKT`,
   `INST` (instrument builders), `CALC` (rate calculators and pricing), `REPORT`. A seventh area
   needs a reason. Verbs for things that build or run (`Build`, `Calibrate`, `Price`), nouns for
   things that read (`DF`, `Zero`, `ParRate`).
3. **One UDF per calculator, no generic dispatcher.** Explicit signatures give traders argument
   names in autocomplete, per-argument marshalling, and errors that name the argument.
4. **The Python module path mirrors the Excel name.** `QTK_CURVE_DF` calls `api.curves.df`;
   `QTK_DATE_AddTerm` calls `api.dates.add_term`. `api/` becomes a package with one module per
   area. `functions.py` stays the thin `#QERR` wrapper layer and is the only file that imports xlwings.
5. **Parameter names are a VBA contract.** xlwings copies them verbatim into the generated VBA; a
   VBA keyword (`currency`, `date`, `type`, `string`, ...) breaks compilation of the whole module.
   `test_functions.py` rejects reserved words; new UDFs inherit the guard.
6. **Handles stay content-addressed and immutable.** `Curve::`, `Inst::`, `Program::`,
   `Market::`. Identical inputs give identical handles, so recalculation is idempotent and
   scenario columns are cheap. A builder that *registers* something by name (calendars) returns
   the name so downstream cells depend on it and recalc in order.
7. **Errors are cell text, never `#VALUE!`.** Every UDF returns `#QERR: <message>` on failure;
   unknown enum or key names list the valid ones in the message.
8. **The add-in holds wrappers only.** The `.xlam` carries VBA wrappers, ribbon XML and a version
   stamp; all logic is in the pip-installed package. `pip install -U` updates logic; `qtk-xl
   install` is needed only when a signature changes. Nobody runs Import Functions on a trader
   machine.

## 1. Package layout

Decided 2026-11-11: the repo becomes a **namespace family** under `quant_toolkit`, one
distribution per portion, lockstep versions. The add-in is one portion (`quant_toolkit.xl`), so
library users never see Excel, and `pip install quant-toolkit[excel]` adds it.

```
pyproject.toml                     quant-toolkit: meta-package (depends on the portions; extra `excel`)
                                   + all tool config (ruff, mypy, pytest, coverage)
packages/common/pyproject.toml     quant-toolkit-common   -> src/quant_toolkit/common/
packages/finance/pyproject.toml    quant-toolkit-finance  -> src/quant_toolkit/finance/   deps: common (later infra)
packages/infra/pyproject.toml      quant-toolkit-infra    -> src/quant_toolkit/infra/     created when it has content
apps/excel/pyproject.toml          quant-toolkit-xl       -> src/quant_toolkit/xl/        deps: finance, xlwings,
                                                                                         openpyxl, pywin32 (win32)
apps/excel/addin/qtk.xlam          template add-in (built in CI: this file + import-to-add-in)
apps/excel/examples/               demo workbook(s)
```

`quant_toolkit/` has **no `__init__.py` in any portion** (PEP 420 implicit namespace; a stray one
hides every other portion — a test in `common` asserts this). The old `quant_toolkit` shim package
is dropped; the public API is `quant_toolkit.finance`.

Inside `apps/excel/src/quant_toolkit/xl/`:

```
_xw.py              xlwings or the no-op shim (unchanged)
cache.py            content-addressed handle cache (unchanged)
marshalling.py      Excel <-> domain coercions (grows: enums, tables, frames)
api/
  dates.py          QTK_DATE_*   -> quant_toolkit.finance.dates
  curves.py         QTK_CURVE_*  -> ...markets.curves, pricing.calibration
  markets.py        QTK_MKT_*    -> ...markets.context
  instruments.py    QTK_INST_*   -> ...instruments.resolution
  calc.py           QTK_CALC_*   -> ...markets.rate_generator, pricing.pricers
  reports.py        QTK_REPORT_* -> ...pricing.results, instruments.priceable
functions.py        the UDFs: one thin try/except per api function, grouped by area
cli.py              qtk-xl: install | launch | demo | doctor | uninstall
tests/              mirrors api/, one test module per area + the test_functions guard
```

Rules of the split.
- **Lockstep versions.** Every portion carries the same version and releases together; `finance`
  pins `quant-toolkit-common==<that version>`, `xl` pins `quant-toolkit-finance` likewise. The
  add-in version equals the `quant-toolkit-xl` version (§4.3).
- **Layering is the dependency graph.** `common` depends on nothing internal; `infra` on `common`;
  `finance` on `common` (+ `infra`); `xl` on `finance`. A portion cannot import what it does not
  declare.
- **Dev flow is editable installs**: `bin/dev_install.sh` runs `pip install -e` on each portion
  (CI does the same). `PYTHONPATH=packages/common/src:packages/finance/src:apps/excel/src` also
  works because namespace portions merge across path entries.
- **One root for tools.** mypy: `namespace_packages = true`, `explicit_package_bases = true`;
  pytest: `--import-mode=importlib`, `testpaths` lists every portion's in-package test tree;
  coverage targets `quant_toolkit`.
- **Not now:** a uv workspace (wants to own the venv; the env is conda). Revisit if conda goes.
- UDF module setting becomes `quant_toolkit.xl.functions`; the xlwings `_xw` shim is unchanged.

## 2. Function catalogue

Phase P1 is the next PR set; P2 depends on core work listed in §5. Every date argument accepts a
real Excel date, an ISO string or a range of either; every function that accepts a range spills a
column of the same length. Enum arguments are strings matched case-insensitively against the enum
member name (`"ModifiedFollowing"`, `"Actual360"`, `"Quarterly"`).

### 2.1 `QTK_DATE_*` — `api/dates.py` → `finance.dates`

| Excel | Library | Arguments | Returns | Phase |
|---|---|---|---|---|
| `QTK_DATE_IsGoodBusinessDay(dates, calendar)` | `is_good_bd` | dates, calendar name | TRUE/FALSE column | P1 |
| `QTK_DATE_AdjustDate(dates, bdc, calendar)` | `adjust_date` | dates, BDC, calendar | dates | P1 |
| `QTK_DATE_AddBusinessDays(dates, days, bdc, calendar)` | `add_business_days` | dates, int, BDC, calendar | dates | P1 |
| `QTK_DATE_AddTerm(dates, term, bdc, calendar)` | `add_term` | dates, `"3M"`, BDC, calendar | dates | P1 |
| `QTK_DATE_SubtractTerm(dates, term, bdc, calendar)` | `subtract_term` | as above | dates | P1 |
| `QTK_DATE_AddFrequency(dates, frequency, bdc, calendar)` | `add_frequency` | dates, Frequency, BDC, calendar | dates | P1 |
| `QTK_DATE_SubtractFrequency(dates, frequency, bdc, calendar)` | `subtract_frequency` | as above | dates | P1 |
| `QTK_DATE_NextImmDate(dates, [frequency])` | `next_imm_date` | dates, Frequency (default Quarterly) | dates | P1 |
| `QTK_DATE_PriorImmDate(dates, [frequency])` | `prior_imm_date` | dates, Frequency | dates | P1 |
| `QTK_DATE_YearFraction(starts, ends, day_count)` | `period_fractions` | two date ranges, DayCountMethod | floats | P1 |
| `QTK_DATE_Schedule(effective, maturity, frequency, day_count, bdc, calendar, [roll], [direction], [payment_delay], [first_regular], [last_regular])` | `build_payment_schedule` | the swap-leg schedule inputs | table: accrual start, accrual end, pay date, year fraction (header row) | P1 |
| `QTK_DATE_Calendars()` | `Calendars().list_calendars()` | — | names column | P1 |
| `QTK_DATE_RegisterCalendar(name, holidays, [week_mask])` | `Calendars().register(StaticCalendar(...))` | name, date range, mask like `"1111100"` | the name | P1 |
| `QTK_DATE_JoinCalendars(name, calendars)` | `cal_a + cal_b` then register | name, range of names | the name | P1 |
| `QTK_DATE_Holidays(calendar, [from], [to])` | `StaticCalendar.holidays` | name, optional window | dates column | P1 |
| `QTK_DATE_IsHoliday(dates, calendar)` | `Calendar.is_holiday` | dates, name | TRUE/FALSE | P1 |

Notes.
- Only `no_holidays` is registered out of the box, so `RegisterCalendar` is the entry point for
  real calendars. Registration lives in the UDF server process: after an Excel restart the
  registering cell recalculates first because every consumer references its output.
- `week_mask` follows `numpy.busdaycalendar` (`"1111100"` = Mon–Fri). Default Mon–Fri.
- `Schedule` is the same generator the instruments use, so what a trader sees is what the pricer accrues.
- Term parsing is the library's (`"5D"`, `"3M"`, `"1Y"`, `"2B"`); no separate parse UDF.

### 2.2 `QTK_CURVE_*` — `api/curves.py` → `finance.markets.curves`, `finance.pricing.calibration`

| Excel | Library | Arguments | Returns | Phase |
|---|---|---|---|---|
| `QTK_CURVE_Build(dates, dfs, [interpolation])` | `ZeroCurve(...)` (today's `build_curve`) | pillar dates, DFs, `CurveInterpolator` name | `Curve::` | P1 |
| `QTK_CURVE_BuildFromZeros(dates, zeros, [interpolation])` | `ZeroCurve` in `CurveSpace.ZeroRate` | pillar dates, cont. zero rates | `Curve::` | P1 |
| `QTK_CURVE_DF(curve, dates)` | `discount_factor` | handle, dates | floats | P1 (exists) |
| `QTK_CURVE_Zero(curve, dates)` | `zero_rate` | handle, dates | floats | P1 (exists) |
| `QTK_CURVE_Forward(curve, dates)` | `instantaneous_forward` | handle, dates | floats | P1 |
| `QTK_CURVE_Nodes(curve)` | `x`, `dfs`, `node_zero_rates` | handle | table: date, DF, zero (header row) | P1 |
| `QTK_CURVE_Shift(curve, bp)` | `curve + ZeroCurve.constant(bp)` (curve algebra) | handle, parallel shift in bp | `Curve::` | P1 |
| `QTK_CURVE_Calibrate(as_of, key, quotes, [interpolation], [solver], [discount_curve])` | `CurveCalibrator(...).calibrate` | 3-column range `type \| tenor-or-contract \| quote`; `GlobalSolver`/`Bootstrapper` | `Curve::` | P2 |
| `QTK_CURVE_CalibrationReport(curve)` | `CalibrationResult` | calibrated handle | table: pillar, type, tenor, quote, implied, residual; converged flag | P2 |

Notes.
- `Calibrate` quote types map onto the existing factories: `Deposit` → `deposit_helper`, `FRA` →
  `fra_helper` (tenor column holds `"3Mx6M"`), `Swap` → `swap_helper`, `Future` →
  `futures_helper` (tenor column holds `"SR3 U26"`, quote is the price), `Spread` → `spread_helper`.
- Single-curve first: `CurveDefinition(key, interpolation, discount_keys=("<CCY>.STDCSA",))`.
  Dual-curve arrives with `discount_curve` taking a handle and the market built from both.
- The calibrated handle is an ordinary `Curve::` so `QTK_MKT_Build` and everything downstream
  are unchanged; the report is cached beside it.

### 2.3 `QTK_MKT_*` — `api/markets.py` → `finance.markets.context`

| Excel | Library | Arguments | Returns | Phase |
|---|---|---|---|---|
| `QTK_MKT_Build(as_of, keys, curves)` | `MarketContext(as_of, {key: curve, ...}).freeze()` | as-of, range of curve keys, range of `Curve::` handles (same length) | `Market::` | P1 |
| `QTK_MKT_With(market, key, curve)` | `with_curve` | market, key, `Curve::` | `Market::` | P1 |
| `QTK_MKT_Bump(market, key, bp)` | `bump` | market, key, parallel bp | `Market::` | P1 |
| `QTK_MKT_Keys(market)` | `groups` / registered keys | market | table: key, curve handle | P1 |
| `QTK_MKT_Build(as_of, dict)` (overload) | same | a `Dict::` handle of key → curve, plus `fixings` | `Market::` | P2 |

Notes.
- This replaces today's one-curve `qMarket`. Aliases are repeated handles: SOFR under both
  `USD.SOFR.1D` and `USD.STDCSA` is two rows pointing at one `Curve::`.
- Every curve's origin must equal `as_of` (today's check, kept).
- Keys follow the grammar in `finance/conventions/keys.py`; an unknown projection family is a `#QERR` naming the valid families.

### 2.4 `QTK_INST_*` — `api/instruments.py` → `finance.instruments.resolution`

| Excel | Library | Arguments | Returns | Phase |
|---|---|---|---|---|
| `QTK_INST_Swap(notional, rate_index, fixed_rate, tenor, as_of, [ccy], [funding_id], [fixed_frequency], [float_frequency])` | `Swap.fixed_float_swap` | sign on notional: + receive fixed, − pay fixed | `Inst::` | P1 (exists as `qSwap`) |
| `QTK_INST_Deposit(rate, tenor, as_of, [rate_index], [ccy], [funding_id], [notional], [day_count])` | `Deposit.spot_deposit` | | `Inst::` | P1 |
| `QTK_INST_Fra(rate, start, end, as_of, [rate_index], [ccy], [funding_id], [notional], [day_count])` | `Fra.forward_starting` | `start`/`end` are terms (`"3M"`, `"6M"`) | `Inst::` | P1 |
| `QTK_INST_Future(contract, month, as_of, [contracts], [entry_price], [rate_index], [ccy], [funding_id], [convexity])` | `SofrFuture.from_contract_month` | `"SR3"`, `"U26"` | `Inst::` | P1 |
| `QTK_INST_Program(instruments)` | pricer `compile` per type | range of `Inst::` | `Program::` | P1 (exists as `qCompileProgram`) |
| `QTK_INST_FromDict(dict)` | `common.object.load` | a `Dict::` document with `type` | `Inst::` | P2 |
| `QTK_INST_ToDict(inst)` | `dump` | `Inst::` | table of the document, dotted keys | P2 |

Notes.
- One handle kind `Inst::` for every product (today's `Swap::` is renamed). The cache stores the
  resolved object; `Program` groups by type and compiles each with its registered pricer.
- Swaption stays out of P1; it needs vols on the market first.

### 2.5 `QTK_CALC_*` — `api/calc.py` → `finance.markets.rate_generator`, `finance.pricing`

| Excel | Library | Arguments | Returns | Phase |
|---|---|---|---|---|
| `QTK_CALC_Price(program_or_insts, market)` | `PricingProgram.price` | `Program::` or range of `Inst::`, `Market::` | PV column (one per instrument) | P1 (exists as `qPrice`) |
| `QTK_CALC_LegPVs(program_or_insts, market)` | `PricingResult.leg_pv` | as above | table: instrument, leg, PV | P1 |
| `QTK_CALC_SimpleRate(market, curve_key, starts, ends, [day_count])` | `RateGenerator.simple_rate` | | floats | P1 |
| `QTK_CALC_ForwardRate(market, curve_key, starts, ends)` | `continuous_forward_rate` | | floats | P1 |
| `QTK_CALC_Spread(market, curve_key, over_key, starts, ends, [day_count])` | `spread` | | floats | P1 |
| `QTK_CALC_ParRate(market, inst)` | `par_swap_rate` | `Market::`, swap `Inst::` | float | P1 |
| `QTK_CALC_CompoundedRate(market, curve_key, start, end, [calendar])` | `compounded_rate` over a daily grid built by `build_payment_schedule(build_observations=True)` | | float | P2 |
| `QTK_CALC_AveragedRate(...)` | `averaged_rate` | as above | float | P2 |
| `QTK_CALC_Risk(program_or_insts, market, curve_key)` | `program.risk(...)` ladder | | table: pillar, DV01 | P2 |

Notes.
- `Price` accepting a range of `Inst::` directly compiles on first call and caches the program
  under the hash of the handle set, so single-instrument pricing never needs a visible `Program::`.
  `Program` stays for portfolios where the trader wants the compile step explicit.
- Compounded and averaged rates need an observation grid; they land with the schedule work.

### 2.6 `QTK_REPORT_*` — `api/reports.py` → `finance.pricing.results`, `Priceable.requests`

| Excel | Library | Arguments | Returns | Phase |
|---|---|---|---|---|
| `QTK_REPORT_Cashflows(program_or_insts, market)` | `PricingResult.cashflows` (`CashflowReport`) | | table: instrument, leg, pay date, notional, rate, year fraction, DF, PV (header row) | P1 |
| `QTK_REPORT_Inspect(handle)` | handle's `dump` or `repr` | any handle | two-column table, dotted keys | P1 |
| `QTK_REPORT_Schedule(inst)` | new `schedule` request on `Priceable` | `Inst::` | table: leg, accrual start, accrual end, pay date, year fraction, notional, rate kind, rate/spread | P2 |
| `QTK_REPORT_Accrued(inst, market)` | new `accrued` request | | table per leg: accrual start, as-of, fraction, rate, accrued | P2 |
| `QTK_REPORT(inst, request, [market])` | `inst(market, requests=[...])` | generic form over the same vocabulary | table | P2 |

Notes.
- P1 `Cashflows` ships without accrual dates (the report carries pay dates today). P2 threads
  accrual starts and ends through `KernelInputs` so cashflows become a superset of the schedule.
- Reports return columnar numpy in the core; `to_frame()` is added on the report classes and the
  Excel layer marshals a frame into a 2-D array with a header row.

## 3. Marshalling rules (`marshalling.py`)

- **Dates in:** Excel serial (1900 system), ISO string, `datetime`, or a range of them → `Date` or
  `DateArray`. **Dates out:** `datetime` so Excel formats them as dates.
- **Enums:** `enum_from_name(DayCountMethod, "actual360")` is case-insensitive on the member name;
  unknown names raise with the valid list.
- **Terms and keys:** passed through as strings; the library parses and validates.
- **Ranges:** a 1-D range (row or column) is a vector; a 2-D range with a header row is a table.
  Vector outputs spill as a column. Tables spill with a header row.
- **Handles:** `kind::digest`; a range of handles is a list. Mixed kinds are a `#QERR` naming the
  offending cell value.
- **Frames:** `frame_to_table(df)` → `[[headers], rows...]`, dates as `datetime`, NaN as `""`.

## 4. The add-in and the CLI

### 4.1 `qtk.xlam`

Built from the xlwings custom add-in template so it embeds the xlwings VBA (no separate xlwings
add-in on trader machines). It carries: the `QTK_*` wrappers (imported to the add-in in CI),
a `QTK` ribbon tab (Inspect object, Restart server, Clear cache, About with versions), a
right-click "QTK: Inspect object" entry installed on open, and a version constant equal to the
package version it shipped with. Config comes from `%USERPROFILE%\.qtk\qtk.conf` (interpreter,
UDF module, Use UDF Server on); the shipped file contains no machine paths.

### 4.2 `qtk-xl` (console script in `cli.py`)

| Command | What it does |
|---|---|
| `qtk-xl install` | copies the bundled `.xlam` to `%APPDATA%\Microsoft\Excel\XLSTART\`, writes `qtk.conf` pointing at the interpreter it runs under, verifies by evaluating one UDF through COM. Refuses while Excel is running. Idempotent. |
| `qtk-xl launch [workbook]` | starts Excel with the add-in and the workbook (the demo when omitted). |
| `qtk-xl demo` | writes the demo workbook into the current directory and opens it. |
| `qtk-xl doctor` | prints package, xlwings, add-in and interpreter versions, trust setting, whether the add-in is in XLSTART and whether the UDF server starts. |
| `qtk-xl uninstall` | removes the add-in and config. |

Requirements the README states up front: Windows, desktop Excel, a native Windows Python 3.11+
(not WSL), a venv recommended, then `pip install quant-toolkit[excel]` and `qtk-xl install`.

### 4.3 Version rule

`.xlam` version == `quant-toolkit-xl` version. On open the add-in asks the UDF server for the package
version and shows one line on mismatch: *"QTK add-in 0.3, package 0.4: run qtk-xl install"*.
Logic updates are `pip install -U` alone.

## 5. Core work the P2 functions depend on

- `schedule` and `accrued` requests on `Priceable`; `to_frame()` on `CashflowReport` and the new
  report classes; accrual dates in `KernelInputs` (ignored by kernels).
- Accrued for compounded legs needs realized fixings on the report path (`HistoricalFixings`).
- Dict/List handles (`QTK_Dict(keys, values)`, `QTK_List(range)`) backed by the content-addressed
  cache, used by `QTK_MKT_Build` and `QTK_INST_FromDict`.

## 6. Order and rough effort

| PR | Content | Effort |
|---|---|---|
| X0 | namespace restructure: `packages/common`, `packages/finance`, `apps/excel` as portions of `quant_toolkit`; imports rewritten (`from finance.` → `from quant_toolkit.finance.`, ~744 lines / 168 files); meta-package root toml; CMake install path for `_core`; docs + CLAUDE.md; Windows venv reinstalled. Lands alone, nothing long-lived in flight | 1 day |
| X1 | `api/` package, rename to `QTK_*`, `QTK_DATE_*` incl. calendars and schedule, marshalling for enums/tables, tests per area, README catalogue | 1 day |
| X2 | `QTK_INST_*` deposit/FRA/future, `QTK_MKT_*` multi-curve, `QTK_CURVE_*` extras, `QTK_CALC_*` P1 | 1 day |
| X3 | `QTK_REPORT_Cashflows`, `QTK_REPORT_Inspect`, `QTK_CALC_LegPVs` | ½ day |
| X4 | `qtk.xlam` template, ribbon, context-menu inspector, `cli.py`, README install/launch, demo workbook regenerated with `QTK_*` names | 1–1½ days |
| X5 | Calibration UDFs and report | ½ day (core exists) |
| X6 | Schedule/accrued requests, `to_frame`, dict handles, generic `QTK_REPORT` | 1–2 days |

## 7. Open points for review

1. Calendars under `DATE` (as planned) or a seventh `CAL` area.
2. `Inst::` as the single instrument handle kind versus per-product kinds in the handle text.
3. Hide `Program::` for single instruments (`QTK_CALC_Price` taking `Inst::` ranges directly), keep
   it for portfolios. Planned as yes.
4. Whether `QTK_DATE_RegisterCalendar` should also accept a CSV path for holiday files kept in the
   repo's `data/` folder.
