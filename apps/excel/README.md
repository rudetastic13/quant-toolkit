# apps/excel — Excel add-in for quant-toolkit (xlOil)

`quant-toolkit-xl` exposes the library to Excel as worksheet functions through
[xlOil](https://xloil.readthedocs.io/). An `.xll` add-in embeds Python inside `EXCEL.EXE`, so a
function call is an in-process call: no COM, no VBA, no Trust Center change. The trade-off is that a
hard crash in Python takes Excel down with it.

Functions are registered as `QuantToolkit.<Group>.<Name>` with CamelCase arguments:

| Group | Module | Functions |
|---|---|---|
| `QuantToolkit.Dates` | `xl/dates.py` | `AddBusinessDays`, `AdjustDate`, `AddTerm`, `SubtractTerm`, `IsBusinessDay`, `NextImmDate`, `PriorImmDate`, `UnadjustedSchedule`, `AdjustedSchedule` |
| `QuantToolkit.IRProducts` | `xl/products.py` | `FixedFloatSwap`, `OvernightIndexSwap` |
| `QuantToolkit.Solver` | `xl/solver.py` | `DepositQuote`, `FraQuote`, `SwapQuote`, `SpreadQuote`, `FuturesQuote`, `Calibrate`, `CalibrationReport` |
| `QuantToolkit.Curves` | `xl/curves.py` | `DiscountFactor`, `ZeroRate`, `ForwardRate`, `Keys` |
| `QuantToolkit.Pricing` | `xl/pricing.py` | `Portfolio`, `Calculate`, `PresentValue`, `DV01` |
| `QuantToolkit.Risk` | `xl/risk.py` | `RiskTemplate`, `RiskReport` |
| `QuantToolkit.Utilities` | `xl/utilities.py` | `ArrayToList`, `ArrayToDict`, `TagDate`, `Version` |

Each module is a set of functions calling the library directly, registered through one decorator:
`@xl_func("Dates")` on `AddTerm` registers `QuantToolkit.Dates.AddTerm`. The shared helpers live in
`xl/core/`: `register.py` (the decorator), `handles.py` (the handle store), `convert.py` (cell-value
parsing) and `book.py` (the compiled portfolio and its measures).

**Windows + desktop Excel only**, with a native Windows Python (not WSL) whose bitness matches Excel's.

| Path | What |
|---|---|
| `src/quant_toolkit/xl/` | the worksheet function modules above |
| `src/quant_toolkit/xl_installer/` | `quant_toolkit_xl_installer` CLI (prototype) and the `quant_toolkit_xl` launcher |
| `examples/sofr_swap_demo.xlsx` | sheets `Dates`, `Curve` (quotes -> calibration -> curve), `Swaps` (build and price), `Risk` (measures, risk template, risk report), `Bench` (250 swaps) |
| `examples/bench_swaps.py` | times cell-by-cell swap construction against plain Python |

---

## 1. Install quant-toolkit first

The add-in depends on the library portions, so set those up in a native Windows env first.
From the repo root in a PowerShell opened **after** `conda init powershell`:

```powershell
. bin/powershell/use_env.ps1 pricing                # prompt shows (pricing-YYYYMM)
python -c "import sys; print(sys.executable)"       # must be the env's python.exe
```

(Git Bash: `source bin/bash/use_env.sh pricing`.)

## 2. Install the Excel app

In the **same** activated env:

```powershell
pip install -e apps/excel
python -c "import xloil, quant_toolkit.xl.dates; print('ok')"
```

**Bitness must match.** 64-bit Excel can only load a 64-bit Python (`win_amd64` wheels), and 32-bit Excel
needs a 32-bit one. Check under **File → Account → About Excel**. Miniforge is 64-bit.

## 3. Run the installer (prototype)

Close Excel, then run this from the activated env:

```powershell
quant_toolkit_xl_installer check              # prerequisites + current state; changes nothing
quant_toolkit_xl_installer install --dry-run  # preview the changes
quant_toolkit_xl_installer install
```

`install` does three things:
1. **Checks:** Windows, Excel is registered, xlOil is importable and its add-in file is present, Python
   bitness, and whether Excel is running.
2. **Points `%APPDATA%\xlOil\xlOil.ini` at this env** (created from xlOil's template if missing). Re-run it
   after switching envs.
3. **Tags our modules into the ini:** sets the installer's entries in `[xlOil_Python] LoadModules` (default: the
   modules listed in `MODULES` in `xl/__init__.py`, override with `--modules a.b c.d`). A marker comment above that line
   records what it manages, so a re-run replaces its old set and leaves entries you added by hand alone. It
   checks that each module imports first and backs up the ini to `xlOil.ini.bak`.

**The add-in is on demand by default.** Nothing is put in Excel's startup folder, so Excel starts without xlOil
and without the Python import cost. A copy left in `XLSTART` by an earlier install is removed. Load it for one
session with the launcher:

```powershell
quant_toolkit_xl                                  # Excel with the add-in loaded
quant_toolkit_xl apps\excel\examples\sofr_swap_demo.xlsx
```

It starts Excel the ordinary way (so your other add-ins load), hands it `xlOil.xll`, waits until the
`QuantToolkit.*` functions are registered, then opens the workbooks. If Excel is already running the add-in is
loaded into that session. Close Excel and it is gone.

- **Order matters.** A workbook opened before the functions are registered shows `#NAME?`, and Excel keeps that
  until the formula is re-entered: recalculating does not clear it. The launcher handles the order. By hand,
  open `<env>\share\xloil\xlOil.xll` with **File → Open** first (the path is printed by `check`), then the workbook.
- **Security prompt.** Excel may ask before loading an add-in from outside a trusted location.
  `install --trust` adds the add-in's folder to your trusted locations (per user; `uninstall` removes it).
- **Persistent instead:** `install --startup` runs `xloil install`, which copies `xlOil.xll` into `XLSTART` so
  every Excel session loads it (pass `--create-xlstart` if Excel has never been started).

An **xlOil** ribbon tab means the add-in loaded, and `=QuantToolkit.Utilities.Version()` in a cell means our
functions loaded.

`quant_toolkit_xl_installer uninstall` removes only the modules its marker recorded and its trusted location,
then runs `xloil remove` (xlOil leaves the ini in place). It also takes `--dry-run`.

---

## 4. Conventions

- **Dates** go in and out as Excel dates (serial numbers), so format result cells as dates.
- **Date or term.** `Effective` is a date, or a term from `AsOf` (`"2BD"`, `"1Y"`). `Maturity` is a date, or a
  term from the effective date (`"5Y"`). `AsOf` is only read when `Effective` is a term, and is today if left
  out; point it at a `=TODAY()` cell if the sheet should roll forward on its own.
- **Optional arguments** left out, or pointed at an empty cell, take the rate index's convention.
- **Enums by name:** `"ModifiedFollowing"` (or `"MF"`), `"Quarterly"`, `"Thirty360"`, `"EOM"`, `"Backward"`.
- **Calendars** are registry names in any case, joined with `+`: `US_FED`, `US_NYSE`, `US_SIFMA`,
  `US_CME_TREASURY_FUTURES`, `NO_HOLIDAYS`.
- **Notional sign:** positive receives fixed (receiver), negative pays fixed (payer).
- **Ranges:** the scalar date functions broadcast their arguments, so a column of dates against a row of terms
  gives a grid.

### Handles

Curves, instruments and containers are too big for a cell, so a function returns a **handle** string that other
functions accept: `Swap;;;74086109689f`, `SwapQuote;;;…`, `Calibration;;;…`, `Portfolio;;;…`, `RiskTemplate;;;…`, `List;;;…`, `Dict;;;…`.

- Handles are content-addressed: the same contract always gives the same handle, so recalculation doesn't leak
  objects. `handle … not found` means the cell that builds it needs recalculating.
- A range of handles can be passed wherever a list is expected, or packed with `ArrayToList`. An element that is
  itself a handle is replaced by its object, so `ArrayToList` over `ArrayToDict` cells is a list of dicts.
- A date inside a list or dict must be wrapped in `TagDate` (`Date;;;2026-06-01`), or it arrives as a number.

## 5. Worked example

Open `examples/sofr_swap_demo.xlsx`, or build it by hand. With the as-of date in `B1`:

```excel
' quotes -> one handle each
E5:   =QuantToolkit.Solver.DepositQuote(0.043,  "USD.SOFR.1D", "2BD", "3M", $B$1)
E6:   =QuantToolkit.Solver.FraQuote(0.0418,     "USD.SOFR.1D", "6M",  "3M", $B$1)
E7:   =QuantToolkit.Solver.SwapQuote(0.038,     "USD.SOFR.1D", "2BD", "5Y", $B$1)
E8:   =QuantToolkit.Solver.SwapQuote(0.0392,    "USD.SOFR.1D", "2BD", "10Y", $B$1)

' calibrate -> a Calibration handle (the curve and the market holding it)
H1:   =QuantToolkit.Solver.Calibrate(E5:E8, "USD.SOFR.1D", $B$1)
H3:   =QuantToolkit.Solver.CalibrationReport(H1)          ' pillar, quote, implied, error, DF, zero rate

' read the curve
K1:   =QuantToolkit.Curves.DiscountFactor(H1, DATE(2031,6,3))
K2:   =QuantToolkit.Curves.ZeroRate(H1, DATE(2031,6,3))
K3:   =QuantToolkit.Curves.ForwardRate(H1, DATE(2027,6,1), DATE(2027,9,1))

' build and price a swap
M1:   =QuantToolkit.IRProducts.OvernightIndexSwap(-100000000, "USD.SOFR.1D", 0.03, "2BD", "5Y", $B$1)
M2:   =QuantToolkit.Pricing.PresentValue(M1, H1)
```

`Calibrate(Instruments, CurveKey, AsOf, Interpolation, Solver, DiscountKeys, Market, SpreadOver)`:

- `Interpolation`: `LogLinearDF` (default), `LogCubicDF`, `RateLinear`, `RateQuadratic`, `RateCubic`.
- `Solver`: `Global` (default) or `Bootstrap` (local interpolations only).
- `DiscountKeys`: if left out, a first curve also serves `CCY.STDCSA`, the key the swaps discount on.
- `Market`: an earlier `Calibration` handle to build a second curve on top of.
- `SpreadOver`: solve the curve as a spread over that key (used with `SpreadQuote`).

The curve functions take an optional `CurveKey`, so the result of a second calibration can still read the
first curve: `=QuantToolkit.Curves.DiscountFactor(SecondCalibration, Date, "USD.SOFR.1D")`.

## 6. Measures and risk

`Portfolio` compiles a set of swaps once; `Calculate` then asks it for any list of measures against a market
and returns a table with a header row and one row per position:

```excel
P1:   =QuantToolkit.Pricing.Portfolio(M1:M8)                      ' Swap handles -> one Portfolio handle
P3:   =QuantToolkit.Pricing.Calculate(P1, H1, {"PV","DV01"})      ' or a range of measure names
```

| Measure | Columns |
|---|---|
| `PV` | present value |
| `LegPV` | `ReceiveLegPV`, `PayLegPV` |
| `DV01` | PV change for +1bp on every pillar of `CurveKey` (the calibrated curve if omitted) |
| `KeyRate01` | one column per curve pillar |
| `Quote01` | one column per `RiskTemplate` quote: PV change for +1bp on that quote |
| `Unexplained01` | pillar risk the template does not explain, summed |

`PresentValue` and `DV01` are the one-measure shortcuts and return a plain column; both take a `Portfolio`
handle or the swap handles directly. **Sign:** DV01 is the PV change for rates +1bp, so a receiver is negative.

A **risk template** is the list of quotes you want risk reported against. It is a view: it changes no PV and
does not recalibrate, so one curve can be read through several templates.

```excel
' quotes to report against (the rate is not used, so 0 will do), with labels in R5:R8
S5:   =QuantToolkit.Solver.SwapQuote(0, "USD.SOFR.1D", "2BD", "2Y", $B$1)
S9:   =QuantToolkit.Risk.RiskTemplate(S5:S8, R5:R8)
U1:   =QuantToolkit.Risk.RiskReport(P1, H1, S9)
```

`RiskReport` returns one row per template quote (label, pillar date, kind, the rate the curve implies, the
portfolio's DV01 to that quote, and the notional of that quote's instrument that flattens it), then three
reconciliation rows: all quotes +1bp, all pillars +1bp, and the largest pillar risk left unexplained. With no
template it reports against the quotes the curve was calibrated to. `RiskTemplate` also takes a `Calibration`
handle in place of quotes, and a `Basis` (`Tent`, `LeastSquares`, `Scheme`) for how risk between template
nodes is attributed. A template coarser than the book leaves unexplained risk; a template finer than the
curve's pillars is rejected.

## 7. Benchmark

With the demo workbook open in Excel:

```powershell
python apps/excel/examples/bench_swaps.py --rows 250 --repeats 5
```

It recalculates `Bench!D` (one `OvernightIndexSwap` per cell) and compares that against calling the library
directly in Python, so the ratio is the cost of going through Excel.

---

## Notes

- **Errors show in the cell** as the Python exception text.
- **Names are a contract.** Saved workbooks bind to the registered names and argument orders.
- **Arrays in the demo workbook** are entered as fixed-size array formulas, because the file is generated
  outside Excel. Typed into Excel 365 the same functions spill on their own.
- **No tests yet** for the worksheet layer: it is still a prototype.

## Troubleshooting

| Symptom | Fix |
|---|---|
| No xlOil ribbon tab | Excel was open during `install`, or the add-in is disabled: **File → Options → Add-ins → Manage: Excel Add-ins**, tick xlOil. |
| xlOil loads but can't start Python / find a DLL | `xlOil.ini` points at a different env than the one with the packages, or the bitness doesn't match. Re-run `quant_toolkit_xl_installer install` from the right activated env. |
| `No module named quant_toolkit` in `%APPDATA%\xlOil\xlOil.log` | Same cause: installed into a different env than the one in `xlOil.ini`. Check `python -c "import sys; print(sys.executable)"`, then re-run `install`. |
| `pip`/`python` resolve to the wrong interpreter | The PowerShell window predates `conda init powershell`, so the conda hook never loaded. Open a new window. |
| `#NAME?` in every `QuantToolkit.*` cell | The workbook was opened before the add-in (open it with `quant_toolkit_xl`, or re-enter the formulas: replace `=` with `=`). Otherwise the modules aren't in `LoadModules`, or one failed to import: check `xlOil.log`, then `quant_toolkit_xl_installer check`. |
| Functions missing after editing Python | Changes to a loaded module need a reload: use the xlOil ribbon's reload, or restart Excel. |
