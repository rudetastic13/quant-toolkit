# apps/excel — Excel add-in for quant-toolkit (xlOil)

`quant-toolkit-xl` exposes the pricing layer to Excel as worksheet functions through
[xlOil](https://xloil.readthedocs.io/). An `.xll` add-in embeds Python inside `EXCEL.EXE`, so a
function call is an in-process call: no COM, no VBA, no Trust Center change. Building 1k swaps
cell by cell is near-instant (it took ~2 min on the old xlwings add-in). The trade-off is that a
hard crash in Python takes Excel down with it.

Curves, instruments and pricing programs are too big to live in a cell, so each builder returns an
opaque **handle** string that other functions accept. Build once, reference many times:

```
=qBuildCurve(dates, dfs)          -> "Curve::8f2a1c"
=qSwap(notional, index, rate, …)  -> "Swap::9dfaf9"
=qCompileProgram(swapHandles)     -> "Program::bc9d67"
=qMarket(asOf, ccy, index, curve) -> "Market::a0a6c8"
=qPrice(program, market)          -> per-instrument PVs (a column)
```

**Windows + desktop Excel only**, with a native Windows Python (not WSL) whose bitness matches Excel's.

| Path | What |
|---|---|
| `src/quant_toolkit/xl/` | `api` (pure logic), `cache` (handles), `marshalling` (Excel values -> domain types), `functions` (the `q*` worksheet functions, the only module that imports `xloil`) |
| `src/quant_toolkit/xl_installer/` | `quant_toolkit_xl_installer` CLI (prototype) |
| `examples/sofr_swap_demo.xlsx` | curve -> swap -> program -> market -> PV, a tenor strip, and a 250-row `Bench` sheet |
| `examples/bench_swaps.py` | times cell-by-cell `qSwap` against plain Python |

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
python -c "import xloil, quant_toolkit.xl.functions; print('ok')"
```

**Bitness must match.** 64-bit Excel can only load a 64-bit Python (`win_amd64` wheels), and 32-bit Excel
needs a 32-bit one. Check under **File → Account → About Excel**. Miniforge is 64-bit.

## 3. Run the installer (prototype)

Close Excel, then run this from the activated env:

```powershell
quant_toolkit_xl_installer check              # prerequisites + current state; changes nothing
quant_toolkit_xl_installer install --dry-run  # preview the ini diff
quant_toolkit_xl_installer install
```

`install` does three things:
1. **Checks:** Windows, `%APPDATA%\Microsoft\Excel\XLSTART` exists (pass `--create-xlstart` if Excel has never
   been started), Excel is registered, xlOil is importable, Python bitness, and Excel isn't running.
2. **Runs `xloil install` with this interpreter:** copies `xlOil.xll` into `XLSTART` so it loads whenever Excel
   starts, and points `%APPDATA%\xlOil\xlOil.ini` at this env. Re-run it after switching envs.
3. **Tags our module into the ini:** sets the installer's entries in `[xlOil_Python] LoadModules` (default
   `quant_toolkit.xl.functions`, override with `--modules a.b c.d`). A marker comment above that line records
   what it manages, so a re-run replaces its old set and leaves entries you added by hand alone. It checks that
   each module imports first and backs up the ini to `xlOil.ini.bak`.

Start Excel. An **xlOil** ribbon tab means the add-in loaded, and `=qVersion()` in a cell means our functions
loaded. Every `@xloil.func` in a listed module becomes a worksheet function, with no import step.

`quant_toolkit_xl_installer uninstall` removes only the modules its marker recorded, then runs `xloil remove`
(xlOil leaves the ini in place). It also takes `--dry-run`.

---

## 4. Worked example

Open `examples/sofr_swap_demo.xlsx`, or lay out a curve and price a 5Y SOFR swap end to end:

| | A | B |
|---|---|---|
| 1 | **Date** | **DF** |
| 2 | 2026-06-01 | 1.000000 |
| 3 | 2026-12-01 | 0.980199 |
| 4 | 2027-06-01 | 0.960789 |
| 5 | 2031-06-01 | 0.818731 |
| 6 | 2036-06-01 | 0.670320 |

```excel
D1:  =qBuildCurve(A2:A6, B2:B6)                          ' Curve handle (default LogLinearDF)
D2:  =qSwap(100000000, "USD.SOFR.1D", 0.041, "5Y", A2)   ' Swap handle (+ notional = receive fixed)
D3:  =qCompileProgram(D2)                                ' Program handle (D2 may be a range of swaps)
D4:  =qMarket(A2, "USD", "USD.SOFR.1D", D1)              ' Market: curve under USD.SOFR.1D and USD.STDCSA
D5:  =qPrice(D3, D4)                                     ' per-instrument PVs, one row per swap

F2:  =qDiscountFactor(D1, A4)                            ' DF to 2027-06-01
F3:  =qZeroRate(D1, A5)                                  ' cont.-comp. zero rate to 2031-06-01
```

`interpolation` (3rd argument of `qBuildCurve`) accepts `LogLinearDF` (default), `LogCubicDF`, `RateLinear`,
`RateQuadratic` and `RateCubic`. Dates can be real Excel dates or ISO strings; numbers are read as Excel
1900-system serials.

## 5. Benchmark

With the demo workbook open in Excel:

```powershell
python apps/excel/examples/bench_swaps.py --rows 250 --repeats 5
```

It recalculates `Bench!D` (one `qSwap` per cell) and compares that against calling the same logic directly
in Python, so the ratio is the cost of going through Excel.

---

## Notes

- **Errors show in the cell.** A bad input returns a readable `#QERR: …` string instead of Excel's opaque
  `#VALUE!`.
- **Handles are content-addressed.** Identical inputs always produce the same handle, so recalculation is
  idempotent and doesn't leak objects. `#QERR: handle … not found` means the upstream cell that builds it
  needs recalculating.
- **Names are a contract.** xlOil registers the Python function name verbatim, and saved workbooks bind to it.
- **Testing without Excel.** `api`, `cache` and `marshalling` don't import `xloil`, so they're unit-testable
  anywhere. `xloil`'s decorator leaves `functions` callable outside Excel too; those tests skip where xloil
  isn't installed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| No xlOil ribbon tab | Excel was open during `install`, or the add-in is disabled: **File → Options → Add-ins → Manage: Excel Add-ins**, tick xlOil. |
| xlOil loads but can't start Python / find a DLL | `xlOil.ini` points at a different env than the one with the packages, or the bitness doesn't match. Re-run `quant_toolkit_xl_installer install` from the right activated env. |
| `No module named quant_toolkit` in `%APPDATA%\xlOil\xlOil.log` | Same cause: installed into a different env than the one in `xlOil.ini`. Check `python -c "import sys; print(sys.executable)"`, then re-run `install`. |
| `pip`/`python` resolve to the wrong interpreter | The PowerShell window predates `conda init powershell`, so the conda hook never loaded. Open a new window. |
| `#NAME?` in every `q*` cell | The module isn't in `LoadModules`, or failed to import: check `xlOil.log`, then `quant_toolkit_xl_installer check`. |
| Functions missing after editing Python | Changes to a loaded module need a reload: use the xlOil ribbon's reload, or restart Excel. |
