# quant_toolkit_xl — Excel add-in for quant-toolkit

A thin [xlwings](https://www.xlwings.org/) adapter that exposes `quant-toolkit`'s pricing
engine to Excel as worksheet functions. Curves, instruments, and pricing programs are too
big to live in a cell, so each builder returns an opaque **handle** string that other
functions accept — build once, reference many.

```
=qBuildCurve(dates, dfs)          -> "Curve::8f2a1c"
=qSwap(notional, index, rate, …)  -> "Swap::9dfaf9"
=qCompileProgram(swapHandles)     -> "Program::bc9d67"
=qMarket(asOf, ccy, index, curve) -> "Market::a0a6c8"
=qPrice(program, market)          -> per-instrument PVs
```

---

## Requirements / OS dependency

> **The UDFs run on Windows only.** xlwings exposes Python functions as Excel worksheet
> functions (UDFs) **only on Windows** — this needs the COM automation server that ships with
> desktop Excel. On macOS xlwings supports macros but **not** UDFs, and on Excel-on-the-web
> there is no UDF support at all.

| | Requirement |
|---|---|
| OS | **Windows 10/11** |
| Excel | Microsoft Excel **desktop** (Microsoft 365 or 2016+) |
| Python | **3.11+**, installed **natively on Windows** (not WSL) |
| Package | `quant-toolkit[excel]` (pulls in `xlwings`) |

**Why native-Windows Python?** Excel talks to the Python interpreter over COM on the local
machine. A WSL interpreter can't serve UDFs to Windows Excel. Develop in WSL if you like, but
the interpreter Excel uses must be a Windows install with this package on it.

---

## 1. Install (on Windows)

Open **PowerShell** (or the Anaconda Prompt) on the Windows side and, from the repo root:

```powershell
# a dedicated env is recommended
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# editable install with the Excel extra (xlwings)
pip install -e ".[excel]"
```

Verify the package imports under that interpreter:

```powershell
python -c "import quant_toolkit_xl, xlwings; print('ok', xlwings.__version__)"
```

## 2. Install the xlwings Excel add-in

This puts the xlwings tab on the Excel ribbon (it must match the installed xlwings version):

```powershell
xlwings addin install
```

## 3. Allow Excel to import UDFs (one-time Excel setting)

In Excel: **File → Options → Trust Center → Trust Center Settings → Macro Settings** →
tick **“Trust access to the VBA project object model.”** UDF import fails without this.

---

## 4. Wire it into Excel and see the functions

1. Open a **new blank workbook** and go to the **xlwings** ribbon tab.
2. **Interpreter** — set it to the `python.exe` of the env from step 1, e.g.
   `...\rates-lib\.venv\Scripts\python.exe`. (Leave blank only if that env's Python is first
   on `PATH`.)
3. **UDF Modules** — enter:
   ```
   quant_toolkit_xl.functions
   ```
   (The package is pip-installed, so no `PYTHONPATH` entry is needed — it's already importable
   by that interpreter.)
4. Click **Import Functions** on the xlwings ribbon. The `q*` functions are now live; type
   `=q` in a cell and they'll show in autocomplete.

> Re-import after changing a function **signature**. Because the interpreter is editable
> (`-e`), pure logic changes are picked up on the next recalc without re-importing.

---

## 5. Worked example

Lay out a curve and price a 5Y SOFR swap end-to-end:

| | A | B |
|---|---|---|
| 1 | **Date** | **DF** |
| 2 | 2026-06-01 | 1.000000 |
| 3 | 2026-12-01 | 0.980199 |
| 4 | 2027-06-01 | 0.960789 |
| 5 | 2031-06-01 | 0.818731 |
| 6 | 2036-06-01 | 0.670320 |

```excel
D1:  =qBuildCurve(A2:A6, B2:B6)              ' -> Curve handle  (default LogLinearDF)
D2:  =qSwap(100000000, "SOFR", 0.041, "5Y", A2)   ' -> Swap handle (sign on notional:
                                                  '    + = receive fixed, − = pay fixed)
D3:  =qCompileProgram(D2)                    ' -> Program handle (D2 may be a range of swaps)
D4:  =qMarket(A2, "USD", "SOFR", D1)         ' -> Market handle (binds the curve under USD.SOFR)
D5:  =qPrice(D3, D4)                         ' -> per-instrument PV(s); spills down

' curve queries (dates may be ranges; results spill)
F2:  =qDiscountFactor(D1, A4)                ' DF to 2027-06-01
F3:  =qZeroRate(D1, A5)                      ' cont.-comp. zero rate to 2031-06-01
```

`interpolation` (5th positional arg of `qBuildCurve`) accepts: `LogLinearDF` (default),
`LogCubicDF`, `RateLinear`, `RateQuadratic`, `RateCubic`. Dates can be real Excel dates or
ISO strings; numbers are read as Excel 1900-system serials.

---

## Notes

- **Errors show in the cell.** A bad input returns a readable `#QERR: …` string instead of
  Excel's opaque `#VALUE!`.
- **Handles are content-addressed.** Identical inputs always produce the same handle, so
  recalculation is idempotent and doesn't leak objects. A `#QERR: handle … not found` means
  the upstream cell that builds it needs recalculating.
- **WSL development.** On Linux/WSL `xlwings` imports but can't drive Excel; a built-in shim
  (`_xw.py`) lets the package import and keeps the pure logic in `api.py` fully unit-testable
  without Excel. The `q*` UDFs only do anything real once loaded by Windows Excel.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `q*` functions not found after Import | Confirm **UDF Modules** = `quant_toolkit_xl.functions` and the **Interpreter** points at the env where you ran `pip install`. |
| Import Functions errors / nothing happens | Enable **Trust access to the VBA project object model** (step 3). |
| `No module named quant_toolkit_xl` | You installed into a different Python than the configured Interpreter. Re-check both point at the same env. |
| Ribbon tab missing | Re-run `xlwings addin install`; ensure the add-in version matches `xlwings.__version__`. |
