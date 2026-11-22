"""Time cell-by-cell swap construction through the xlOil ``qSwap`` worksheet function (prototype).

Drives an Excel instance you already have open, so the add-in is live: an Excel started over
COM skips XLSTART add-ins, so xlOil would not be loaded in it.

1. Open ``sofr_swap_demo.xlsx`` in Excel with the add-in working (``=qVersion()`` answers).
2. From the env the add-in uses::

       python apps/excel/examples/bench_swaps.py --rows 250 --repeats 5

``Bench!D`` is recalculated with ``Range.Calculate``, which forces every cell's function to run
once. Calculation is set to manual for the run and restored afterwards. The in-process baseline
calls ``quant_toolkit.xl.api.make_swap`` directly on the same inputs: the floor, so the ratio is
what the Excel round trip costs on top.
"""
from __future__ import annotations

import argparse
import statistics
import time

import win32com.client

XL_CALC_MANUAL = -4135
COLUMN = "D"


def _excel_workbook(name: str):
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
    except Exception as e:  # noqa: BLE001 - pywintypes.com_error
        raise SystemExit(f"no running Excel found ({e}); open {name} in Excel first") from None
    for wb in excel.Workbooks:
        if wb.Name.lower() == name.lower():
            return excel, wb
    raise SystemExit(f"{name} is not open in the running Excel; open it first")


def _check(values) -> None:
    cells = [row[0] for row in values]
    bad = [v for v in cells if not (isinstance(v, str) and v.startswith("Swap::"))]
    if bad:
        raise SystemExit(f"{len(bad)}/{len(cells)} cells are not swap handles, e.g. {bad[0]!r}")


def _time_block(rng, repeats: int) -> list[float]:
    rng.Calculate()  # warm-up: imports, first-call caches
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        rng.Calculate()
        out.append(time.perf_counter() - t0)
    return out


def _baseline(inputs, as_of, repeats: int) -> list[float]:
    from quant_toolkit.xl import api

    tenor, rate, notional = inputs[0]
    api.make_swap(notional, "USD.SOFR.1D", rate, tenor, as_of)  # warm-up
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for tenor, rate, notional in inputs:
            api.make_swap(notional, "USD.SOFR.1D", rate, tenor, as_of)
        out.append(time.perf_counter() - t0)
    return out


def _report(label: str, times: list[float], rows: int, ref: float | None) -> None:
    med = statistics.median(times)
    rel = f"{med / ref:7.1f}x" if ref else "   base"
    print(f"{label:<16} {med * 1e3:10.1f} ms {med / rows * 1e6:10.1f} us/cell {rel}   (min {min(times) * 1e3:.1f} ms)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workbook", default="sofr_swap_demo.xlsx", help="name of the open workbook")
    p.add_argument("--rows", type=int, default=250, help="Bench rows to time (sheet has 250)")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    excel, wb = _excel_workbook(args.workbook)
    sheet = wb.Worksheets("Bench")
    last = args.rows + 1
    inputs = [tuple(r) for r in sheet.Range(f"A2:C{last}").Value]
    as_of = wb.Worksheets("Demo").Range("A2").Value

    saved_calc, saved_screen = excel.Calculation, excel.ScreenUpdating
    excel.Calculation, excel.ScreenUpdating = XL_CALC_MANUAL, False
    try:
        rng = sheet.Range(f"{COLUMN}2:{COLUMN}{last}")
        xl_times = _time_block(rng, args.repeats)
        _check(rng.Value)
    finally:
        excel.Calculation, excel.ScreenUpdating = saved_calc, saved_screen

    base = _baseline(inputs, as_of, args.repeats)
    ref = statistics.median(base)
    print(f"\n{args.rows} swaps, median of {args.repeats} recalcs\n")
    _report("python (no xl)", base, args.rows, None)
    _report("xlOil qSwap", xl_times, args.rows, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
