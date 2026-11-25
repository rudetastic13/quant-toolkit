"""Time cell-by-cell swap construction through ``QuantToolkit.IRProducts.OvernightIndexSwap`` (prototype).

Drives an Excel instance you already have open, so the add-in is live: an Excel started over
COM skips XLSTART add-ins, so xlOil would not be loaded in it.

1. Open ``sofr_swap_demo.xlsx`` in Excel with the add-in working
   (``=QuantToolkit.Utilities.Version()`` answers).
2. From the env the add-in uses::

       python apps/excel/examples/bench_swaps.py --rows 250 --repeats 5

``Bench!D`` is recalculated with ``Range.Calculate``, which forces every cell's function to run
once. Calculation is set to manual for the run and restored afterwards. The in-process baseline
calls the library's ``Swap.fixed_float_swap`` directly on the same inputs: the floor, so the ratio
is what the Excel round trip costs on top.
"""
from __future__ import annotations

import argparse
import statistics
import time

import win32com.client

XL_CALC_MANUAL = -4135
COLUMN = "D"


def _excel_workbook(name: str):
    """Attach to the running Excel and find an open workbook by name.

    Parameters
    ----------
    name : str
        The workbook's file name, e.g. ``"sofr_swap_demo.xlsx"``; case-insensitive.

    Returns
    -------
    excel, workbook
        The COM ``Excel.Application`` and ``Workbook`` objects.

    Raises
    ------
    SystemExit
        If Excel is not running or the workbook is not open in it.
    """
    try:
        excel = win32com.client.GetActiveObject("Excel.Application")
    except Exception as e:  # noqa: BLE001 - pywintypes.com_error
        raise SystemExit(f"no running Excel found ({e}); open {name} in Excel first") from None
    for wb in excel.Workbooks:
        if wb.Name.lower() == name.lower():
            return excel, wb
    raise SystemExit(f"{name} is not open in the running Excel; open it first")


def _check(values) -> None:
    """Confirm the timed cells produced swap handles and not errors.

    Parameters
    ----------
    values : tuple of tuple
        ``Range.Value`` of the timed column: one single-element row per cell.

    Raises
    ------
    SystemExit
        If any cell is not a ``Swap;;;`` handle, so a run that timed error cells is not reported.
    """
    cells = [row[0] for row in values]
    bad = [v for v in cells if not (isinstance(v, str) and v.startswith("Swap;;;"))]
    if bad:
        raise SystemExit(f"{len(bad)}/{len(cells)} cells are not swap handles, e.g. {bad[0]!r}")


def _time_block(rng, repeats: int) -> list[float]:
    """Time recalculating a range.

    Parameters
    ----------
    rng : Range
        The COM range to recalculate.
    repeats : int
        Timed recalculations, after one untimed warm-up.

    Returns
    -------
    list of float
        Seconds per recalculation of the whole range.
    """
    rng.Calculate()  # warm-up: imports, first-call caches
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        rng.Calculate()
        out.append(time.perf_counter() - t0)
    return out


def _baseline(inputs, as_of, repeats: int) -> list[float]:
    """Time building the same swaps directly in Python, without Excel.

    Parameters
    ----------
    inputs : list of tuple
        ``(tenor, rate, notional)`` per row, read from the Bench sheet.
    as_of : datetime
        The as-of date read from the workbook.
    repeats : int
        Timed passes over all rows, after one untimed warm-up call.

    Returns
    -------
    list of float
        Seconds per pass.
    """
    from quant_toolkit.finance.dates import Date, Term
    from quant_toolkit.finance.instruments.resolution import Swap

    as_of_date = Date.from_datetime(as_of)
    spot = Term.from_str("2BD")

    def build(tenor, rate, notional):
        """One swap through ``Swap.fixed_float_swap``, as the worksheet function builds it."""
        return Swap.fixed_float_swap(
            notional=notional, rate_index="USD.SOFR.1D", fixed_rate=rate,
            effective=spot, maturity=Term.from_str(tenor), as_of=as_of_date,
        )

    build(*inputs[0])  # warm-up
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for row in inputs:
            build(*row)
        out.append(time.perf_counter() - t0)
    return out


def _report(label: str, times: list[float], rows: int, ref: float | None) -> None:
    """Print one result line: median time, time per cell, and the ratio to the baseline.

    Parameters
    ----------
    label : str
    times : list of float
        Seconds per pass.
    rows : int
        Cells per pass.
    ref : float or None
        The baseline's median in seconds; ``None`` for the baseline line itself.
    """
    med = statistics.median(times)
    rel = f"{med / ref:7.1f}x" if ref else "   base"
    print(f"{label:<16} {med * 1e3:10.1f} ms {med / rows * 1e6:10.1f} us/cell {rel}   (min {min(times) * 1e3:.1f} ms)")


def main() -> int:
    """Run the benchmark against the open workbook and print the comparison.

    Returns
    -------
    int
        Exit code, 0 on success.

    Notes
    -----
    Excel's calculation mode and screen updating are switched off for the timed section and
    restored afterwards, including on error.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workbook", default="sofr_swap_demo.xlsx", help="name of the open workbook")
    p.add_argument("--rows", type=int, default=250, help="Bench rows to time (sheet has 250)")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    excel, wb = _excel_workbook(args.workbook)
    sheet = wb.Worksheets("Bench")
    last = args.rows + 1
    inputs = [tuple(r) for r in sheet.Range(f"A2:C{last}").Value]
    as_of = wb.Worksheets("Dates").Range("B4").Value

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
    _report("xlOil (Excel)", xl_times, args.rows, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
