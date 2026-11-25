"""``QuantToolkit.Pricing.*`` worksheet functions (xlOil): price swap handles off a calibrated market.

``Portfolio`` compiles a set of swaps once and returns a handle; ``Calculate`` asks that
portfolio for any list of measures against a market, one column each. ``PresentValue`` and
``DV01`` are the single-measure shortcuts, and take either a ``Portfolio`` handle or the swap
handles themselves.
"""
from __future__ import annotations

import numpy as np

from quant_toolkit.finance.pricing.risk import RiskEngine

from quant_toolkit.xl.solver import market_of
from quant_toolkit.xl.core import book, handles
from quant_toolkit.xl.core.convert import given
from quant_toolkit.xl.core.register import xl_func


def _column(values) -> np.ndarray:
    """A per-position result as the column Excel spills.

    Parameters
    ----------
    values : array_like of float, shape (n,)

    Returns
    -------
    ndarray of float64, shape (n, 1)
    """
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


@xl_func("Pricing")
def Portfolio(Instruments):  # noqa: N802, N803
    """Compile a set of swaps once, for repeated pricing and risk.

    Returns a Portfolio handle that Calculate, PresentValue, DV01 and RiskReport accept.

    Parameters
    ----------
    Instruments : range
        Swap handles from the IRProducts functions: one cell, a range, or a List handle. Empty cells are skipped.

    Returns
    -------
    str
        A Portfolio handle ("Portfolio;;;3963abe45079"). The same swaps in the same order give the same handle.

    Raises
    ------
    TypeError
        If an element is not a swap.
    ValueError
        If there are no swaps.

    Notes
    -----
    The portfolio is independent of the market: one handle can be priced against any number
    of calibrations without recompiling.
    """
    compiled = book.compile_book(Instruments)
    return handles.store("Portfolio", compiled, *(s.to_json(indent=None) for s in compiled.swaps))


@xl_func("Pricing")
def Calculate(Portfolio, Market, Measures, CurveKey=None, RiskTemplate=None):  # noqa: N802, N803
    """Requested measures for each position of a portfolio.

    Returns a table: a header row, then one row per position, one or more columns per measure.

    Parameters
    ----------
    Portfolio : str or range
        A Portfolio handle, or swap handles (compiled on the spot).
    Market : str
        A Calibration handle.
    Measures : str or range
        Measure names: PV, LegPV, DV01, KeyRate01, Quote01, Unexplained01. Columns follow the order given.
    CurveKey : str, optional
        The curve DV01 and KeyRate01 bump. The calibrated curve if omitted.
    RiskTemplate : str, optional
        A RiskTemplate handle for Quote01 and Unexplained01. The calibration's own quotes if omitted.

    Returns
    -------
    table
        ``PV``: present value. ``LegPV``: ReceiveLegPV and PayLegPV. ``DV01``: PV change for
        +1bp on every pillar of the curve. ``KeyRate01``: one column per pillar. ``Quote01``:
        one column per template quote, PV change for +1bp on that quote. ``Unexplained01``:
        pillar risk the template does not explain, summed.

    Raises
    ------
    ValueError
        If no measure is given or a name is not recognised.

    Notes
    -----
    All 01 measures are the PV change for rates up one basis point, so a receiver is negative.

    Examples
    --------
    ``=QuantToolkit.Pricing.Calculate(B7, B6, {"PV","DV01"})``
    """
    state = handles.load(Market)
    market, key = market_of(state, CurveKey)
    names = [str(m) for m in handles.flatten(Measures) if given(m)]
    if not names:
        raise ValueError(f"no measures requested: use {book.MEASURE_NAMES}")
    return book.calculate(book.book_of(Portfolio), market, names, key, book.template_of(RiskTemplate, state))


@xl_func("Pricing")
def PresentValue(Instruments, Market):  # noqa: N802, N803
    """Present value of each swap, as a column.

    Parameters
    ----------
    Instruments : str or range
        A Portfolio handle, a Swap handle, a range of Swap handles, or a List handle.
    Market : str
        A Calibration handle.

    Returns
    -------
    column of float
        One value per swap, in the order given, in the swap's currency.
    """
    market, _ = market_of(handles.load(Market))
    return _column(book.book_of(Instruments).program.price(market).instrument_pv)


@xl_func("Pricing")
def DV01(Instruments, Market, CurveKey=None):  # noqa: N802, N803
    """PV change of each swap for rates up one basis point, as a column.

    Parameters
    ----------
    Instruments : str or range
        A Portfolio handle, a Swap handle, a range of Swap handles, or a List handle.
    Market : str
        A Calibration handle.
    CurveKey : str, optional
        Bump only that curve, e.g. "USD.SOFR.1D". Every curve in the market if omitted.

    Returns
    -------
    column of float
        One value per swap, in the order given. A receiver is negative, a payer positive.

    Notes
    -----
    The bump is +1bp on the zero rate at every pillar, computed analytically (not by repricing).
    """
    market, key = market_of(handles.load(Market), CurveKey)
    engine = RiskEngine(book.book_of(Instruments).program, market)
    return _column(book.dv01(engine, key if given(CurveKey) else None))
