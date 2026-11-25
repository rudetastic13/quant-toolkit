"""``QuantToolkit.Curves.*`` worksheet functions (xlOil): read curves out of a calibrated market.

``Curve`` is a ``Calibration`` handle. ``CurveKey`` picks a curve in its market and defaults to
the one that calibration solved, so the result of a second calibration can still read the first
curve. Dates take a cell or a range and the result has the same shape.
"""
from __future__ import annotations

import numpy as np

from quant_toolkit.finance.dates import Date
from quant_toolkit.finance.markets.rate_generator import RateGenerator

from quant_toolkit.xl.core import handles
from quant_toolkit.xl.core.register import xl_func
from quant_toolkit.xl.solver import market_of


def _dates(values) -> tuple[np.ndarray, tuple | None]:
    """Excel date cells as a flat date array, remembering the shape they came in.

    Parameters
    ----------
    values : float or ndarray
        One Excel date serial, or a 2-D range of them.

    Returns
    -------
    dates : ndarray of datetime64[D], shape (n,)
        The dates, row by row.
    shape : tuple or None
        The range's shape, or ``None`` if ``values`` was a single cell.
    """
    if isinstance(values, np.ndarray):
        flat = [Date.from_excel(v).to_numpy() for v in values.ravel()]
        return np.array(flat, dtype="datetime64[D]"), values.shape
    return np.array([Date.from_excel(values).to_numpy()], dtype="datetime64[D]"), None


def _shaped(values: np.ndarray, shape: tuple | None):
    """Give a flat result the shape its dates came in.

    Parameters
    ----------
    values : ndarray of float64, shape (n,)
    shape : tuple or None
        As returned by :func:`_dates`.

    Returns
    -------
    float or ndarray of float64
        A scalar for a single cell, otherwise an array of ``shape``.
    """
    return float(values[0]) if shape is None else np.asarray(values, dtype=np.float64).reshape(shape)


def _curve(Curve, CurveKey):  # noqa: N803
    """The market behind a Calibration handle and the key to read from it.

    Parameters
    ----------
    Curve : str
        A Calibration handle.
    CurveKey : str or None
        A curve key in that market; the calibrated key if ``None``.

    Returns
    -------
    market : MarketContext
    key : str
    """
    market, key = market_of(handles.load(Curve), CurveKey)
    return market, key


@xl_func("Curves")
def DiscountFactor(Curve, Dates, CurveKey=None):  # noqa: N802, N803
    """Discount factor from the curve's as-of date to a date.

    Parameters
    ----------
    Curve : str
        A Calibration handle from QuantToolkit.Solver.Calibrate.
    Dates : date or range
        The date to discount from, or a range of them.
    CurveKey : str, optional
        Which curve in the market to read, e.g. "USD.SOFR.1D". The calibrated curve if omitted.

    Returns
    -------
    float or range
        Discount factors in the shape of Dates.
    """
    market, key = _curve(Curve, CurveKey)
    dates, shape = _dates(Dates)
    return _shaped(market[key].discount_factor(dates), shape)


@xl_func("Curves")
def ZeroRate(Curve, Dates, CurveKey=None):  # noqa: N802, N803
    """Continuously compounded Act/365 zero rate from the curve's as-of date to a date.

    Parameters
    ----------
    Curve : str
        A Calibration handle from QuantToolkit.Solver.Calibrate.
    Dates : date or range
        The maturity date, or a range of them.
    CurveKey : str, optional
        Which curve in the market to read, e.g. "USD.SOFR.1D". The calibrated curve if omitted.

    Returns
    -------
    float or range
        Zero rates as decimals, in the shape of Dates.
    """
    market, key = _curve(Curve, CurveKey)
    dates, shape = _dates(Dates)
    return _shaped(market[key].zero_rate(dates), shape)


@xl_func("Curves")
def ForwardRate(Curve, StartDates, EndDates, CurveKey=None):  # noqa: N802, N803
    """Simple forward rate of the index between two dates, on the index's day count.

    Parameters
    ----------
    Curve : str
        A Calibration handle from QuantToolkit.Solver.Calibrate.
    StartDates : date or range
        Start of each accrual period.
    EndDates : date or range
        End of each accrual period; the same number of cells as StartDates.
    CurveKey : str, optional
        Which projection curve to read, e.g. "USD.SOFR.1D". The calibrated curve if omitted.

    Returns
    -------
    float or range
        Forward rates as decimals, in the shape of StartDates.

    Raises
    ------
    ValueError
        If StartDates and EndDates hold a different number of cells.
    """
    market, key = _curve(Curve, CurveKey)
    starts, shape = _dates(StartDates)
    ends, end_shape = _dates(EndDates)
    if starts.size != ends.size:
        raise ValueError(f"{starts.size} start dates but {ends.size} end dates")
    return _shaped(RateGenerator(market).simple_rate(key, starts, ends), shape if shape is not None else end_shape)


@xl_func("Curves")
def Keys(Curve):  # noqa: N802, N803
    """The curve keys registered in a calibrated market, as a column.

    Parameters
    ----------
    Curve : str
        A Calibration handle from QuantToolkit.Solver.Calibrate.

    Returns
    -------
    column of str
        Every key the market answers to, sorted, e.g. "USD.SOFR.1D" and "USD.STDCSA".
    """
    market, _ = _curve(Curve, None)
    return np.array(sorted(market), dtype=object).reshape(-1, 1)
