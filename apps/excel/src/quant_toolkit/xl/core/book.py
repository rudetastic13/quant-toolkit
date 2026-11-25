"""The compiled book behind a ``Portfolio`` handle, and the measures that can be requested of it.

A book is compiled once and repriced against any market, so every measure in a request shares
one program, and a sheet that reprices the same swaps under several calibrations compiles once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_toolkit.finance.dates import Date
from quant_toolkit.finance.instruments import resolution
from quant_toolkit.finance.instruments.priceable import compile_instruments
from quant_toolkit.finance.pricing.risk import RiskEngine, RiskTemplate

from quant_toolkit.xl.core import handles
from quant_toolkit.xl.core.convert import given

_EPOCH_SERIAL = Date.fromordinal(0).to_excel()
BP = 1e-4
MEASURE_NAMES = "PV, LegPV, DV01, KeyRate01, Quote01, Unexplained01"


@dataclass(frozen=True)
class Book:
    """What a ``Portfolio`` handle holds.

    Attributes
    ----------
    swaps : tuple of Swap
        The positions, in the order given; every per-position result is in this order.
    program : PricingProgram
        The swaps compiled once by their registered pricer; repriced against any market.
    """

    swaps: tuple
    program: object


def swaps_of(values) -> list:
    """Swaps from a cell, a range, or ``List`` handles.

    Parameters
    ----------
    values : str or ndarray
        A Swap handle, or a 2-D range of them. A ``List`` handle is expanded in place; empty
        cells are skipped.

    Returns
    -------
    list of Swap
        In the order given.

    Raises
    ------
    TypeError
        If any element is not a ``Swap``.
    ValueError
        If nothing is left after skipping empty cells.
    """
    swaps: list = []
    for item in handles.flatten(values):
        if isinstance(item, list):
            swaps.extend(item)
        elif given(item):
            swaps.append(item)
    bad = [s for s in swaps if not isinstance(s, resolution.Swap)]
    if bad:
        raise TypeError(f"{type(bad[0]).__name__} is not a swap: pass handles from QuantToolkit.IRProducts")
    if not swaps:
        raise ValueError("no swaps to price")
    return swaps


def compile_book(values) -> Book:
    """Compile swap handles into a book.

    Parameters
    ----------
    values : str or ndarray
        As for :func:`swaps_of`.

    Returns
    -------
    Book
    """
    swaps = swaps_of(values)
    return Book(tuple(swaps), compile_instruments(swaps))


def book_of(values) -> Book:
    """The book behind a worksheet argument.

    Parameters
    ----------
    values : str or ndarray
        A ``Portfolio`` handle, or anything :func:`swaps_of` accepts.

    Returns
    -------
    Book
        The stored book for a ``Portfolio`` handle; otherwise one compiled on the spot.
    """
    if isinstance(values, str) and values.startswith(f"Portfolio{handles.SEP}"):
        return handles.load(values)
    return compile_book(values)


def serials(dates) -> list[float]:
    """Dates as Excel serial numbers.

    Parameters
    ----------
    dates : array_like of datetime64

    Returns
    -------
    list of float
    """
    return [float(d) for d in np.asarray(dates).astype("datetime64[D]").astype(np.int64) + _EPOCH_SERIAL]


def template_of(value, state) -> RiskTemplate:
    """The risk template behind an optional worksheet argument.

    Parameters
    ----------
    value : str or None
        A ``RiskTemplate`` handle, or ``None`` / an empty string if omitted.
    state : CalibrationState
        The calibration being reported against.

    Returns
    -------
    RiskTemplate
        The stored template, or the identity template over ``state.quotes`` if ``value`` was omitted.
    """
    if given(value):
        return handles.load(value, "RiskTemplate")
    return RiskTemplate.identity(state.quotes, name="calibration quotes")


def dv01(engine: RiskEngine, key: str | None) -> np.ndarray:
    """Parallel DV01 per position.

    Parameters
    ----------
    engine : RiskEngine
        Bound to the book's program and the market.
    key : str or None
        Bump only this curve's pillars; every curve block in the market if ``None``.

    Returns
    -------
    ndarray of float64, shape (n_positions,)
        PV change for +1bp on each bumped pillar's zero rate.
    """
    if key is not None:
        return engine.dv01(key)
    return engine.pillar_risk().values.sum(axis=1) * BP


# -- measures: (book, market, engine, key, template) -> (column headers, values (N, k)) -------------------


def _pv(book, market, engine, key, template):
    """Measure ``PV``: present value per position.

    Every measure has this signature. Parameters: ``book`` (Book), ``market`` (MarketContext),
    ``engine`` (RiskEngine on that book and market, carrying the template), ``key`` (the curve
    key for the 01 measures), ``template`` (RiskTemplate).

    Returns
    -------
    headers : list of str
        One per column.
    values : ndarray of float64, shape (n_positions, len(headers))
    """
    return ["PV"], book.program.price(market).instrument_pv.reshape(-1, 1)


def _leg_pv(book, market, engine, key, template):
    """Measure ``LegPV``: present value of the receive leg and of the pay leg; see :func:`_pv`."""
    return ["ReceiveLegPV", "PayLegPV"], np.asarray(book.program.price(market).leg_pv).reshape(len(book.swaps), 2)


def _dv01(book, market, engine, key, template):
    """Measure ``DV01``: PV change for +1bp on every pillar of ``key``; see :func:`_pv`."""
    return ["DV01"], dv01(engine, key).reshape(-1, 1)


def _key_rate(book, market, engine, key, template):
    """Measure ``KeyRate01``: one column per pillar of ``key``, headed ``KR <date>``; see :func:`_pv`."""
    report = engine.ladder(key)
    return [f"KR {d}" for d in np.datetime_as_string(report.pillar_dates, unit="D")], report.ladder


def _quote(book, market, engine, key, template):
    """Measure ``Quote01``: one column per template quote, PV change for +1bp on it; see :func:`_pv`."""
    risk = engine.quote_risk()
    return list(risk.labels), risk.dv01


def _unexplained(book, market, engine, key, template):
    """Measure ``Unexplained01``: pillar risk the template leaves, summed over pillars; see :func:`_pv`."""
    return ["Unexplained01"], (engine.quote_risk().unexplained_sum * BP).reshape(-1, 1)


MEASURES = {
    "pv": _pv,
    "legpv": _leg_pv,
    "dv01": _dv01,
    "keyrate01": _key_rate,
    "quote01": _quote,
    "unexplained01": _unexplained,
}


def calculate(book: Book, market, names: list[str], key: str, template: RiskTemplate) -> np.ndarray:
    """Evaluate a list of measures for a book against a market.

    Parameters
    ----------
    book : Book
    market : MarketContext
    names : list of str
        Measure names, any case; see ``MEASURES``.
    key : str
        The curve key the 01 measures bump.
    template : RiskTemplate
        For ``Quote01`` and ``Unexplained01``.

    Returns
    -------
    ndarray of object, shape (n_positions + 1, n_columns)
        Row 0 is the column headers; the measures follow left to right in the order of ``names``.

    Raises
    ------
    ValueError
        If a name is not in ``MEASURES``.
    """
    engine = RiskEngine(book.program, market, template=template)
    headers: list = []
    blocks: list[np.ndarray] = []
    for name in names:
        try:
            measure = MEASURES[name.strip().lower()]
        except KeyError:
            raise ValueError(f"unknown measure {name!r}: use {MEASURE_NAMES}") from None
        head, values = measure(book, market, engine, key, template)
        headers.extend(head)
        blocks.append(np.asarray(values, dtype=np.float64))
    table = np.empty((len(book.swaps) + 1, len(headers)), dtype=object)
    table[0] = headers
    table[1:] = np.hstack(blocks)
    return table
