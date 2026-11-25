"""``QuantToolkit.Solver.*`` worksheet functions (xlOil): market quotes in, a calibrated curve out.

One quote function per instrument type, each returning a handle; ``Calibrate`` takes a range
(or a ``List`` handle) of them and returns a ``Calibration`` handle. That handle carries the
market with the solved curve registered, so the ``QuantToolkit.Curves`` and
``QuantToolkit.Pricing`` functions take it directly, and so does a later ``Calibrate`` that
builds a second curve on top of it.

Quotes are dated like the swaps: ``Effective`` is a date or a term from ``AsOf`` (``"2BD"``),
``Maturity`` a date or a term from the effective date (``"5Y"``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_toolkit.finance.conventions.keys import CurveKey as _Key
from quant_toolkit.finance.dates import Date
from quant_toolkit.finance.markets.context import MarketContext
from quant_toolkit.finance.markets.curves import CurveInterpolator
from quant_toolkit.finance.pricing import calibration as cal

from quant_toolkit.xl.core import handles
from quant_toolkit.xl.core.convert import date_or_term, enum_member, given
from quant_toolkit.xl.core.register import xl_func

_EPOCH_SERIAL = Date.fromordinal(0).to_excel()
_SOLVERS = {"global": cal.GlobalSolver, "bootstrap": cal.Bootstrapper}


@dataclass(frozen=True)
class CalibrationState:
    """What a ``Calibration`` handle holds.

    Attributes
    ----------
    key : str
        The curve key that was solved (``"USD.SOFR.1D"``); the default the curve functions read.
    result : CalibrationResult
        The library result. ``result.market`` is the market with the solved curve registered,
        which is what pricing, risk and a later calibration use.
    quotes : tuple of CalibrationInstrument
        The quotes the curve was solved to, in pillar-date order, so they line up with
        ``result.pillar_dates`` and ``result.residuals``.
    """

    key: str
    result: cal.CalibrationResult
    quotes: tuple


def market_of(source: object, curve_key: object = None) -> tuple[MarketContext, str | None]:
    """The market behind a ``Calibration`` handle's object, and the curve key to read from it.

    Parameters
    ----------
    source : CalibrationState
        The object a ``Calibration`` handle loads to.
    curve_key : str, optional
        A curve key in that market. Defaults to the key that calibration solved.

    Returns
    -------
    market : MarketContext
    key : str
        ``curve_key`` in canonical form, or the calibrated key.

    Raises
    ------
    TypeError
        If ``source`` is not a ``CalibrationState`` (the cell passed was not a Calibration handle).
    """
    if not isinstance(source, CalibrationState):
        raise TypeError(f"expected a Calibration handle, got {type(source).__name__}")
    return source.result.market, str(_Key.parse(str(curve_key))) if given(curve_key) else source.key


def _as_of(value) -> Date | None:
    """An optional as-of cell as a ``Date``.

    Parameters
    ----------
    value : float or None
        An Excel date serial, or ``None`` / an empty string for an omitted argument.

    Returns
    -------
    Date or None
    """
    return Date.from_excel(value) if given(value) else None


def _dating(Effective, Maturity, AsOf) -> dict:  # noqa: N803
    """The quote functions' three dating cells as keyword arguments for the library helpers.

    Parameters
    ----------
    Effective, Maturity : float or str
        An Excel date serial, or a term string.
    AsOf : float or None
        An Excel date serial, or ``None``.

    Returns
    -------
    dict
        ``effective`` and ``maturity`` as ``Date`` or ``Term``, ``as_of`` as ``Date`` or ``None``.
    """
    return {"effective": date_or_term(Effective), "maturity": date_or_term(Maturity), "as_of": _as_of(AsOf)}


def _index(RateIndex) -> dict:  # noqa: N803
    """A rate index cell as the ``rate_index`` and ``currency`` keyword arguments.

    Parameters
    ----------
    RateIndex : str
        A projection curve key such as ``"USD.SOFR.1D"``.

    Returns
    -------
    dict
        ``rate_index`` (the key, stripped) and ``currency`` (its first component).
    """
    index = str(RateIndex).strip()
    return {"rate_index": index, "currency": _Key.parse(index).currency}


def _store(tag: str, helper, contract, *extra) -> str:
    """Store a calibration helper and return its handle.

    Parameters
    ----------
    tag : str
        The handle tag, ``"SwapQuote"`` and so on.
    helper : CalibrationInstrument
        The library helper to store.
    contract : Serializable
        The instrument inside the helper; its JSON keys the handle.
    *extra : object
        Further inputs that distinguish two helpers on the same contract.

    Returns
    -------
    str
        ``<tag>;;;<digest>``, the digest taken over the contract, the quoted value and ``extra``.
    """
    return handles.store(tag, helper, contract.to_json(indent=None), helper.quote.value, *extra)


@xl_func("Solver")
def DepositQuote(Rate, RateIndex, Effective, Maturity, AsOf=None):  # noqa: N802, N803
    """Cash deposit quote for calibration, quoted as a simple rate.

    Returns a DepositQuote handle for Calibrate or RiskTemplate.

    Parameters
    ----------
    Rate : float
        The deposit rate as a decimal (0.043 for 4.30%).
    RateIndex : str
        The index the deposit is quoted on, as a curve key, e.g. "USD.SOFR.1D".
    Effective : date or str
        Start date, or a term from AsOf such as "2BD".
    Maturity : date or str
        End date, or a term from the effective date such as "3M".
    AsOf : date, optional
        Date a term Effective is counted from. Today if omitted; not used when Effective is a date.

    Returns
    -------
    str
        A DepositQuote handle. Its curve pillar is the maturity date.

    Examples
    --------
    ``=QuantToolkit.Solver.DepositQuote(0.043, "USD.SOFR.1D", "2BD", "3M", $B$1)``
    """
    helper = cal.deposit_helper(rate=float(Rate), **_index(RateIndex), **_dating(Effective, Maturity, AsOf))
    return _store("DepositQuote", helper, helper.deposit)


@xl_func("Solver")
def FraQuote(Rate, RateIndex, Effective, Maturity, AsOf=None):  # noqa: N802, N803
    """Forward rate agreement quote for calibration, quoted as a simple forward rate.

    Returns a FraQuote handle for Calibrate or RiskTemplate.

    Parameters
    ----------
    Rate : float
        The forward rate as a decimal (0.0418 for 4.18%).
    RateIndex : str
        The index the FRA settles on, as a curve key, e.g. "USD.SOFR.1D".
    Effective : date or str
        Start of the forward period: a date, or a term from AsOf such as "6M".
    Maturity : date or str
        End of the forward period: a date, or a term from the effective date such as "3M".
    AsOf : date, optional
        Date a term Effective is counted from. Today if omitted; not used when Effective is a date.

    Returns
    -------
    str
        A FraQuote handle. Its curve pillar is the end of the forward period.

    Examples
    --------
    ``=QuantToolkit.Solver.FraQuote(0.0418, "USD.SOFR.1D", "6M", "3M", $B$1)`` is a 6x9 FRA.
    """
    helper = cal.fra_helper(rate=float(Rate), **_index(RateIndex), **_dating(Effective, Maturity, AsOf))
    return _store("FraQuote", helper, helper.fra)


@xl_func("Solver")
def SwapQuote(Rate, RateIndex, Effective, Maturity, AsOf=None):  # noqa: N802, N803
    """Par swap quote for calibration, quoted as its fixed rate.

    Returns a SwapQuote handle for Calibrate or RiskTemplate.

    Parameters
    ----------
    Rate : float
        The par fixed rate as a decimal (0.038 for 3.80%). Not used by RiskTemplate, so 0 will do there.
    RateIndex : str
        The floating index as a curve key, e.g. "USD.SOFR.1D". The swap takes its conventions.
    Effective : date or str
        Start date, or a term from AsOf such as "2BD".
    Maturity : date or str
        End date, or a term from the effective date such as "5Y".
    AsOf : date, optional
        Date a term Effective is counted from. Today if omitted; not used when Effective is a date.

    Returns
    -------
    str
        A SwapQuote handle. Its curve pillar is the swap's maturity date.

    Examples
    --------
    ``=QuantToolkit.Solver.SwapQuote(0.038, "USD.SOFR.1D", "2BD", "5Y", $B$1)``
    """
    helper = cal.swap_helper(rate=float(Rate), **_index(RateIndex), **_dating(Effective, Maturity, AsOf))
    return _store("SwapQuote", helper, helper.swap)


@xl_func("Solver")
def SpreadQuote(Spread, RateIndex, OverIndex, Effective, Maturity, AsOf=None):  # noqa: N802, N803
    """Basis quote for calibration: one index's par swap rate minus another's, on the same dates.

    Returns a SpreadQuote handle. Used to solve RateIndex as a spread curve over OverIndex.

    Parameters
    ----------
    Spread : float
        Par rate of the RateIndex swap minus par rate of the OverIndex swap, as a decimal (0.0005 for 5bp).
    RateIndex : str
        The index being calibrated, as a curve key.
    OverIndex : str
        The index the spread is quoted over, as a curve key. Its curve must already be in the market.
    Effective : date or str
        Start date, or a term from AsOf such as "2BD".
    Maturity : date or str
        End date, or a term from the effective date such as "5Y".
    AsOf : date, optional
        Date a term Effective is counted from. Today if omitted; not used when Effective is a date.

    Returns
    -------
    str
        A SpreadQuote handle. Its curve pillar is the swaps' maturity date.

    Notes
    -----
    Pass the first curve's Calibration handle as Market, and OverIndex as SpreadOver, when
    calibrating with these quotes.
    """
    over = str(OverIndex).strip()
    helper = cal.spread_helper(
        spread=float(Spread), over=over, **_index(RateIndex), **_dating(Effective, Maturity, AsOf)
    )
    return _store("SpreadQuote", helper, helper.swap.swap, over)


@xl_func("Solver")
def FuturesQuote(Price, RateIndex, Contract, Month, AsOf=None, Convexity=None):  # noqa: N802, N803
    """SOFR futures quote for calibration, quoted as a price.

    Returns a FuturesQuote handle for Calibrate or RiskTemplate.

    Parameters
    ----------
    Price : float
        The futures price, e.g. 96.25 (an implied rate of 3.75%).
    RateIndex : str
        The index the contract settles on, as a curve key, e.g. "USD.SOFR.1D".
    Contract : str
        The contract code: "SR1" (one-month) or "SR3" (three-month).
    Month : str
        The contract month, as a futures code ("Z26") or year and month ("2026-12").
    AsOf : date, optional
        The valuation date. Today if omitted.
    Convexity : float, optional
        Convexity adjustment subtracted from the futures rate, as a decimal. Zero if omitted.

    Returns
    -------
    str
        A FuturesQuote handle. Its curve pillar is the end of the contract's reference period.
    """
    convexity = float(Convexity) if given(Convexity) else 0.0
    helper = cal.futures_helper(
        price=float(Price),
        contract=str(Contract).strip(),
        month=str(Month).strip(),
        as_of=_as_of(AsOf) or Date.today(),
        convexity=convexity,
        **_index(RateIndex),
    )
    return _store("FuturesQuote", helper, helper.future, convexity)


def _quotes(values) -> list:
    """Quote handles from a cell, a range, or ``List`` handles, as library helpers.

    Parameters
    ----------
    values : str or ndarray
        A handle, or a 2-D range of handles. A ``List`` handle is expanded in place; empty
        cells are skipped.

    Returns
    -------
    list of CalibrationInstrument
        In the order given.

    Raises
    ------
    TypeError
        If any element is not a calibration quote.
    """
    out: list = []
    for item in handles.flatten(values):
        if isinstance(item, list):
            out.extend(item)
        elif given(item):
            out.append(item)
    bad = [q for q in out if not isinstance(q, cal.CalibrationInstrument)]
    if bad:
        raise TypeError(f"{bad[0]!r} is not a quote: build the range from the QuantToolkit.Solver quote functions")
    return out


@xl_func("Solver")
def Calibrate(  # noqa: N802
    Instruments,
    CurveKey,
    AsOf=None,
    Interpolation=None,
    Solver=None,
    DiscountKeys=None,
    Market=None,
    SpreadOver=None,
):
    """Solve one curve from quote handles.

    Returns a Calibration handle: the curve and the market holding it. The Curves, Pricing and
    Risk functions take that handle, and so does a later Calibrate that adds a second curve.

    Parameters
    ----------
    Instruments : range
        Quote handles from the Solver quote functions (a range, or a List handle). One pillar per quote.
    CurveKey : str
        The curve to solve, e.g. "USD.SOFR.1D".
    AsOf : date, optional
        The valuation date. Today if omitted; ignored when Market is given.
    Interpolation : str, optional
        "LogLinearDF" (default), "LogCubicDF", "RateLinear", "RateQuadratic" or "RateCubic".
    Solver : str, optional
        "Global" (default; all pillars at once) or "Bootstrap" (one pillar at a time; local interpolations only).
    DiscountKeys : range, optional
        Discount keys this curve also serves. If omitted, a first curve also serves the currency's STDCSA key.
    Market : str, optional
        An earlier Calibration handle to build on, so this curve can depend on curves already solved.
    SpreadOver : str, optional
        Solve this curve as a spread over that key, which must be in Market. Used with SpreadQuote.

    Returns
    -------
    str
        A Calibration handle ("Calibration;;;3f1c0a9d2b77").

    Raises
    ------
    TypeError
        If Instruments holds something that is not a quote handle.
    ValueError
        If Solver is not "Global" or "Bootstrap", or the interpolation name is unknown.

    Notes
    -----
    Swaps discount on the currency's standard CSA key (``USD.STDCSA``). Registering the first
    curve under that key as well is what lets the same handle price swaps; a second curve in
    the same currency leaves the existing discount curve in place.

    Examples
    --------
    ``=QuantToolkit.Solver.Calibrate(E10:E19, "USD.SOFR.1D", $B$1)``
    """
    helpers = _quotes(Instruments)
    key = str(_Key.parse(str(CurveKey).strip()))

    if given(Market):
        market, _ = market_of(handles.load(Market))
    else:
        market = MarketContext(as_of=_as_of(AsOf) or Date.today())

    interpolation = enum_member(CurveInterpolator, Interpolation if given(Interpolation) else "LogLinearDF")
    solver_name = str(Solver).strip().lower() if given(Solver) else "global"
    if solver_name not in _SOLVERS:
        raise ValueError(f"unknown solver {Solver!r}: use one of {sorted(_SOLVERS)}")

    if given(DiscountKeys):
        discount_keys = tuple(str(k).strip() for k in handles.flatten(DiscountKeys) if given(k))
    else:
        # a first curve also funds its own quotes, so it serves the standard discount key too
        standard = f"{_Key.parse(key).currency}.STDCSA"
        discount_keys = () if standard in market or given(SpreadOver) else (standard,)

    target = cal.CurveDefinition(key, interpolation, discount_keys=discount_keys)
    if given(SpreadOver) and key not in market:
        over = str(SpreadOver).strip()
        market = market.with_curve(key, market[over].zero_curve + target.flat_on_pillars(market, helpers))

    result = cal.CurveCalibrator(helpers, _SOLVERS[solver_name](), target).calibrate(market)
    ordered = tuple(sorted(helpers, key=lambda h: h.pillar_date))
    return handles.store(
        "Calibration", CalibrationState(key, result, ordered),
        Instruments, key, market.as_of.to_str(), interpolation.name, solver_name, discount_keys, Market, SpreadOver,
    )


@xl_func("Solver")
def CalibrationReport(Calibration):  # noqa: N802, N803
    """Fit of a calibration, one row per pillar.

    Parameters
    ----------
    Calibration : str
        A Calibration handle from Calibrate.

    Returns
    -------
    table
        A header row, then per quote in pillar order: Pillar (date), Quote, Implied (what the
        solved curve gives), Error (bp) (implied minus quote), DiscountFactor and ZeroRate at the pillar.
    """
    calibration = handles.load(Calibration, "Calibration")
    result = calibration.result
    curve = result.market[calibration.key]
    pillars = result.pillar_dates
    rows: list[list] = [["Pillar", "Quote", "Implied", "Error (bp)", "DiscountFactor", "ZeroRate"]]
    dfs, zeros = curve.discount_factor(pillars), curve.zero_rate(pillars)
    for i, quote in enumerate(calibration.quotes):
        implied = quote.implied(result.market)
        rows.append([
            float(pillars[i].astype("datetime64[D]").astype(np.int64) + _EPOCH_SERIAL),
            float(quote.quote.value),
            float(implied),
            float(result.residuals[i] * 1e4),
            float(dfs[i]),
            float(zeros[i]),
        ])
    return np.array(rows, dtype=object)
